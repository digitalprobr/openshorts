"""Central video-encoder selection for every ffmpeg encode call site.

FFMPEG_ENCODER env values:
  x264  (default) — CPU libx264, exact pre-GPU behavior
  nvenc           — force h264_nvenc; probed once and falls back to x264
                    (with a warning) if the GPU/driver is unavailable
  auto            — h264_nvenc when the probe succeeds, else x264

Only the codec/quality args live here; surrounding args (-movflags, -pix_fmt,
audio codecs, filters) stay at each call site.
"""
import os
import shutil
import subprocess
import threading

# Quality tiers pinning the historical libx264 settings.
QUALITY = "quality"            # was: -preset medium -crf 18
QUALITY_FAST = "quality_fast"  # was: -preset fast -crf 18
DELIVERY = "delivery"          # was: -preset fast -crf 22

_X264_ARGS = {
    QUALITY: ["-c:v", "libx264", "-preset", "medium", "-crf", "18"],
    QUALITY_FAST: ["-c:v", "libx264", "-preset", "fast", "-crf", "18"],
    DELIVERY: ["-c:v", "libx264", "-preset", "fast", "-crf", "22"],
}

# NVENC -cq is not 1:1 with x264 CRF: benchmarked on the prod GPU (RTX 4000
# Ada), cq ≈ crf + 7 lands in the same file-size ballpark, with vbr + AQ for
# quality. Presets p1-p7: p5 ≈ "medium", p4 ≈ "fast".
# -pix_fmt yuv420p is REQUIRED: with RGB input (the bgr24 rawvideo pipe from
# OpenCV) nvenc otherwise emits H.264 in gbrp/GBR colorspace, which ffmpeg
# reads fine but web players render as a magenta/green mess.
_NVENC_ARGS = {
    QUALITY: ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq",
              "-rc", "vbr", "-cq", "25", "-b:v", "0",
              "-spatial-aq", "1", "-temporal-aq", "1", "-pix_fmt", "yuv420p"],
    QUALITY_FAST: ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq",
                   "-rc", "vbr", "-cq", "25", "-b:v", "0", "-spatial-aq", "1",
                   "-pix_fmt", "yuv420p"],
    DELIVERY: ["-c:v", "h264_nvenc", "-preset", "p4",
               "-rc", "vbr", "-cq", "29", "-b:v", "0", "-spatial-aq", "1",
               "-pix_fmt", "yuv420p"],
}

# Output args that drop container/stream metadata carried over from the source
# — most notably YouTube's "produced by Google Inc." stream handler, which
# otherwise survives every re-encode (ffmpeg copies input metadata by default)
# and rides into the published clip. The per-stream specifiers are required:
# global -map_metadata -1 alone leaves the audio handler_name intact on a
# stream copy. Empty audio/video specifiers are harmless when a clip lacks that
# stream (ffmpeg ignores them, verified). Spliced in before the output filename
# at each final-artifact producer; kept out of video_encode_args() so that
# stays purely codec/quality args.
METADATA_SCRUB = ["-map_metadata", "-1", "-map_chapters", "-1",
                  "-map_metadata:s:v", "-1", "-map_metadata:s:a", "-1"]

# Loudness normalisation for the delivered clip.
#
# Without this the clip inherits whatever the source was mastered at, so a
# user's clips land anywhere: measured across real delivered clips on
# 26-jul-2026, from -13.8 LUFS on a loud upload down to -28 LUFS on a quiet
# talk. TikTok, Reels and Shorts all normalise playback to roughly -14 LUFS,
# which means the quiet ones just sound thin next to everything else in the
# feed — the loud ones aren't rewarded, the quiet ones are punished.
#
# I=-14 matches the platforms' target, LRA=11 is the usual allowance for speech.
# Applied at the clip cut, where the audio is being encoded to AAC anyway, so it
# costs nothing extra. AUDIO_NORMALIZE=0 turns it off.
#
# TP=-2.0, not the -1.5 that matches the platforms' own advice, because the
# ceiling is enforced BEFORE the AAC encode and the encoder then adds
# inter-sample peaks on top. Measured over 14 corpus clips (31-jul-2026):
#
#   TP=-1.5   peak reached +0.2 dBTP, 1 clip clipping,  8 above -1.0
#   TP=-2.0   peak reached -0.3 dBTP, 0 clipping,       5 above -1.0
#   TP=-3.0   peak reached -0.9 dBTP, 0 clipping,       1 above -1.0
#
# -3.0 also costs level: only 8 of 14 stayed inside -15..-13 LUFS versus 12 at
# -2.0, and level is what the listener notices. Two other fixes were tried and
# do NOT work, so don't reach for them again: an `alimiter` after loudnorm
# (limits sample peaks, not inter-sample, and measured WORSE at +0.7), and
# two-pass loudnorm with linear=true (+0.4, still clipping). The overshoot is
# the codec's, so the only lever is headroom.
LOUDNORM_FILTER = "loudnorm=I=-14:TP=-2.0:LRA=11"


# AI Act art. 50(2): a provider whose system generates synthetic audio or video
# has to mark the output in a machine-readable way. The regulation asks for the
# marking, not for a particular standard, and the cheap one that survives every
# player and every upload is a container tag: `ffprobe -show_format` reads it
# back, and TikTok/Reels/YouTube ignore it. Stamped by a stream copy, so it is
# a remux (no quality loss, ~0.2 s) and never a re-encode.
#
# Deliberately narrow: it goes on outputs where a machine actually synthesised
# voice or a person (dubbing, AI actors), not on an ordinary clip, whose audio
# and pixels are the user's own footage. Marking everything would make the tag
# mean nothing.
AI_DISCLOSURE = "AI-generated content produced with OpenShorts (openshorts.app)"


def mark_ai_generated(path, detail=""):
    """Stamp AI Act art. 50(2) machine-readable tags on a finished file.

    Returns True when the file now carries the tags. Never raises: a missing
    tag must not lose the user the video they just paid minutes for.
    """
    note = f"{AI_DISCLOSURE}: {detail}" if detail else AI_DISCLOSURE
    tmp = f"{path}.aitag.mp4"
    # `comment` and not a custom `ai_generated` key: mp4 only carries the
    # standard iTunes-style tags, and ffmpeg drops anything else without
    # warning (verified with ffprobe -show_entries format_tags).
    # -map 0 because ffmpeg's default picks one stream per type: a dubbed file
    # that ever ships two audio tracks would come back with one.
    cmd = [ffmpeg_binary(), "-y", "-i", path, "-map", "0", "-c", "copy",
           "-metadata", f"comment={note}",
           "-movflags", "+faststart", tmp]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
            print(f"[ai-tag] skipped for {os.path.basename(path)}: {r.stderr[-300:]}")
            if os.path.exists(tmp):
                os.remove(tmp)
            return False
        os.replace(tmp, path)
        return True
    except Exception as e:
        print(f"[ai-tag] skipped for {os.path.basename(path)}: {e}")
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        return False


def audio_encode_args():
    """AAC encode args for a delivered clip, with loudness normalisation."""
    args = ["-c:a", "aac"]
    if os.environ.get("AUDIO_NORMALIZE", "1").strip() != "0":
        args = ["-af", LOUDNORM_FILTER] + args
    return args

_probe_lock = threading.Lock()
_nvenc_ok = None  # None = not probed yet
_announced = False


# --- Binary resolution ------------------------------------------------------
#
# Every ffmpeg invocation in this codebase historically used the bare string
# "ffmpeg", which only works when the binary is on PATH. A machine without it
# used to fail mid-job with a cryptic "FileNotFoundError: [WinError 2]" at the
# clip-cut step, three minutes into a paid render. These helpers fix that:
#
#   FFMPEG_PATH env var > shutil.which("ffmpeg") > imageio-ffmpeg package
#
# FFMPEG_PATH may point at the ffmpeg executable itself OR at its bin folder.
# resolve_ffmpeg() also prepends the binary's directory to THIS process's PATH
# so every existing bare "ffmpeg"/"ffprobe" subprocess call site (main.py,
# active_speaker.py, editor.py, ...) resolves even when the binary lives
# outside the inherited PATH.

_EXE = ".exe" if os.name == "nt" else ""

_ffmpeg_bin = None
_ffprobe_bin = None
_bin_lock = threading.Lock()


class FFmpegNotFoundError(RuntimeError):
    """No usable ffmpeg binary could be located."""


def _install_hint():
    return (
        "FFmpeg was not found. Fix it with ONE of:\n"
        "  1. Install it:  winget install Gyan.FFmpeg   (or choco install ffmpeg,\n"
        "     or download https://www.gyan.dev/ffmpeg/builds/ and add bin/ to PATH)\n"
        "  2. Set the FFMPEG_PATH env var to the folder containing ffmpeg(.exe)\n"
        "     — or to the ffmpeg(.exe) file itself\n"
        "  3. pip install imageio-ffmpeg  (ffmpeg only; ffprobe still needs a\n"
        "     full install)"
    )


def _candidates_from_ffmpeg_path_env():
    """(ffmpeg, ffprobe) candidates from FFMPEG_PATH; (None, None) if unset."""
    raw = os.environ.get("FFMPEG_PATH", "").strip().strip('"')
    if not raw:
        return None, None
    if os.path.isfile(raw):
        base = os.path.dirname(os.path.abspath(raw))
        return os.path.abspath(raw), os.path.join(base, "ffprobe" + _EXE)
    if os.path.isdir(raw):
        return (os.path.join(raw, "ffmpeg" + _EXE),
                os.path.join(raw, "ffprobe" + _EXE))
    return None, None


def _candidate_from_imageio():
    try:
        import imageio_ffmpeg  # optional dependency
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def resolve_ffmpeg():
    """Locate ffmpeg/ffprobe once per process (thread-safe, cached).

    Returns (ffmpeg_path, ffprobe_path_or_None). Raises FFmpegNotFoundError
    when nothing usable is found. Side effect: the resolved binary's folder
    is prepended to this process's PATH so the bare "ffmpeg"/"ffprobe" strings
    at all other call sites resolve too.
    """
    global _ffmpeg_bin, _ffprobe_bin
    if _ffmpeg_bin is not None:
        return _ffmpeg_bin, _ffprobe_bin
    with _bin_lock:
        if _ffmpeg_bin is not None:
            return _ffmpeg_bin, _ffprobe_bin

        ffmpeg = ffprobe = None
        env_ff, env_fp = _candidates_from_ffmpeg_path_env()
        if env_ff and os.path.isfile(env_ff):
            ffmpeg = env_ff
            if env_fp and os.path.isfile(env_fp):
                ffprobe = env_fp
        if ffmpeg is None:
            ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            ffmpeg = _candidate_from_imageio()
        if not ffmpeg:
            raise FFmpegNotFoundError(_install_hint())

        # Prepend so this beats any stale entry; keep idempotent.
        bin_dir = os.path.dirname(os.path.abspath(ffmpeg))
        current = os.environ.get("PATH", "")
        if bin_dir not in current.split(os.pathsep):
            os.environ["PATH"] = bin_dir + os.pathsep + current

        if ffprobe is None:
            ffprobe = shutil.which("ffprobe")

        _ffmpeg_bin, _ffprobe_bin = ffmpeg, ffprobe
    return _ffmpeg_bin, _ffprobe_bin


def ffmpeg_binary():
    """Path to the ffmpeg executable (resolves once, then cached)."""
    return resolve_ffmpeg()[0]


def ffprobe_binary():
    """Path to ffprobe, or None when only ffmpeg was found (imageio-ffmpeg)."""
    return resolve_ffmpeg()[1]


def require_ffmpeg():
    """Startup validation: resolve the binaries and run `ffmpeg -version` once.

    Raises FFmpegNotFoundError with human-fix instructions. Cheap (one short
    subprocess); call it at process startup so a missing binary is discovered
    before a job invests minutes of transcription/inference work.
    """
    ffmpeg, ffprobe = resolve_ffmpeg()
    try:
        r = subprocess.run([ffmpeg, "-version"], capture_output=True,
                           timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        raise FFmpegNotFoundError(
            f"ffmpeg at {ffmpeg!r} could not be executed: {e}\n"
            + _install_hint()) from e
    if r.returncode != 0:
        raise FFmpegNotFoundError(
            f"ffmpeg at {ffmpeg!r} exists but exits with code {r.returncode} "
            f"on -version.\n" + _install_hint())
    if ffprobe is None:
        print("⚠️ [ffmpeg] ffprobe was not found (only ffmpeg). Features that "
              "probe durations will fail until ffprobe is on PATH or "
              "FFMPEG_PATH points at its folder.")
    return ffmpeg, ffprobe


def _probe_nvenc():
    """One tiny lavfi encode to prove h264_nvenc works end-to-end.

    NVENC rejects frames smaller than ~145px, so the probe uses 256x256.
    Any failure (no ffmpeg binary, no GPU, no driver libs) means False.
    """
    cmd = [
        ffmpeg_binary(), "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
        "-c:v", "h264_nvenc", "-f", "null", "-",
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30
        )
        return result.returncode == 0
    except Exception:
        return False


def nvenc_available():
    """Probe h264_nvenc once and cache the verdict (thread-safe)."""
    global _nvenc_ok
    if _nvenc_ok is None:
        with _probe_lock:
            if _nvenc_ok is None:
                _nvenc_ok = _probe_nvenc()
    return _nvenc_ok


def reset_encoder_cache():
    """Test hook: forget the cached probe result."""
    global _nvenc_ok, _announced
    with _probe_lock:
        _nvenc_ok = None
        _announced = False


def video_encode_args(tier=QUALITY):
    """Return the codec/quality args for one encode, honoring FFMPEG_ENCODER."""
    global _announced
    if tier not in _X264_ARGS:
        raise ValueError(f"Unknown encode tier: {tier!r}")

    mode = os.environ.get("FFMPEG_ENCODER", "x264").strip().lower()
    use_nvenc = False
    if mode in ("nvenc", "auto"):
        use_nvenc = nvenc_available()
        if mode == "nvenc" and not use_nvenc:
            print("⚠️ [Encoder] FFMPEG_ENCODER=nvenc but h264_nvenc is not "
                  "usable here — falling back to libx264")

    if not _announced:
        _announced = True
        print(f"🎞️ [Encoder] video encoder: {'h264_nvenc' if use_nvenc else 'libx264'} "
              f"(FFMPEG_ENCODER={mode})")

    return list((_NVENC_ARGS if use_nvenc else _X264_ARGS)[tier])


def escape_filter_value(value):
    r"""Escape a path/value for use inside a quoted FFmpeg filter argument.

    Windows absolute paths are why this exists: ``:`` separates filter options,
    so an interpolated ``C:/x/y.txt`` makes the parser look for an option named
    ``/x/y.txt`` and the whole filtergraph fails to build.

    NOTE: an apostrophe in the path cannot be made safe here. ffmpeg's
    filtergraph parser is not a shell -- the shell idiom ``'\''`` was tried on
    29-jul-2026 and is worse than doing nothing: it drops the apostrophe AND
    swallows the following option, so ``ass='...Earth'\''s.ass':fontsdir='...'``
    resolved to a filename of "...Earths.ass:fontsdir=..." and failed to open.

    The only reliable answer is to keep apostrophes OUT of any path that is
    interpolated into a filter. Callers generate their own filenames, so they
    control this: use a neutral name, never one derived from a video title.
    """
    return value.replace('\\', '/').replace(':', '\\:').replace("'", "\\'")
