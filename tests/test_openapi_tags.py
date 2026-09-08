"""Swagger organization: every route carries an OpenAPI tag from the registry.

app.OPENAPI_TAGS is what groups /docs into readable categories instead of one
flat "default" list. This module keeps the registry and the route decorators
in sync: a new endpoint without tags=[...] would silently fall back into
FastAPI's catch-all "default" group in Swagger.
"""
import pytest

app_module = pytest.importorskip("app")

# Routes FastAPI adds itself; they are documentation plumbing, not API surface.
_DOC_ROUTES = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}


def _api_routes():
    for route in app_module.app.routes:
        if not getattr(route, "methods", None):
            continue  # static mounts have no HTTP methods
        if route.path in _DOC_ROUTES:
            continue
        yield route


def test_every_route_has_a_registered_tag():
    declared = {t["name"] for t in app_module.OPENAPI_TAGS}
    seen = set()
    for route in _api_routes():
        assert route.tags, f"{sorted(route.methods)} {route.path} has no tags"
        for tag in route.tags:
            assert tag in declared, f"{route.path} uses unregistered tag {tag!r}"
            seen.add(tag)
    assert seen == declared, (
        f"tag registry out of sync: declared-but-unused={sorted(declared - seen)}, "
        f"used-but-undeclared={sorted(seen - declared)}"
    )


def test_legacy_pages_are_marked_deprecated():
    hits = [r for r in _api_routes() if r.path in ("/gallery", "/video/{video_id}")]
    assert len(hits) == 2
    for route in hits:
        assert route.deprecated, f"{route.path} must be marked deprecated"
        assert route.tags == ["Legacy Pages"]


def test_mcp_routes_are_tagged():
    mcp = [r for r in _api_routes() if r.path == "/mcp"]
    assert len(mcp) == 2  # GET + POST, from mcp_server.router
    for route in mcp:
        assert "MCP" in route.tags
