from fastapi.testclient import TestClient

from app.main import create_app


def test_health_ok(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "loan-pledge-ocr-api"
    assert body["ocrInitialized"] is False


def test_request_id_header_present_and_unique(client):
    a = client.get("/api/v1/health").headers["X-Request-ID"]
    b = client.get("/api/v1/health").headers["X-Request-ID"]
    assert a and b and a != b


def test_unknown_route_uses_error_format(client):
    r = client.get("/api/v1/nope")
    assert r.status_code == 404
    body = r.json()
    assert body["success"] is False
    assert body["requestId"]
    assert body["error"]["code"] == "http_404"


def test_unhandled_error_is_generic_and_has_request_id():
    app = create_app()

    @app.get("/boom")
    async def boom():
        raise RuntimeError("secret internal detail C:\\private\\path")

    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/boom")
    assert r.status_code == 500
    body = r.json()
    assert body["success"] is False
    assert body["requestId"]
    assert body["error"]["code"] == "internal_error"
    assert "secret" not in r.text
    assert "private" not in r.text