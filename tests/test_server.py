"""HTTP control API.

Exercised over a real socket rather than by calling handlers directly -- the
routing, auth, and JSON shapes are the contract other people will build on.
"""

import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from camerafollow.config import AppConfig
from camerafollow.server import ControlServer, ServerConfig, SharedState


@pytest.fixture
def rig():
    cfg = AppConfig()
    state = SharedState()
    server = ControlServer(ServerConfig(host="127.0.0.1", port=0), state, lambda: cfg)
    # Port 0 lets the OS pick a free one, so tests never collide.
    server.start()
    server.cfg.port = server._httpd.server_address[1]
    try:
        yield server, state, cfg
    finally:
        server.stop()


def _url(server, path):
    return f"http://127.0.0.1:{server.cfg.port}{path}"


def _get(server, path, token=None):
    url = _url(server, path) + (f"?token={token}" if token else "")
    with urllib.request.urlopen(url, timeout=3) as r:
        return r.status, r.read()


def _post(server, path, payload=None, token=None):
    url = _url(server, path) + (f"?token={token}" if token else "")
    req = urllib.request.Request(
        url,
        data=json.dumps(payload or {}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=3) as r:
        return r.status, json.loads(r.read())


# -- basics -----------------------------------------------------------------
def test_index_documents_the_api(rig):
    """No web page: `/` lists the endpoints so the API is self-describing."""
    server, _, _ = rig
    status, body = _get(server, "/")
    assert status == 200
    payload = json.loads(body)
    assert payload["name"] == "camerafollow"
    for route in ("GET /api/status", "POST /api/lock", "GET /program.mjpg"):
        assert route in payload["endpoints"]


def test_no_html_is_served(rig):
    """This is an API, not a user interface. Keep it that way."""
    server, _, _ = rig
    _, body = _get(server, "/")
    assert b"<html" not in body.lower() and b"<script" not in body.lower()


def test_status_reflects_what_the_pipeline_published(rig):
    server, state, _ = rig
    state.publish(status={"control_fps": 59.9, "subject": "#3"})
    _, body = _get(server, "/api/status")
    assert json.loads(body)["subject"] == "#3"


def test_unknown_route_is_404(rig):
    server, _, _ = rig
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(server, "/nope")
    assert exc.value.code == 404


# -- config -----------------------------------------------------------------
def test_config_round_trips_and_applies_live(rig):
    server, _, cfg = rig
    _, body = _get(server, "/api/config")
    assert json.loads(body)["pan"]["kp"] == cfg.pan.kp

    _post(server, "/api/config", {"pan.kp": 2.25, "framing.target_y": 0.29})
    # The pipeline holds this same object, so the change is live immediately.
    assert cfg.pan.kp == 2.25
    assert cfg.framing.target_y == 0.29


def test_bad_config_key_is_rejected_not_ignored(rig):
    server, _, cfg = rig
    before = cfg.pan.kp
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(server, "/api/config", {"pan.not_a_key": 1})
    assert exc.value.code == 400
    assert cfg.pan.kp == before


def test_config_yaml_is_copy_pasteable(rig):
    """The tuning workflow: POST overrides, then save the result to a file."""
    server, _, _ = rig
    _post(server, "/api/config", {"pan.kp": 1.85})
    _, body = _get(server, "/api/config.yaml")
    text = body.decode()
    assert "pan:" in text and "1.85" in text
    reloaded = AppConfig.from_dict(__import__("yaml").safe_load(text))
    assert reloaded.pan.kp == 1.85


# -- commands ---------------------------------------------------------------
def test_commands_are_queued_for_the_control_thread(rig):
    """Actions must not touch pipeline state from an HTTP thread."""
    server, state, _ = rig
    _post(server, "/api/lock", {"id": 7})
    _post(server, "/api/unlock")
    _post(server, "/api/home")
    assert state.take_commands() == [("lock", {"id": 7}), ("unlock", {}), ("home", {})]
    assert state.take_commands() == []  # draining is destructive


# -- auth -------------------------------------------------------------------
def test_token_is_required_when_set(rig):
    server, state, _ = rig
    server.cfg.token = "s3cret"
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(server, "/api/status")
    assert exc.value.code == 401
    assert _get(server, "/api/status", token="s3cret")[0] == 200


def test_wrong_token_cannot_move_the_camera(rig):
    server, state, _ = rig
    server.cfg.token = "s3cret"
    with pytest.raises(urllib.error.HTTPError):
        _post(server, "/api/home", token="wrong")
    assert state.take_commands() == []


def test_non_local_bind_generates_a_token_automatically():
    """Exposing camera control to a LAN without a secret would be a footgun."""
    cfg = ServerConfig(host="0.0.0.0", port=0)
    ControlServer(cfg, SharedState(), lambda: AppConfig())
    assert cfg.token


def test_localhost_bind_needs_no_token():
    cfg = ServerConfig(host="127.0.0.1", port=0)
    ControlServer(cfg, SharedState(), lambda: AppConfig())
    assert cfg.token is None


# -- imagery ----------------------------------------------------------------
def test_jpeg_endpoints_serve_published_frames(rig):
    server, state, _ = rig
    frame = (np.random.rand(180, 320, 3) * 255).astype(np.uint8)
    state.publish(director=frame, program=frame)
    for path in ("/director.jpg", "/program.jpg"):
        status, body = _get(server, path)
        assert status == 200 and body[:3] == b"\xff\xd8\xff", path


def test_director_request_signals_the_pipeline_to_render_it(rig):
    """The overlay is skipped when unobserved; asking must switch it back on."""
    server, state, _ = rig
    assert not state.director_wanted

    def ask():
        try:
            _get(server, "/director.jpg")  # 503 is fine; the signal is the point
        except urllib.error.HTTPError:
            pass

    threading.Thread(target=ask, daemon=True).start()
    deadline = time.perf_counter() + 2
    while time.perf_counter() < deadline and not state.director_wanted:
        time.sleep(0.02)
    assert state.director_wanted


def test_published_frames_are_downscaled_for_browsers(rig):
    _, state, _ = rig
    big = (np.random.rand(2160, 3840, 3) * 255).astype(np.uint8)
    state.publish(program=big, width=640)
    import cv2

    decoded = cv2.imdecode(np.frombuffer(state.program_jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] == 640
