"""HTTP control API.

Optional, off by default, and deliberately not a user interface -- it exists so
a headless rig can be controlled and scripted.

Two things it makes possible:

**Runtime control without a monitor.** A machine in a rack can be told to lock
onto a specific person, released back to automatic, or sent home.

**Live tuning without restarting.** Every gain in this system is a feel
judgement that needs watching a real subject move, and a restart costs several
seconds of model loading. Because each controller reads its config dataclass on
every step, a POST to /api/config takes effect on the very next frame::

    curl -X POST localhost:8477/api/config -d '{"pan.kp": 1.8}'
    curl localhost:8477/api/config.yaml > tuned.yaml

Stdlib only: ``http.server`` plus OpenCV's JPEG encoder. No web framework, no
HTML, no new dependencies.

Security: binds to localhost by default, because this API can move a physical
motor and read a camera. Setting ``host: 0.0.0.0`` exposes it to your whole
network; a token is generated automatically when you do.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2


@dataclass
class ServerConfig:
    enabled: bool = False
    #: Localhost by default. "0.0.0.0" exposes control of a physical camera to
    #: the network -- pair it with a token.
    host: str = "127.0.0.1"
    port: int = 8477
    #: Shared secret required as ?token=... or an X-Auth-Token header.
    #: Auto-generated and printed at startup when the host is not localhost.
    token: str | None = None
    #: Max frames per second for the JPEG/MJPEG endpoints. Encoding costs CPU,
    #: so this sits well below the control rate.
    stream_fps: float = 12.0
    stream_quality: int = 70
    #: Long edge of served frames. These are for diagnosis, not for program out.
    stream_width: int = 960


class SharedState:
    """Hand-off point between the control loop and the HTTP threads."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.director_jpeg: bytes | None = None
        self.program_jpeg: bytes | None = None
        self.frame_seq = 0
        self.status: dict = {}
        self.viewers = 0
        self.commands: list[tuple[str, dict]] = []
        self._interest_until = 0.0

    # -- producer side (control loop) --------------------------------------
    def publish(
        self,
        *,
        director=None,
        program=None,
        status: dict | None = None,
        quality: int = 70,
        width: int = 960,
    ) -> None:
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        d_jpeg = _encode(director, enc, width) if director is not None else None
        p_jpeg = _encode(program, enc, width) if program is not None else None
        with self.lock:
            if d_jpeg is not None:
                self.director_jpeg = d_jpeg
            if p_jpeg is not None:
                self.program_jpeg = p_jpeg
            if status is not None:
                self.status = status
            self.frame_seq += 1

    def note_director_interest(self, seconds: float = 2.0) -> None:
        """Someone wants the annotated view; keep rendering it for a moment.

        The overlay is only drawn when it will be looked at -- it copies the
        full frame, which is wasted work on a headless rig nobody is watching.
        A one-shot JPEG request has to switch it back on.
        """
        with self.lock:
            self._interest_until = max(self._interest_until, time.perf_counter() + seconds)

    @property
    def director_wanted(self) -> bool:
        with self.lock:
            return self.viewers > 0 or time.perf_counter() < self._interest_until

    def take_commands(self) -> list[tuple[str, dict]]:
        with self.lock:
            out, self.commands = self.commands, []
        return out

    # -- consumer side (HTTP threads) --------------------------------------
    def push_command(self, name: str, payload: dict) -> None:
        with self.lock:
            self.commands.append((name, payload))

    def snapshot(self):
        with self.lock:
            return self.director_jpeg, self.program_jpeg, self.frame_seq, dict(self.status)


def _encode(frame, params, width: int) -> bytes | None:
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if w > width:
        scale = width / float(w)
        frame = cv2.resize(frame, (width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, params)
    return buf.tobytes() if ok else None


class ControlServer:
    def __init__(self, cfg: ServerConfig, state: SharedState, config_provider) -> None:
        self.cfg = cfg
        self.state = state
        #: Callable returning the live AppConfig, so the UI always reflects
        #: what the running pipeline is actually using.
        self.config_provider = config_provider
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

        if cfg.token is None and cfg.host not in ("127.0.0.1", "localhost", "::1"):
            cfg.token = secrets.token_urlsafe(12)

    @property
    def url(self) -> str:
        host = "localhost" if self.cfg.host in ("0.0.0.0", "127.0.0.1") else self.cfg.host
        base = f"http://{host}:{self.cfg.port}/"
        return f"{base}?token={self.cfg.token}" if self.cfg.token else base

    def start(self) -> ControlServer:
        handler = _make_handler(self)
        self._httpd = ThreadingHTTPServer((self.cfg.host, self.cfg.port), handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


def _make_handler(server: ControlServer):
    cfg = server.cfg
    state = server.state

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "camerafollow"

        def log_message(self, *args) -> None:  # quiet; the console is for the pipeline
            pass

        # -- helpers -------------------------------------------------------
        def _authorised(self, query) -> bool:
            if not cfg.token:
                return True
            supplied = self.headers.get("X-Auth-Token") or (query.get("token", [None])[0])
            return bool(supplied) and secrets.compare_digest(supplied, cfg.token)

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _json(self, payload, code: int = 200) -> None:
            self._send(code, json.dumps(payload).encode(), "application/json")

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return {}

        # -- routes --------------------------------------------------------
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if not self._authorised(query):
                return self._json({"error": "unauthorised"}, 401)

            route = parsed.path
            if route == "/":
                # A plain index of the API. There is no web page: this server
                # exists so headless rigs can be controlled and scripted, not
                # to be a product surface.
                return self._json(
                    {
                        "name": "camerafollow",
                        "endpoints": {
                            "GET /api/status": "live pipeline state as JSON",
                            "GET /api/config": "full running config as JSON",
                            "GET /api/config.yaml": "running config as YAML, paste-ready",
                            "POST /api/config": "live override, e.g. {'pan.kp': 1.8}",
                            "POST /api/lock": "follow a track: {'id': 3}, or {} for most central",
                            "POST /api/unlock": "return to automatic subject selection",
                            "POST /api/home": "send the head to its home position",
                            "GET /director.jpg": "annotated diagnostic frame",
                            "GET /program.jpg": "clean output frame",
                            "GET /director.mjpg": "annotated frame, MJPEG stream",
                            "GET /program.mjpg": "clean output, MJPEG stream",
                        },
                    }
                )
            if route == "/api/status":
                return self._json(state.snapshot()[3])
            if route == "/api/config":
                return self._json(server.config_provider().to_dict())
            if route == "/api/config.yaml":
                import yaml

                text = yaml.safe_dump(server.config_provider().to_dict(), sort_keys=False)
                return self._send(200, text.encode(), "text/plain; charset=utf-8")
            if route in ("/director.mjpg", "/program.mjpg"):
                if route.startswith("/director"):
                    state.note_director_interest(5.0)
                return self._stream(program=route.startswith("/program"))
            if route in ("/director.jpg", "/program.jpg"):
                want_program = route.startswith("/program")
                if not want_program:
                    state.note_director_interest()
                img = self._await_frame(program=want_program)
                if img is None:
                    return self._json({"error": "no frame yet"}, 503)
                return self._send(200, img, "image/jpeg")
            return self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if not self._authorised(query):
                return self._json({"error": "unauthorised"}, 401)

            route = parsed.path
            body = self._body()

            if route == "/api/lock":
                state.push_command("lock", body)
                return self._json({"ok": True})
            if route == "/api/unlock":
                state.push_command("unlock", {})
                return self._json({"ok": True})
            if route == "/api/home":
                state.push_command("home", {})
                return self._json({"ok": True})
            if route == "/api/config":
                try:
                    server.config_provider().apply_overrides(body)
                except (KeyError, TypeError, ValueError) as exc:
                    return self._json({"error": str(exc)}, 400)
                return self._json({"ok": True})
            return self._json({"error": "not found"}, 404)

        def _await_frame(self, *, program: bool, timeout: float = 1.5):
            deadline = time.perf_counter() + timeout
            while True:
                d, p, _, _ = state.snapshot()
                img = p if program else d
                if img is not None or time.perf_counter() > deadline:
                    return img
                time.sleep(0.03)

        # -- mjpeg ---------------------------------------------------------
        def _stream(self, *, program: bool) -> None:
            boundary = "camerafollowframe"
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

            with state.lock:
                state.viewers += 1
            period = 1.0 / max(1.0, cfg.stream_fps)
            last_seq = -1
            try:
                while True:
                    d, p, seq, _ = state.snapshot()
                    img = p if program else d
                    if img is None or seq == last_seq:
                        time.sleep(period / 2)
                        continue
                    last_seq = seq
                    head = (
                        f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                        f"Content-Length: {len(img)}\r\n\r\n"
                    ).encode()
                    self.wfile.write(head)
                    self.wfile.write(img)
                    self.wfile.write(b"\r\n")
                    time.sleep(period)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with state.lock:
                    state.viewers = max(0, state.viewers - 1)

    return Handler
