"""Deep client test for arxiv_lab.backends.systemone (localhost mock server).

Exercises SystemoneClient.decide() end-to-end over real HTTP against an
in-process mock of Ollama's /v1/systemone: success, HTTP errors, malformed
bodies, timeouts, and connection failures. No external network.
"""
import http.server
import json
import os
import socket
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arxiv_lab.backends.systemone import (  # noqa: E402
    SystemoneClient, DEFAULT_BASE_URL, DEFAULT_MODEL,
)

PASS = []
FAIL = []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
    except Exception as e:  # noqa: BLE001 - harness reports, not raises
        FAIL.append((name, f"{type(e).__name__}: {e}"))


def expect_raises(name, exc, fn, match=None):
    def _f():
        try:
            fn()
        except exc as e:
            if match and match not in str(e):
                raise AssertionError(
                    f"error {e!r} does not contain {match!r}")
            return
        except Exception as e:  # noqa: BLE001
            raise AssertionError(
                f"expected {exc.__name__}, got {type(e).__name__}: {e}")
        raise AssertionError(f"expected {exc.__name__}, nothing raised")
    check(name, _f)


OK_BODY = {
    "model": "clef-flash:latest",
    "answers": {
        "page_oncall": {"type": "noul", "noul": 0.92},
        "severity": {
            "type": "choice", "choice": "high",
            "probabilities": {"high": 0.6, "low": 0.1, "medium": 0.3},
            "confidence": 0.2,
        },
        "urgency": {
            "type": "score", "score": 1.4,
            "legend": {"0": 0, "1": 0.5, "2": 1},
            "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
        },
    },
}

QUESTIONS = {
    "page_oncall": {"type": "noul", "instructions": "Page?"},
    "severity": {"type": "choice", "instructions": "Severity?",
                 "criteria": {"low": "minor", "high": "critical"}},
    "urgency": {"type": "score", "instructions": "Urgency?",
                "criteria": [0, 0.5, 1]},
}


class MockHandler(http.server.BaseHTTPRequestHandler):
    mode = "ok"
    seen_bodies = []

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        MockHandler.seen_bodies.append(body)
        return body

    def do_POST(self):  # noqa: N802 - http.server convention
        self._read_body()
        mode = MockHandler.mode
        if mode == "ok":
            self._send(200, json.dumps(OK_BODY))
        elif mode == "http400":
            self._send(400, json.dumps(
                {"error": 'question "x": score criteria must be an array'}))
        elif mode == "badjson":
            self._send(200, "this is not json{{{")
        elif mode == "badshape":
            self._send(200, json.dumps({"nope": True}))
        elif mode == "slow":
            import time
            time.sleep(5)
            self._send(200, json.dumps(OK_BODY))
        else:
            self._send(500, "unknown mock mode")

    def _send(self, code, body):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):  # silence
        pass


def start_server():
    srv = http.server.HTTPServer(("127.0.0.1", 0), MockHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


SRV = start_server()
BASE = f"http://127.0.0.1:{SRV.server_port}"


def client(**kw):
    kw.setdefault("base_url", BASE)
    kw.setdefault("timeout", 10)
    return SystemoneClient(**kw)


def t_success():
    MockHandler.mode = "ok"
    MockHandler.seen_bodies = []
    ans = client().decide("db slow", QUESTIONS)
    assert ans["page_oncall"] == {"type": "noul", "p": 0.92}
    assert ans["severity"]["choice"] == "high"
    assert ans["urgency"]["score"] == 1.4
    # request body went through build_request validation + shaping
    sent = json.loads(MockHandler.seen_bodies[-1].decode())
    assert sent["model"] == DEFAULT_MODEL
    assert sent["state"] == "db slow"
    assert set(sent["questions"]) == {"page_oncall", "severity", "urgency"}


def t_trailing_slash():
    MockHandler.mode = "ok"
    c = SystemoneClient(base_url=BASE + "///", timeout=10)
    assert c.base_url == BASE
    ans = c.decide("db slow", QUESTIONS)
    assert ans["page_oncall"]["p"] == 0.92


def t_defaults():
    c = SystemoneClient()
    assert c.base_url == DEFAULT_BASE_URL
    assert c.model == DEFAULT_MODEL
    assert DEFAULT_MODEL == "clef-flash:latest"


for good in (t_success, t_trailing_slash, t_defaults):
    check(good.__name__, good)


def _http400():
    MockHandler.mode = "http400"
    client().decide("db slow", QUESTIONS)


def _badjson():
    MockHandler.mode = "badjson"
    client().decide("db slow", QUESTIONS)


def _badshape():
    MockHandler.mode = "badshape"
    client().decide("db slow", QUESTIONS)


def _conn_refused():
    # Port that nothing listens on.
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    SystemoneClient(base_url=f"http://127.0.0.1:{port}",
                    timeout=5).decide("db slow", QUESTIONS)


def _timeout():
    MockHandler.mode = "slow"
    SystemoneClient(base_url=BASE, timeout=1).decide("db slow", QUESTIONS)


expect_raises("http400_is_runtimeerror", RuntimeError, _http400, "HTTP 400")
expect_raises("http400_carries_server_message", RuntimeError, _http400,
              "score criteria must be an array")
expect_raises("malformed_json_is_runtimeerror", RuntimeError, _badjson)
expect_raises("bad_shape_is_runtimeerror", RuntimeError, _badshape)
expect_raises("conn_refused_is_runtimeerror", RuntimeError, _conn_refused)
expect_raises("timeout_is_runtimeerror", RuntimeError, _timeout)

SRV.shutdown()
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
for name, err in FAIL:
    print("FAIL", name, "-", err)
sys.exit(1 if FAIL else 0)
