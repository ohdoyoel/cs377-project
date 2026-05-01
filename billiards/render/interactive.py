"""Interactive 4-ball carom simulator (browser GUI).

Aim by clicking the table, set power with a slider, choose where to hit the
cue ball with a contact-point widget, then press Shoot to watch the result
play back. Useful for sanity-checking that the physics behaves intuitively
(cushion bounces, follow/draw, side english, friction decay).

Run from the project directory:

    uv run python -m billiards.render.interactive            # default port 8765
    uv run python -m billiards.render.interactive 9000       # custom port

Then open the URL printed in the terminal.
"""

from __future__ import annotations

import json
import math
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from ..physics import CueAction, TableSpec, TableState, simulate_shot
from .replay import _sprite_data_url

_HTML_PATH = Path(__file__).with_name("interactive.html")
_SPEC = TableSpec()
_STATE: TableState = TableState.initial_4ball(spec=_SPEC)


def _state_payload(state: TableState) -> dict[str, Any]:
    return {
        "balls": [[b.x, b.y, b.vx, b.vy, b.wx, b.wy, b.wz] for b in state.balls],
        "cue_id": state.cue_id,
        "spec": {
            "width_m": state.spec.width,
            "height_m": state.spec.height,
            "ball_radius_m": state.spec.ball_radius,
        },
    }


def _build_index_html() -> bytes:
    html = _HTML_PATH.read_text(encoding="utf-8")
    html = html.replace("{{SPRITE_WHITE}}", _sprite_data_url("white"))
    html = html.replace("{{SPRITE_YELLOW}}", _sprite_data_url("yellow"))
    html = html.replace("{{SPRITE_RED}}", _sprite_data_url("red"))
    html = html.replace("{{INITIAL_STATE_JSON}}",
                        json.dumps(_state_payload(_STATE)))
    return html.encode("utf-8")


def _serialize_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ev in events:
        detail: dict[str, Any] = {}
        for k, v in ev.get("detail", {}).items():
            if isinstance(v, np.ndarray):
                detail[k] = v.tolist()
            elif isinstance(v, tuple):
                detail[k] = list(v)
            else:
                detail[k] = v
        out.append({"t": float(ev["t"]), "type": ev["type"], "detail": detail})
    return out


def _serialize_trajectory(traj) -> list[dict[str, Any]]:
    return [{"t": float(t), "balls": np.asarray(arr).tolist()} for t, arr in traj]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args, **_kw):  # silence default request logging
        pass

    def _send_json(self, code: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path in ("/", "/index.html"):
            self._send_html(_build_index_html()); return
        if self.path == "/state":
            self._send_json(200, _state_payload(_STATE)); return
        self.send_response(404); self.end_headers()

    def do_POST(self):  # noqa: N802
        global _STATE
        n = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(n) if n else b""
        try:
            data = json.loads(body or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "bad json"}); return

        if self.path == "/shoot":
            try:
                ax = float(data.get("a", 0.0))
                ay = float(data.get("b", 0.0))
                r = math.hypot(ax, ay)
                if r > 1.0:
                    ax /= r; ay /= r
                action = CueAction(
                    theta=float(data["theta"]) % (2.0 * math.pi),
                    power=max(0.0, min(1.0, float(data["power"]))),
                    a=ax, b=ay,
                )
                result = simulate_shot(_STATE, action)
                self._send_json(200, {
                    "frames": _serialize_trajectory(result["trajectory"]),
                    "events": _serialize_events(result["events"]),
                    "score": int(result["score"]),
                    "fouled": bool(result["fouled"]),
                    "cushion_hits": int(result["cushion_hits"]),
                    "duration": float(result["duration"]),
                    "final_state": _state_payload(_STATE),
                })
            except Exception as e:
                self._send_json(400, {"error": f"{type(e).__name__}: {e}"})
            return

        if self.path == "/reset":
            _STATE = TableState.initial_4ball(spec=_SPEC)
            self._send_json(200, _state_payload(_STATE)); return

        self.send_response(404); self.end_headers()


def main(host: str = "127.0.0.1", port: int = 8765) -> None:
    print(f"4-ball interactive simulator → http://{host}:{port}")
    print("(Ctrl-C to stop)")
    ThreadingHTTPServer((host, port), _Handler).serve_forever()


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    main(port=p)
