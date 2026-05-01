"""Render a billiards trajectory as a self-contained HTML replay.

The output is a single HTML document with sprites inlined as data URLs,
trajectory inlined as JSON, and a small JS player for play/pause/restart
and variable speed. Designed to be returned as ``IPython.display.HTML``
inside a Jupyter notebook, but also writable to disk via ``save_path``.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

_VIEWER_HTML_PATH = Path(__file__).with_name("viewer.html")
_SPRITES_DIR = Path(__file__).with_name("sprites")
_SPRITE_NAMES = ("white", "yellow", "red")
_DEFAULT_SPEC = {"width_m": 2.540, "height_m": 1.270, "ball_radius_m": 0.03275}

# Cache: read sprites + template once per process.
_SPRITE_CACHE: dict[str, str] = {}
_TEMPLATE_CACHE: str | None = None


def _sprite_data_url(name: str) -> str:
    if name not in _SPRITE_CACHE:
        path = _SPRITES_DIR / f"ball_{name}.png"
        if not path.exists():
            raise FileNotFoundError(
                f"sprite {path} missing — run "
                f"`uv run python billiards/render/generate_sprites.py` first"
            )
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        _SPRITE_CACHE[name] = f"data:image/png;base64,{b64}"
    return _SPRITE_CACHE[name]


def _template() -> str:
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is None:
        _TEMPLATE_CACHE = _VIEWER_HTML_PATH.read_text(encoding="utf-8")
    return _TEMPLATE_CACHE


def _coerce_spec(spec) -> dict:
    """Accept TableSpec, dict, or None. Return JSON-serializable dict."""
    if spec is None:
        return dict(_DEFAULT_SPEC)
    if isinstance(spec, dict):
        out = dict(_DEFAULT_SPEC)
        # Allow keys like 'width' / 'height' / 'ball_radius' as well.
        if "width" in spec and "width_m" not in spec:
            out["width_m"] = float(spec["width"])
        if "height" in spec and "height_m" not in spec:
            out["height_m"] = float(spec["height"])
        if "ball_radius" in spec and "ball_radius_m" not in spec:
            out["ball_radius_m"] = float(spec["ball_radius"])
        for k in ("width_m", "height_m", "ball_radius_m"):
            if k in spec:
                out[k] = float(spec[k])
        return out
    # Duck-type a TableSpec.
    return {
        "width_m": float(getattr(spec, "width")),
        "height_m": float(getattr(spec, "height")),
        "ball_radius_m": float(getattr(spec, "ball_radius")),
    }


def _resample(
    trajectory: Sequence[tuple[float, np.ndarray]],
    sample_dt: float,
) -> list[dict]:
    """Resample (t, (N,7)) frames onto a uniform grid via linear interp on
    positions and velocities/spins. Returns ``[{"t": float, "balls": [[...], ...]}]``.

    Each ball entry is ``[x, y, vx, vy, wx, wy, wz]`` so the JS side can
    consume positions plus optionally spins for visual rotation.
    """
    if not trajectory:
        return []

    times = np.asarray([float(t) for t, _ in trajectory], dtype=np.float64)
    arrs = np.stack([np.asarray(a, dtype=np.float64) for _, a in trajectory])
    # Ensure monotonic.
    order = np.argsort(times, kind="stable")
    times = times[order]
    arrs = arrs[order]

    t0, t1 = float(times[0]), float(times[-1])
    if t1 <= t0:
        # Single timestamp — emit a single frame.
        a = arrs[0]
        return [{
            "t": t0,
            "balls": [[float(v) for v in a[i, :7]] for i in range(a.shape[0])],
        }]

    n_steps = max(2, int(np.ceil((t1 - t0) / float(sample_dt))) + 1)
    grid = np.linspace(t0, t1, n_steps)

    # Interpolate each component independently.
    N, D = arrs.shape[1], arrs.shape[2]
    out_arrs = np.empty((len(grid), N, D), dtype=np.float64)
    for i in range(N):
        for d in range(D):
            out_arrs[:, i, d] = np.interp(grid, times, arrs[:, i, d])

    frames: list[dict] = []
    for k, t in enumerate(grid):
        balls = [[float(v) for v in out_arrs[k, i, :7]] for i in range(N)]
        frames.append({"t": float(t), "balls": balls})
    return frames


def _initial_layout_from_spec(spec_dict: dict) -> list[list[float]]:
    """Fallback initial positions when trajectory is empty.

    Tries to import TableState.initial_4ball; if unavailable, uses a
    spec-derived default layout matching its conventions.
    """
    try:
        from billiards.physics.state import TableSpec, TableState
        ts = TableSpec(
            width=spec_dict["width_m"],
            height=spec_dict["height_m"],
            ball_radius=spec_dict["ball_radius_m"],
        )
        s0 = TableState.initial_4ball(spec=ts)
        return [[float(b.x), float(b.y)] for b in s0.balls]
    except Exception:
        L = float(spec_dict["width_m"])
        W = float(spec_dict["height_m"])
        cy = W / 2.0
        offset = 0.1825
        return [
            [3 * L / 4, cy],
            [3 * L / 4, cy - offset],
            [L / 4, cy],
            [L / 2, cy],
        ]


def render_html(
    trajectory: Iterable[tuple[float, np.ndarray]] | None,
    spec=None,
    sample_dt: float = 0.02,
    save_path: Path | str | None = None,
):
    """Build a self-contained HTML replay for ``trajectory``.

    Parameters
    ----------
    trajectory:
        Iterable of ``(t_seconds, (N,7) ndarray)`` pairs. Timestamps need
        not be uniform — frames are resampled to ``sample_dt``.
    spec:
        ``TableSpec``, dict (with keys ``width``/``height``/``ball_radius``
        or their ``*_m`` counterparts), or ``None`` for the medium-table
        default (2.540 × 1.270, ball radius 0.03275).
    sample_dt:
        Target replay sample period. Should be ≤ original frame spacing.
    save_path:
        Optional path to write the inlined HTML to disk.

    Returns
    -------
    IPython.display.HTML
        The inlined viewer, ready to display in a Jupyter cell.
    """
    spec_dict = _coerce_spec(spec)

    traj_list = list(trajectory) if trajectory is not None else []
    frames = _resample(traj_list, float(sample_dt)) if traj_list else []

    if frames:
        spec_payload = dict(spec_dict)
    else:
        # Empty trajectory: still show the table with the initial layout.
        spec_payload = dict(spec_dict)
        spec_payload["initial"] = _initial_layout_from_spec(spec_dict)

    traj_json = json.dumps({"frames": frames}, separators=(",", ":"))
    spec_json = json.dumps(spec_payload, separators=(",", ":"))

    html = _template()
    html = html.replace("{{TRAJ_JSON}}", traj_json)
    html = html.replace("{{SPEC_JSON}}", spec_json)
    html = html.replace("{{SPRITE_WHITE_DATAURL}}",  _sprite_data_url("white"))
    html = html.replace("{{SPRITE_YELLOW_DATAURL}}", _sprite_data_url("yellow"))
    html = html.replace("{{SPRITE_RED_DATAURL}}",    _sprite_data_url("red"))

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html, encoding="utf-8")

    try:
        from IPython.display import HTML as _HTML
        return _HTML(html)
    except Exception:
        # Fallback shim so callers can still get the raw string.
        class _Shim:
            def __init__(self, data: str) -> None:
                self.data = data

            def _repr_html_(self) -> str:
                return self.data

            def __str__(self) -> str:
                return self.data

        return _Shim(html)


def _merge_shot_trajectories(
    shot_trajs: Sequence[Sequence[tuple[float, np.ndarray]]],
) -> list[tuple[float, np.ndarray]]:
    """Concatenate per-shot trajectories into a single inning-global timeline.

    Each shot is assumed to start at local t=0; offsets are derived by
    cumulating each shot's last-frame timestamp. Returns a flat list with
    monotonically non-decreasing global timestamps.
    """
    merged: list[tuple[float, np.ndarray]] = []
    cumulative = 0.0
    for traj in shot_trajs:
        traj_list = list(traj)
        if not traj_list:
            continue
        local_origin = float(traj_list[0][0])
        for t_local, arr in traj_list:
            merged.append((cumulative + (float(t_local) - local_origin), arr))
        last_t = float(traj_list[-1][0]) - local_origin
        cumulative += last_t
    merged.sort(key=lambda tp: tp[0])
    return merged


def render_inning_html(
    shot_trajs: Sequence[Sequence[tuple[float, np.ndarray]]],
    spec=None,
    sample_dt: float = 0.02,
    save_path: Path | str | None = None,
):
    """Hash-merge per-shot trajectories into one viewer.

    Parameters
    ----------
    shot_trajs:
        ``[[(t_local, (N,7)), ...], ...]`` — one list per shot, each with a
        local-zero time origin.
    spec, sample_dt, save_path:
        Forwarded to ``render_html``.
    """
    merged = _merge_shot_trajectories(shot_trajs)
    return render_html(merged, spec=spec, sample_dt=sample_dt, save_path=save_path)
