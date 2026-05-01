"""Side-by-side A/B replay for preference labeling.

``render_dual_html(pair_id, traj_A, traj_B, spec, descriptors_A, descriptors_B,
save_path=None)`` produces a single self-contained HTML document showing
both trajectories on synchronized playback, with vote buttons that
``window.postMessage({preference, pair_id})`` so a hosting page can
collect human preferences.

Trajectories are linearly resampled to a common 50 fps grid (per side).
Sprites are inlined as data URLs via ``billiards.render._sprites``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from billiards.render._sprites import load_template, sprite_data_url
from billiards.render.replay import (
    _coerce_spec,
    _initial_layout_from_spec,
    _resample,
)

_DUAL_TEMPLATE_NAME = "dual_viewer.html"
_DEFAULT_SAMPLE_DT = 0.02   # 50 fps


def _frames_payload(
    trajectory: Iterable[tuple[float, np.ndarray]] | None,
    spec_dict: dict,
    sample_dt: float,
) -> dict:
    """Resample a (t, (N,7)) trajectory to a 50 fps frame list."""
    traj_list = list(trajectory) if trajectory is not None else []
    frames = _resample(traj_list, float(sample_dt)) if traj_list else []
    if frames:
        return {"frames": frames}
    # Empty side — emit a single still frame with the standard layout so the
    # viewer still draws balls (mirrors render_html's behavior).
    initial = _initial_layout_from_spec(spec_dict)
    balls = [[float(x), float(y), 0.0, 0.0, 0.0, 0.0, 0.0] for (x, y) in initial]
    return {"frames": [{"t": 0.0, "balls": balls}]}


def _format_descriptors(value) -> str:
    """Coerce descriptor argument to a display string.

    Accepts strings, mappings (rendered as ``key: value`` lines), or
    iterables of (key, value) pairs / strings.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return "\n".join(f"{k}: {v}" for k, v in value.items())
    if isinstance(value, Sequence):
        lines = []
        for item in value:
            if isinstance(item, str):
                lines.append(item)
            elif isinstance(item, Sequence) and len(item) == 2:
                lines.append(f"{item[0]}: {item[1]}")
            else:
                lines.append(str(item))
        return "\n".join(lines)
    return str(value)


def render_dual_html(
    pair_id: str,
    traj_A: Iterable[tuple[float, np.ndarray]] | None,
    traj_B: Iterable[tuple[float, np.ndarray]] | None,
    spec=None,
    descriptors_A=None,
    descriptors_B=None,
    sample_dt: float = _DEFAULT_SAMPLE_DT,
    save_path: Path | str | None = None,
):
    """Build a self-contained dual-table HTML preference viewer.

    Parameters
    ----------
    pair_id:
        Identifier surfaced both visually and via ``postMessage`` votes.
    traj_A, traj_B:
        Iterables of ``(t_seconds, (N,7) ndarray)`` pairs (or ``None`` /
        empty for a still-layout panel).
    spec:
        Same shapes as ``render_html``: ``TableSpec``, dict, or ``None``.
    descriptors_A, descriptors_B:
        Free-form annotations rendered under each panel — useful for showing
        score / cushion-hits / cue-position summaries to a human labeler.
        Accepts a string, a mapping, or an iterable of pairs.
    sample_dt:
        Per-side resample period (default 0.02 s ≈ 50 fps).
    save_path:
        Optional disk write target.

    Returns
    -------
    IPython.display.HTML (or a fallback shim with ``_repr_html_``).
    """
    spec_dict = _coerce_spec(spec)

    payload_A = _frames_payload(traj_A, spec_dict, sample_dt)
    payload_B = _frames_payload(traj_B, spec_dict, sample_dt)

    spec_payload = dict(spec_dict)
    spec_payload["initial"] = _initial_layout_from_spec(spec_dict)

    pair_meta = {"pair_id": str(pair_id)}

    traj_A_json = json.dumps(payload_A, separators=(",", ":"))
    traj_B_json = json.dumps(payload_B, separators=(",", ":"))
    spec_json = json.dumps(spec_payload, separators=(",", ":"))
    pair_meta_json = json.dumps(pair_meta, separators=(",", ":"))

    desc_A_text = _format_descriptors(descriptors_A)
    desc_B_text = _format_descriptors(descriptors_B)

    html = load_template(_DUAL_TEMPLATE_NAME)
    html = html.replace("{{TRAJ_A_JSON}}", traj_A_json)
    html = html.replace("{{TRAJ_B_JSON}}", traj_B_json)
    html = html.replace("{{SPEC_JSON}}", spec_json)
    html = html.replace("{{PAIR_META_JSON}}", pair_meta_json)
    html = html.replace("{{PAIR_ID}}", str(pair_id))
    html = html.replace("{{DESC_A_TEXT}}", _escape_html(desc_A_text))
    html = html.replace("{{DESC_B_TEXT}}", _escape_html(desc_B_text))
    html = html.replace("{{SPRITE_WHITE_DATAURL}}",  sprite_data_url("white"))
    html = html.replace("{{SPRITE_YELLOW_DATAURL}}", sprite_data_url("yellow"))
    html = html.replace("{{SPRITE_RED_DATAURL}}",    sprite_data_url("red"))

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html, encoding="utf-8")

    try:
        from IPython.display import HTML as _HTML
        return _HTML(html)
    except Exception:
        class _Shim:
            def __init__(self, data: str) -> None:
                self.data = data

            def _repr_html_(self) -> str:
                return self.data

            def __str__(self) -> str:
                return self.data

        return _Shim(html)


def _escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )
