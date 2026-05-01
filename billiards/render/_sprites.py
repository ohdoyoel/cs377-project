"""Shared sprite + viewer-template loading for replay renderers.

Both ``replay.render_html`` (single-table) and ``dual_replay.render_dual_html``
(side-by-side A/B comparison) inline the same per-ball sprite PNGs as
``data:image/png;base64,...`` URLs. This module owns the cache and the
small bit of path logic so the two renderers stay in sync.
"""

from __future__ import annotations

import base64
from pathlib import Path

_SPRITES_DIR = Path(__file__).with_name("sprites")
_RENDER_DIR = Path(__file__).parent

_SPRITE_NAMES = ("white", "yellow", "red")
_SPRITE_CACHE: dict[str, str] = {}
_TEMPLATE_CACHE: dict[str, str] = {}


def sprite_data_url(name: str) -> str:
    """Return a ``data:image/png;base64,...`` URL for ball sprite ``name``.

    Cached per-process to avoid re-encoding on every render.
    """
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


def all_sprite_data_urls() -> dict[str, str]:
    """Eager-load all sprites; returns a dict ``{'white': '...', ...}``."""
    return {n: sprite_data_url(n) for n in _SPRITE_NAMES}


def load_template(filename: str) -> str:
    """Read ``billiards/render/<filename>`` as text, cached per-process."""
    if filename not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE[filename] = (_RENDER_DIR / filename).read_text(encoding="utf-8")
    return _TEMPLATE_CACHE[filename]
