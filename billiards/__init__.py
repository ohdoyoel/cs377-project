"""Korean 4-ball carom billiards RL environment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from billiards.env import Billiards4BallEnv as _Billiards4BallEnv  # noqa: F401
    from billiards.render.replay import render_html as _render_html  # noqa: F401

__all__ = ["Billiards4BallEnv", "render_html"]


def __getattr__(name: str) -> Any:
    if name == "render_html":
        from billiards.render.replay import render_html
        return render_html
    if name == "Billiards4BallEnv":
        from billiards.env import Billiards4BallEnv
        return Billiards4BallEnv
    raise AttributeError(f"module 'billiards' has no attribute {name!r}")
