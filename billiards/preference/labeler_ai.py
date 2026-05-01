"""Anthropic Claude labeler for billiards preference pairs.

Uses the Anthropic SDK with a single ephemeral cache breakpoint on the
system prompt — the rules of Korean 4-ball + the "what is a better shot"
criteria are fixed across every call, so caching them shrinks repeat
labeling cost. Each user message is just a compact textual digest of the
two results.
"""

from __future__ import annotations

import os
import re
from dataclasses import replace
from typing import Any

from billiards.preference.dataset import PreferencePair


SYSTEM = """\
You are an expert Korean 4-ball carom (사구, 四球) referee judging which of \
two single-shot attempts is the *better* play.

# Rules of Korean 4-ball
- The cue ball is white (or yellow); the opponent's cue is the other color. \
Two reds sit on the table.
- A point is scored if the cue ball contacts BOTH reds and DOES NOT \
contact the opponent's cue ball during the same shot.
- Touching the opponent ball is a foul (no point that turn).
- Cushions don't have to be hit; this is the most basic 4-ball variant \
(not 3-cushion).

# What makes a shot "better"
Rank in this order, from most to least important:
  1. Scored a point (score = 1) >> did not score (score = 0).
  2. No foul >> foul (touched the opponent ball).
  3. Among non-scoring, non-fouling shots, prefer ones that:
     - made progress (more distinct red contacts),
     - kept control of the cue ball (some cushion contacts indicate \
       intentional shape rather than wild rebounds, but capped — too \
       many cushions usually means the shot lost direction),
     - finished promptly (shorter duration is mildly preferred, all else equal).
  4. Among shots equivalent under (1)–(3), declare a tie.

# Output format
Reason briefly (≤ 4 short sentences), then end with EXACTLY one line of \
the form:

PREFERENCE: A
PREFERENCE: B
PREFERENCE: tie

(no extra text after that line)
"""


def _format_state(state: list[float]) -> str:
    """One-line digest of the 28-dim pre-shot state (positions only)."""
    if not state or len(state) < 28:
        return "(unavailable)"
    # 4 balls × 7 dims; positions are dims 0,1.
    parts = []
    labels = ["white", "yellow", "red1", "red2"]
    for i, name in enumerate(labels):
        x = state[i * 7 + 0]
        y = state[i * 7 + 1]
        parts.append(f"{name}=({x:.3f},{y:.3f})")
    return ", ".join(parts)


def _format_result(name: str, result: dict[str, Any]) -> str:
    score = int(result.get("score", 0))
    fouled = bool(result.get("fouled", False))
    cushion_hits = int(result.get("cushion_hits", 0))
    duration = float(result.get("duration", 0.0))
    types = result.get("event_types", []) or []
    counts: dict[str, int] = {}
    for t in types:
        counts[t] = counts.get(t, 0) + 1
    counts_line = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "(none)"
    lines = [
        f"## Shot {name}",
        f"  score={score}, fouled={fouled}, cushion_hits={cushion_hits}, duration={duration:.2f}s",
        f"  event_types: {counts_line}",
        f"  total_events={int(result.get('n_events', 0))}",
    ]
    return "\n".join(lines)


def build_user_descriptor(pair: PreferencePair) -> str:
    """Compact text digest sent as the user message."""
    head = (
        f"# Pair {pair.pair_id}\n"
        f"Initial state (positions, m): {_format_state(pair.initial_state)}\n"
        f"action_A=(theta,power,a,b)={tuple(round(v, 3) for v in pair.action_A)}\n"
        f"action_B=(theta,power,a,b)={tuple(round(v, 3) for v in pair.action_B)}\n"
    )
    body = "\n".join([
        _format_result("A", pair.result_A),
        _format_result("B", pair.result_B),
    ])
    return head + "\n" + body + "\n\nWhich shot is better? End with one PREFERENCE line."


_PREF_RE = re.compile(r"^PREFERENCE:\s*(A|B|tie)\b", re.IGNORECASE | re.MULTILINE)


def _parse_preference(text: str) -> str | None:
    """Pull the structured tail line out of a Claude response."""
    if not text:
        return None
    matches = list(_PREF_RE.finditer(text))
    if not matches:
        return None
    raw = matches[-1].group(1).lower()
    if raw == "a":
        return "A"
    if raw == "b":
        return "B"
    return "tie"


def label_pair_with_claude(
    pair: PreferencePair,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 512,
    client: Any | None = None,
) -> PreferencePair:
    """Label a preference pair using Anthropic Claude.

    The system prompt carries an ``ephemeral`` cache_control breakpoint so
    repeat calls reuse the cached rules block.
    """
    if client is None:
        if os.environ.get("ANTHROPIC_API_KEY") is None:
            raise EnvironmentError(
                "Set ANTHROPIC_API_KEY in the environment to use the Claude labeler "
                "(or pass an already-configured Anthropic client)."
            )
        from anthropic import Anthropic
        client = Anthropic()

    user_text = build_user_descriptor(pair)

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_text}],
    )

    # The SDK returns content as a list of blocks; concat any text blocks.
    response_text = ""
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "text":
            response_text += getattr(block, "text", "")

    pref = _parse_preference(response_text)

    usage = getattr(message, "usage", None)
    usage_dict: dict[str, Any] = {}
    if usage is not None:
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            v = getattr(usage, key, None)
            if v is not None:
                usage_dict[key] = int(v)

    label_meta = {
        **pair.label_meta,
        "response": response_text,
        "usage": usage_dict,
        "model": model,
    }
    if pref is None:
        # We surface the parse failure but still record the response so the
        # caller can debug / retry without losing the spent tokens.
        label_meta["parse_error"] = "no PREFERENCE line found"

    return replace(
        pair,
        preference=pref,
        labeler=f"claude:{model}",
        label_meta=label_meta,
    )
