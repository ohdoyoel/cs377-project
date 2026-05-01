"""Generate a heuristic-labeled preference dataset for Phase D.

Usage:
    uv run python scripts/generate_preference_dataset.py \
        --n_pairs 5000 --env mix --out_path data/preference_pairs_5k.jsonl

For each ``pair_idx`` we use a deterministic seed = ``seed + pair_idx``,
sample two random actions, run env.step on two fresh env instances seeded
identically, label the pair with the heuristic labeler, and append to a
JSONL file. Every 500 pairs we print pair count, A/B/tie distribution and
mean |score_A - score_B|.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from billiards.env import Billiards4BallEnv  # noqa: E402
from billiards.inning_env import Billiards4BallInningEnv  # noqa: E402
from billiards.preference.dataset import (  # noqa: E402
    PreferencePair,
    _result_digest,
    _sample_random_action,
    append_pair,
)
from billiards.preference.labeler_heuristic import label_pair_heuristic  # noqa: E402

ENV_CHOICES = ("single", "inning", "mix")


def _make_env(env_kind: str):
    if env_kind == "single":
        return Billiards4BallEnv()
    if env_kind == "inning":
        return Billiards4BallInningEnv(max_shots=1)
    raise ValueError(env_kind)


def _generate_one(env_kind: str, env_seed: int) -> tuple[PreferencePair, dict]:
    rng = np.random.default_rng(env_seed * 7919 + 1)

    env_A = _make_env(env_kind)
    obs_A, _ = env_A.reset(seed=env_seed)
    initial_state = [float(v) for v in np.asarray(obs_A).reshape(-1).tolist()]
    action_A = _sample_random_action(rng)
    _, _, _, _, info_A = env_A.step(action_A)

    env_B = _make_env(env_kind)
    env_B.reset(seed=env_seed)
    action_B = _sample_random_action(rng)
    _, _, _, _, info_B = env_B.step(action_B)

    import uuid
    pair = PreferencePair(
        pair_id=uuid.uuid4().hex,
        initial_state=initial_state,
        action_A=[float(v) for v in action_A.tolist()],
        action_B=[float(v) for v in action_B.tolist()],
        result_A=_result_digest(info_A),
        result_B=_result_digest(info_B),
        preference=None,
        labeler="unlabeled",
        label_meta={"seed": env_seed, "env_kind": env_kind},
    )
    pair = label_pair_heuristic(pair)
    diag = {
        "score_A": pair.label_meta["score_A"],
        "score_B": pair.label_meta["score_B"],
        "preference": pair.preference,
    }
    return pair, diag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pairs", type=int, default=5000)
    parser.add_argument("--env", choices=ENV_CHOICES, default="mix")
    parser.add_argument("--out_path", type=str,
                        default="data/preference_pairs_5k.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_path = Path(args.out_path)
    if out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pref_counter: Counter[str] = Counter()
    abs_diffs: list[float] = []

    start = time.time()
    for pair_idx in range(int(args.n_pairs)):
        env_seed = int(args.seed) + pair_idx
        if args.env == "mix":
            kind = "single" if (pair_idx % 2 == 0) else "inning"
        else:
            kind = args.env

        pair, diag = _generate_one(kind, env_seed)
        append_pair(pair, path=out_path)

        pref_counter[diag["preference"] or "none"] += 1
        abs_diffs.append(abs(diag["score_A"] - diag["score_B"]))

        if (pair_idx + 1) % 500 == 0 or (pair_idx + 1) == args.n_pairs:
            elapsed = time.time() - start
            mean_abs = float(np.mean(abs_diffs)) if abs_diffs else 0.0
            print(
                f"[{pair_idx + 1}/{args.n_pairs}] "
                f"A={pref_counter.get('A', 0)} "
                f"B={pref_counter.get('B', 0)} "
                f"tie={pref_counter.get('tie', 0)} "
                f"none={pref_counter.get('none', 0)} "
                f"mean|sA-sB|={mean_abs:.3f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    elapsed = time.time() - start
    print(f"\nWrote {out_path} ({out_path.stat().st_size} bytes) in {elapsed:.1f}s")
    print(json.dumps({"n_pairs": args.n_pairs, "distribution": dict(pref_counter)},
                     indent=2))


if __name__ == "__main__":
    main()
