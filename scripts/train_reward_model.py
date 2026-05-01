"""Train the preference reward model on a JSONL pairs file.

Usage:
    uv run python scripts/train_reward_model.py \
        --pairs_path data/preference_pairs_5k.jsonl \
        --epochs 20 --out_dir models/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from billiards.preference.dataset import load_pairs  # noqa: E402
from billiards.reward_model import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_path", type=str,
                        default="data/preference_pairs_5k.jsonl")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default="models/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--val_frac", type=float, default=0.15)
    args = parser.parse_args()

    pairs = load_pairs(args.pairs_path)
    print(f"Loaded {len(pairs)} pairs from {args.pairs_path}")
    if not pairs:
        raise SystemExit("No pairs to train on.")

    out_dir = Path(args.out_dir)
    model, history = train(
        pairs,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        seed=args.seed,
        save_dir=out_dir,
        verbose=True,
    )
    final_acc = history["val_acc"][-1] if history["val_acc"] else float("nan")
    final_loss = history["val_loss"][-1] if history["val_loss"] else float("nan")
    print(f"\nFinal val_acc={final_acc:.3f}  val_loss={final_loss:.4f}")
    print(f"Saved model -> {out_dir / 'reward_model.pt'}")
    print(f"Saved history -> {out_dir / 'reward_history.json'}")


if __name__ == "__main__":
    main()
