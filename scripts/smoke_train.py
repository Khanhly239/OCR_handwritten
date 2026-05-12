"""Quick local smoke test for the VietOCR fine-tune pipeline.

Subsamples the train/val splits to ~100/~20 lines and runs the trainer for
only a few iterations. Intended to catch wiring bugs before launching a real
Colab GPU run -- NOT to produce a usable model.

Usage::

    python scripts/smoke_train.py
    python scripts/smoke_train.py --iters 100 --batch-size 4 --device cpu
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.prepare_data import _read_annotation, _write_annotation  # noqa: E402
from src.train.train_vietocr import _read_yaml, run_training  # noqa: E402


def _subsample(src: Path, dst: Path, n: int, seed: int) -> int:
    rows = _read_annotation(src)
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n]
    return _write_annotation(dst, rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the VietOCR fine-tune pipeline.")
    parser.add_argument("--config", default="configs/train_vietocr.yaml")
    parser.add_argument("--train-size", type=int, default=100, dest="train_size")
    parser.add_argument("--val-size", type=int, default=20, dest="val_size")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4, dest="batch_size")
    parser.add_argument("--device", default="cpu", help="auto / cpu / cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    cfg = _read_yaml(Path(args.config))
    full_train = Path(cfg["dataset"]["train_annotation"])
    full_val = Path(cfg["dataset"]["valid_annotation"])

    if not full_train.is_file() or not full_val.is_file():
        print(
            "[error] train/val split files not found. Run `python -m src.train.prepare_data` first.",
            file=sys.stderr,
        )
        return 2

    smoke_dir = ROOT / "data" / "data_line" / "_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke_train = smoke_dir / "train.txt"
    smoke_val = smoke_dir / "val.txt"

    n_train = _subsample(full_train, smoke_train, args.train_size, args.seed)
    n_val = _subsample(full_val, smoke_val, args.val_size, args.seed)
    print(f"[ok] subsampled {n_train} train / {n_val} val rows for smoke test")

    overrides = {
        "iters": args.iters,
        "batch_size": args.batch_size,
        "device": args.device,
        "train_annotation": str(smoke_train),
        "valid_annotation": str(smoke_val),
        # Validate on every iteration block so we exercise the eval path too.
        "valid_every": max(1, args.iters // 2),
        "print_every": max(1, args.iters // 10),
    }

    run_training(cfg, overrides=overrides)
    print("[ok] smoke test completed without errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
