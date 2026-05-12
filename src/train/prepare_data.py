"""Prepare the InkData annotations for fine-tuning VietOCR.

This script:

1. Reads the source annotation file (TSV: ``relative_image_path\\tlabel``).
2. Verifies every image exists and every line is well-formed.
3. Computes the character vocabulary and reports characters that are not in
   VietOCR's default Vietnamese vocab.
4. Shuffles deterministically and splits into ``train_split.txt`` and
   ``val_split.txt`` according to ``val_ratio``.

Usage::

    python -m src.train.prepare_data --config configs/train_vietocr.yaml
    python -m src.train.prepare_data --config configs/train_vietocr.yaml --strict
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

import yaml


# Accept either a TAB or a run of 2+ spaces as the path/label separator. We
# require 2+ spaces (not just one) because Vietnamese labels contain plenty
# of single spaces between words.
_SEPARATOR_RE = re.compile(r"\t+| {2,}")


# VietOCR's built-in Vietnamese vocabulary (verbatim from the package).
DEFAULT_VIETOCR_VOCAB = (
    "aA\u00e0\u00c0\u1ea3\u1ea2\u00e3\u00c3\u00e1\u00c1\u1ea1\u1ea0"
    "\u0103\u0102\u1eb1\u1eb0\u1eb3\u1eb2\u1eb5\u1eb4\u1eaf\u1eae\u1eb7\u1eb6"
    "\u00e2\u00c2\u1ea7\u1ea6\u1ea9\u1ea8\u1eab\u1eaa\u1ea5\u1ea4\u1ead\u1eac"
    "bBcCdD\u0111\u0110eE\u00e8\u00c8\u1ebb\u1eba\u1ebd\u1ebc\u00e9\u00c9"
    "\u1eb9\u1eb8\u00ea\u00ca\u1ec1\u1ec0\u1ec3\u1ec2\u1ec5\u1ec4\u1ebf\u1ebe"
    "\u1ec7\u1ec6fFgGhHiI\u00ec\u00cc\u1ec9\u1ec8\u0129\u0128\u00ed\u00cd"
    "\u1ecb\u1ecajJkKlLmMnNoO\u00f2\u00d2\u1ecf\u1ece\u00f5\u00d5\u00f3\u00d3"
    "\u1ecd\u1ecc\u00f4\u00d4\u1ed3\u1ed2\u1ed5\u1ed4\u1ed7\u1ed6\u1ed1\u1ed0"
    "\u1ed9\u1ed8\u01a1\u01a0\u1edd\u1edc\u1edf\u1ede\u1ee1\u1ee0\u1edb\u1eda"
    "\u1ee3\u1ee2pPqQrRsStTuU\u00f9\u00d9\u1ee7\u1ee6\u0169\u0168\u00fa\u00da"
    "\u1ee5\u1ee4\u01b0\u01af\u1eeb\u1eea\u1eed\u1eec\u1eef\u1eee\u1ee9\u1ee8"
    "\u1ef1\u1ef0vVwWxXyY\u1ef3\u1ef2\u1ef7\u1ef6\u1ef9\u1ef8\u00fd\u00dd"
    "\u1ef5\u1ef4zZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
)


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _read_annotation(path: Path) -> List[Tuple[str, str]]:
    """Return a list of ``(image_path, label)`` tuples.

    Accepts both TAB-separated (e.g. the original InkData annotation) and
    multi-space-separated (e.g. the address subset) lines so that mixed
    files load correctly.
    """
    out: List[Tuple[str, str]] = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = _SEPARATOR_RE.split(line, maxsplit=1)
            if len(parts) != 2:
                skipped += 1
                continue
            img, label = parts
            label = label.strip()
            img = img.strip()
            if not img or not label:
                skipped += 1
                continue
            out.append((img, label))
    if skipped:
        print(
            f"[warn] {skipped} unparseable lines in {path.name} (missing TAB / multi-space separator)",
            file=sys.stderr,
        )
    return out


def _write_annotation(path: Path, rows: Iterable[Tuple[str, str]]) -> int:
    """Write rows as TAB-separated UTF-8 (VietOCR's loader expects TAB)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for img, label in rows:
            f.write(f"{img}\t{label}\n")
            n += 1
    return n


def _check_images(rows: List[Tuple[str, str]], data_root: Path) -> List[Tuple[str, str]]:
    """Drop rows whose image file does not exist under ``data_root``."""
    keep: List[Tuple[str, str]] = []
    missing = 0
    for img, label in rows:
        if (data_root / img).is_file():
            keep.append((img, label))
        else:
            missing += 1
    if missing:
        print(f"[warn] {missing} rows skipped because the image was not found", file=sys.stderr)
    return keep


def _check_vocab(rows: List[Tuple[str, str]], vocab: str) -> List[Tuple[str, int]]:
    """Return a list of ``(char, count)`` for characters not in ``vocab``."""
    counter: Counter = Counter()
    vocab_set = set(vocab)
    for _, label in rows:
        for ch in label:
            if ch not in vocab_set:
                counter[ch] += 1
    return sorted(counter.items(), key=lambda kv: -kv[1])


def split(
    rows: List[Tuple[str, str]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Deterministically split ``rows`` into (train, val)."""
    rng = random.Random(seed)
    shuffled = rows.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Split the InkData train annotations into train/val and "
                    "check vocabulary coverage."
    )
    parser.add_argument(
        "--config",
        default="configs/train_vietocr.yaml",
        help="Path to the training config (default: configs/train_vietocr.yaml).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail with a non-zero exit code if any out-of-vocab characters are found.",
    )
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    cfg = _read_yaml(cfg_path)

    split_cfg = cfg.get("split", {}) or {}
    source = Path(split_cfg.get("source", "data/data_line/train_line_annotation.txt"))
    val_ratio = float(split_cfg.get("val_ratio", 0.10))
    seed = int(split_cfg.get("seed", 42))

    data_root = Path(cfg.get("dataset", {}).get("data_root", "data/data_line"))
    train_out = Path(cfg["dataset"]["train_annotation"])
    val_out = Path(cfg["dataset"]["valid_annotation"])

    print(f"[info] source       = {source}")
    print(f"[info] data_root    = {data_root}")
    print(f"[info] val_ratio    = {val_ratio}  seed = {seed}")
    print(f"[info] train output = {train_out}")
    print(f"[info] val output   = {val_out}")

    if not source.is_file():
        print(f"[error] source annotation file not found: {source}", file=sys.stderr)
        return 2

    rows = _read_annotation(source)
    print(f"[info] read {len(rows)} rows from {source.name}")

    rows = _check_images(rows, data_root)
    print(f"[info] {len(rows)} rows have a matching image under {data_root}")

    user_vocab = cfg.get("vocab") or DEFAULT_VIETOCR_VOCAB
    oov = _check_vocab(rows, user_vocab)
    if oov:
        print(f"[warn] {len(oov)} out-of-vocab characters detected (top 15):")
        for ch, n in oov[:15]:
            display = ch.encode("unicode_escape").decode("ascii")
            print(f"       {display!r}  count={n}")
        if args.strict:
            print("[error] --strict was set; aborting.", file=sys.stderr)
            return 3
    else:
        print("[ok] every label character is covered by the configured vocab")

    train, val = split(rows, val_ratio=val_ratio, seed=seed)
    n_train = _write_annotation(train_out, train)
    n_val = _write_annotation(val_out, val)
    print(f"[ok] wrote {n_train} train rows -> {train_out}")
    print(f"[ok] wrote {n_val} val   rows -> {val_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
