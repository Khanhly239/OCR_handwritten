"""Evaluate a (fine-tuned) VietOCR checkpoint on a held-out annotation file.

Reports three metrics:

- **CER** (Character Error Rate) = Levenshtein(pred, ref) / len(ref), averaged.
- **WER** (Word Error Rate)      = same, but tokenized on whitespace.
- **EM**  (Exact Match)          = fraction of samples where pred == ref.

Usage::

    python -m src.train.evaluate \\
        --config configs/train_vietocr.yaml \\
        --weights models/vietocr/vietocr_seq2seq_inkdata.pth \\
        --annotation data/data_line/test_line_annotation.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import yaml
from PIL import Image

from .prepare_data import _read_annotation


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Plain DP Levenshtein distance. Works for strings and word lists."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i, ca in enumerate(a, start=1):
        curr[0] = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost, # substitution
            )
        prev, curr = curr, prev
    return prev[len(b)]


def compute_metrics(
    refs: Sequence[str], hyps: Sequence[str]
) -> dict:
    """Return ``{'cer', 'wer', 'em', 'num_samples'}`` for paired references/hypotheses."""
    if len(refs) != len(hyps):
        raise ValueError(f"len mismatch: refs={len(refs)} hyps={len(hyps)}")

    total_char_ref = 0
    total_char_err = 0
    total_word_ref = 0
    total_word_err = 0
    exact = 0
    for r, h in zip(refs, hyps):
        total_char_err += levenshtein(r, h)
        total_char_ref += max(len(r), 1)

        r_words = r.split()
        h_words = h.split()
        total_word_err += levenshtein(r_words, h_words)
        total_word_ref += max(len(r_words), 1)

        if r == h:
            exact += 1

    return {
        "cer": total_char_err / total_char_ref,
        "wer": total_word_err / total_word_ref,
        "em": exact / len(refs),
        "num_samples": len(refs),
    }


def _build_recognizer(
    backbone: str, device: str, weights_path: Optional[str]
) -> "TextRecognizer":  # noqa: F821
    """Build a TextRecognizer, optionally loading a fine-tuned checkpoint."""
    from src.recognizer import TextRecognizer

    rec = TextRecognizer(model=backbone, device=device, weights_path=weights_path)
    rec._load()  # noqa: SLF001 - eager-load
    return rec


def evaluate(
    cfg: dict,
    *,
    weights_path: Optional[str],
    annotation_path: Path,
    data_root: Path,
    limit: Optional[int] = None,
    device: Optional[str] = None,
    batch_size: int = 32,
    save_predictions: Optional[Path] = None,
) -> dict:
    """Run the recognizer over the dataset and return metric dict."""
    backbone = cfg.get("backbone", "vgg_seq2seq")
    use_device = device or cfg.get("device", "auto")

    rows = _read_annotation(annotation_path)
    if limit is not None:
        rows = rows[: int(limit)]

    rec = _build_recognizer(backbone, use_device, weights_path)
    print(f"[info] recognizer loaded ({backbone}, device={rec.device})")
    print(f"[info] evaluating {len(rows)} samples from {annotation_path.name}")

    refs: List[str] = []
    hyps: List[str] = []
    img_paths: List[str] = []
    missing = 0

    t0 = time.time()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        pil_images = []
        kept_rows = []
        for img_rel, label in batch:
            full = data_root / img_rel
            if not full.is_file():
                missing += 1
                continue
            try:
                pil_images.append(Image.open(full).convert("RGB"))
                kept_rows.append((img_rel, label))
            except Exception as exc:
                print(f"[warn] failed to read {full}: {exc}", file=sys.stderr)
                missing += 1
        if not pil_images:
            continue

        batch_preds = rec.recognize_batch(pil_images)
        for (img_rel, label), (text, _conf) in zip(kept_rows, batch_preds):
            refs.append(label)
            hyps.append(text)
            img_paths.append(img_rel)

        if (start // batch_size) % 10 == 0 and start > 0:
            elapsed = time.time() - t0
            done = len(refs)
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  ... {done}/{len(rows)}  ({rate:.1f} samples/s)")

    elapsed = time.time() - t0

    if missing:
        print(f"[warn] {missing} samples skipped (image missing or unreadable)")

    if not refs:
        print("[error] no samples evaluated", file=sys.stderr)
        return {"error": "no samples"}

    metrics = compute_metrics(refs, hyps)
    metrics["elapsed_seconds"] = round(elapsed, 2)
    metrics["throughput_samples_per_sec"] = round(len(refs) / elapsed, 2) if elapsed > 0 else 0.0

    print("\n=== Evaluation results ===")
    print(f"  samples:     {metrics['num_samples']}")
    print(f"  CER:         {metrics['cer'] * 100:.2f} %")
    print(f"  WER:         {metrics['wer'] * 100:.2f} %")
    print(f"  Exact match: {metrics['em'] * 100:.2f} %")
    print(f"  elapsed:     {metrics['elapsed_seconds']} s "
          f"({metrics['throughput_samples_per_sec']} samples/s)")

    if save_predictions:
        save_predictions.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metrics": metrics,
            "predictions": [
                {"image": ip, "reference": r, "hypothesis": h}
                for ip, r, h in zip(img_paths, refs, hyps)
            ],
        }
        with open(save_predictions, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[ok] wrote predictions -> {save_predictions}")

    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a VietOCR checkpoint on a TSV annotation file."
    )
    parser.add_argument("--config", default="configs/train_vietocr.yaml")
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to a fine-tuned .pth file. If omitted, the pretrained backbone is used.",
    )
    parser.add_argument(
        "--annotation",
        default=None,
        help="Annotation file to evaluate on. Defaults to dataset.test_annotation from the config.",
    )
    parser.add_argument("--data-root", default=None, dest="data_root")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument(
        "--save-predictions",
        default=None,
        dest="save_predictions",
        help="Optional: write per-sample predictions to this JSON file.",
    )
    args = parser.parse_args(argv)

    cfg = _read_yaml(Path(args.config))
    ds = cfg.get("dataset", {}) or {}
    ann = Path(args.annotation or ds.get("test_annotation"))
    data_root = Path(args.data_root or ds.get("data_root", "data/data_line"))

    if not ann.is_file():
        print(f"[error] annotation file not found: {ann}", file=sys.stderr)
        return 2

    evaluate(
        cfg,
        weights_path=args.weights,
        annotation_path=ann,
        data_root=data_root,
        limit=args.limit,
        device=args.device,
        batch_size=args.batch_size,
        save_predictions=Path(args.save_predictions) if args.save_predictions else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
