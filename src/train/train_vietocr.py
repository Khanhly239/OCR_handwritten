"""Fine-tune VietOCR on a Vietnamese handwriting dataset.

This module is a thin driver around :class:`vietocr.model.trainer.Trainer`
that takes our YAML config (``configs/train_vietocr.yaml``) and runs the
training loop. The pretrained backbone is downloaded automatically on first
use.

Usage::

    # 1) split train_line_annotation.txt into train_split.txt / val_split.txt
    python -m src.train.prepare_data --config configs/train_vietocr.yaml

    # 2) train
    python -m src.train.train_vietocr --config configs/train_vietocr.yaml

    # quick local sanity-check (override iters)
    python -m src.train.train_vietocr --config configs/train_vietocr.yaml \\
        --iters 50 --batch-size 4 --device cpu

After training, the exported weights live at
``models/vietocr/vietocr_seq2seq_inkdata.pth`` and can be loaded by
:class:`src.recognizer.TextRecognizer` via ``weights_path=...``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import yaml


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_vietocr_config(cfg: dict, *, overrides: Optional[dict] = None) -> dict:
    """Translate our YAML config into the dict format VietOCR's Cfg expects."""
    from vietocr.tool.config import Cfg

    overrides = overrides or {}

    backbone = overrides.get("backbone") or cfg.get("backbone", "vgg_seq2seq")
    if backbone not in {"vgg_seq2seq", "vgg_transformer"}:
        raise ValueError(f"Unsupported backbone: {backbone!r}")
    vietocr_cfg = Cfg.load_config_from_name(backbone)

    device = _resolve_device(overrides.get("device") or cfg.get("device", "auto"))
    vietocr_cfg["device"] = device

    if cfg.get("vocab"):
        vietocr_cfg["vocab"] = cfg["vocab"]

    ds = cfg.get("dataset", {}) or {}
    out = cfg.get("output", {}) or {}
    trainer_cfg = dict(cfg.get("trainer", {}) or {})
    aug_cfg = dict(cfg.get("aug", {}) or {})
    opt_cfg = dict(cfg.get("optimizer", {}) or {})

    # Apply CLI overrides to the trainer block (iters / batch-size / etc.).
    if overrides.get("iters") is not None:
        trainer_cfg["iters"] = int(overrides["iters"])
    if overrides.get("batch_size") is not None:
        trainer_cfg["batch_size"] = int(overrides["batch_size"])
    if overrides.get("valid_every") is not None:
        trainer_cfg["valid_every"] = int(overrides["valid_every"])
    if overrides.get("print_every") is not None:
        trainer_cfg["print_every"] = int(overrides["print_every"])

    out_dir = Path(out.get("dir", "models/vietocr"))
    out_dir.mkdir(parents=True, exist_ok=True)
    export_path = out_dir / out.get("export_name", f"{backbone}_inkdata.pth")
    ckpt_path = out_dir / out.get("checkpoint_name", f"{backbone}_inkdata_ckpt.pth")
    log_path = out_dir / out.get("log_name", "train.log")

    # If the user supplied an explicit train annotation override, use it
    # (smoke_train.py points this at a subset).
    train_ann = overrides.get("train_annotation") or ds.get("train_annotation")
    val_ann = overrides.get("valid_annotation") or ds.get("valid_annotation")

    vietocr_cfg["dataset"] = {
        "name": ds.get("name", "inkdata"),
        "data_root": str(ds.get("data_root", "data/data_line")),
        "train_annotation": str(train_ann) if train_ann else None,
        "valid_annotation": str(val_ann) if val_ann else None,
        "image_height": int(ds.get("image_height", 32)),
        "image_min_width": int(ds.get("image_min_width", 32)),
        "image_max_width": int(ds.get("image_max_width", 512)),
    }

    vietocr_cfg["trainer"] = {
        "batch_size": int(trainer_cfg.get("batch_size", 32)),
        "print_every": int(trainer_cfg.get("print_every", 200)),
        "valid_every": int(trainer_cfg.get("valid_every", 2500)),
        "iters": int(trainer_cfg.get("iters", 20000)),
        "export": str(export_path),
        "checkpoint": str(ckpt_path),
        "log": str(log_path),
        "metrics": int(trainer_cfg.get("metrics", 1000)),
    }

    # Augmentation flags. masked_language_model only matters for transformer.
    vietocr_cfg["aug"] = {
        "image_aug": bool(aug_cfg.get("image_aug", True)),
        "masked_language_model": bool(aug_cfg.get("masked_language_model", False)),
    }

    if opt_cfg:
        vietocr_cfg.setdefault("optimizer", {})
        vietocr_cfg["optimizer"].update(
            {
                "max_lr": float(opt_cfg.get("max_lr", 0.0003)),
                "pct_start": float(opt_cfg.get("pct_start", 0.1)),
            }
        )

    # Predictor block: never beamsearch during training (slows validation).
    vietocr_cfg.setdefault("predictor", {})
    vietocr_cfg["predictor"]["beamsearch"] = False

    return vietocr_cfg


def run_training(cfg: dict, *, overrides: Optional[dict] = None) -> Path:
    """Build the VietOCR Trainer and run training. Returns the export path."""
    from vietocr.model.trainer import Trainer

    pretrained = bool(cfg.get("pretrained", True))
    vietocr_cfg = build_vietocr_config(cfg, overrides=overrides)

    trainer_cfg_block = vietocr_cfg["trainer"]
    print("=" * 60)
    print("VietOCR fine-tune")
    print(f"  backbone      : {cfg.get('backbone')}")
    print(f"  device        : {vietocr_cfg['device']}")
    print(f"  pretrained    : {pretrained}")
    print(f"  train_ann     : {vietocr_cfg['dataset']['train_annotation']}")
    print(f"  valid_ann     : {vietocr_cfg['dataset']['valid_annotation']}")
    print(f"  iters         : {trainer_cfg_block['iters']}")
    print(f"  batch_size    : {trainer_cfg_block['batch_size']}")
    print(f"  export        : {trainer_cfg_block['export']}")
    print("=" * 60)

    trainer = Trainer(vietocr_cfg, pretrained=pretrained)
    try:
        trainer.train()
    finally:
        # Save the resolved config alongside the weights for reproducibility.
        out_yaml = Path(trainer_cfg_block["export"]).with_suffix(".yml")
        try:
            trainer.config.save(str(out_yaml))
            print(f"[ok] saved resolved config -> {out_yaml}")
        except Exception as exc:
            print(f"[warn] could not save resolved config: {exc}", file=sys.stderr)
    return Path(trainer_cfg_block["export"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune VietOCR on InkData.")
    parser.add_argument(
        "--config",
        default="configs/train_vietocr.yaml",
        help="Path to YAML config (default: configs/train_vietocr.yaml).",
    )
    parser.add_argument("--backbone", default=None, choices=["vgg_seq2seq", "vgg_transformer"])
    parser.add_argument("--device", default=None, help="auto / cpu / cuda:0 ...")
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    parser.add_argument("--valid-every", type=int, default=None, dest="valid_every")
    parser.add_argument("--print-every", type=int, default=None, dest="print_every")
    parser.add_argument(
        "--train-annotation", default=None, dest="train_annotation",
        help="Override the train annotation file (useful for smoke tests).",
    )
    parser.add_argument(
        "--valid-annotation", default=None, dest="valid_annotation",
        help="Override the validation annotation file.",
    )
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"[error] config not found: {cfg_path}", file=sys.stderr)
        return 2
    cfg = _read_yaml(cfg_path)

    overrides = {
        "backbone": args.backbone,
        "device": args.device,
        "iters": args.iters,
        "batch_size": args.batch_size,
        "valid_every": args.valid_every,
        "print_every": args.print_every,
        "train_annotation": args.train_annotation,
        "valid_annotation": args.valid_annotation,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}

    export_path = run_training(cfg, overrides=overrides)
    print(f"[ok] training complete -> {export_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
