"""Command-line interface for the handwriting OCR pipeline.

Example::

    python -m src.cli --image data/samples/note.jpg --save-viz
    python -m src.cli --image note.jpg --rec-model vgg_seq2seq --output-json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import HandwritingOCRPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="handw",
        description="Vietnamese handwriting OCR: PaddleOCR (detect) + VietOCR (recognize).",
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML config file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--rec-model",
        choices=["vgg_transformer", "vgg_seq2seq"],
        default=None,
        help="Override the VietOCR backbone from the config.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device: 'auto', 'cpu', 'cuda:0', ...",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to a fine-tuned VietOCR .pth checkpoint (overrides config).",
    )
    parser.add_argument(
        "--save-viz",
        action="store_true",
        help="Save a visualization PNG (bbox + text) next to the JSON output.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="If set, write the JSON result to this path.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress printing the JSON to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not Path(args.image).is_file():
        print(f"[error] image not found: {args.image}", file=sys.stderr)
        return 2

    pipe = HandwritingOCRPipeline.from_config(
        args.config,
        rec_model_override=args.rec_model,
        device_override=args.device,
        weights_path_override=args.weights,
    )

    output_stem = Path(args.image).stem
    result = pipe.run(
        args.image,
        save_visualization=args.save_viz,
        output_stem=output_stem,
    )

    if args.output_json:
        pipe.save_json(result, args.output_json)
        print(f"[ok] wrote JSON -> {args.output_json}", file=sys.stderr)

    if not args.quiet:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
