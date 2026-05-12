"""End-to-end pipeline: detect text regions, then recognize each region.

Typical usage::

    from src.pipeline import HandwritingOCRPipeline

    pipe = HandwritingOCRPipeline.from_config("configs/default.yaml")
    result = pipe.run("data/samples/note.jpg")
    print(result["full_text"])
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import yaml
from PIL import Image

from .detector import TextDetector
from .recognizer import TextRecognizer
from .utils.image import (
    expand_polygon,
    four_point_transform,
    polygon_to_bbox,
    sort_polygons_reading_order,
)
from .utils.viz import draw_ocr_result


ImageInput = Union[str, os.PathLike, np.ndarray, Image.Image]


@dataclass
class PipelineConfig:
    """Configuration object for :class:`HandwritingOCRPipeline`."""

    device: str = "auto"
    detector: Dict[str, Any] = field(default_factory=dict)
    recognizer: Dict[str, Any] = field(default_factory=dict)
    pipeline: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=lambda: {"dir": "data/outputs"})

    @classmethod
    def from_yaml(cls, path: Union[str, os.PathLike]) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            device=raw.get("device", "auto"),
            detector=raw.get("detector", {}) or {},
            recognizer=raw.get("recognizer", {}) or {},
            pipeline=raw.get("pipeline", {}) or {},
            output=raw.get("output", {"dir": "data/outputs"}) or {"dir": "data/outputs"},
        )


def _read_image(source: ImageInput) -> np.ndarray:
    """Return a BGR ``np.ndarray`` for an image path, PIL.Image or array."""
    if isinstance(source, np.ndarray):
        return source
    if isinstance(source, Image.Image):
        return cv2.cvtColor(np.array(source.convert("RGB")), cv2.COLOR_RGB2BGR)
    path = str(source)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    # cv2.imread fails silently on non-ASCII Windows paths -> use np+imdecode.
    with open(path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image: {path}")
    return img


class HandwritingOCRPipeline:
    """Detect + recognize text on Vietnamese handwriting images."""

    def __init__(
        self,
        detector: TextDetector,
        recognizer: TextRecognizer,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.detector = detector
        self.recognizer = recognizer
        self.config = config or PipelineConfig()

    @classmethod
    def from_config(
        cls,
        config_path: Union[str, os.PathLike] = "configs/default.yaml",
        *,
        rec_model_override: Optional[str] = None,
        device_override: Optional[str] = None,
        weights_path_override: Optional[str] = None,
    ) -> "HandwritingOCRPipeline":
        """Build a pipeline from a YAML config, with optional CLI overrides."""
        cfg = PipelineConfig.from_yaml(config_path)
        device = device_override or cfg.device

        det_kwargs = dict(cfg.detector)
        det_kwargs.setdefault("lang", "vi")
        use_gpu = device not in ("cpu",) and device != "auto"
        if device == "auto":
            try:
                import torch
                use_gpu = bool(torch.cuda.is_available())
            except Exception:
                use_gpu = False
        det_kwargs["use_gpu"] = use_gpu

        detector = TextDetector(**det_kwargs)

        rec_kwargs = dict(cfg.recognizer)
        rec_kwargs.pop("batch_size", None)
        if rec_model_override:
            rec_kwargs["model"] = rec_model_override
        rec_kwargs.setdefault("model", "vgg_transformer")
        rec_kwargs["device"] = device
        if weights_path_override is not None:
            rec_kwargs["weights_path"] = weights_path_override
        # Drop falsy weights_path so the recognizer falls back to pretrained.
        if not rec_kwargs.get("weights_path"):
            rec_kwargs.pop("weights_path", None)

        recognizer = TextRecognizer(**rec_kwargs)

        return cls(detector, recognizer, cfg)

    # ------------------------------------------------------------------ run

    def run(
        self,
        image: ImageInput,
        *,
        save_visualization: Optional[bool] = None,
        output_dir: Optional[Union[str, os.PathLike]] = None,
        output_stem: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the full pipeline on a single image.

        Returns
        -------
        dict
            ``{"image_size": (w, h), "num_boxes": int,
               "items": [{"bbox": [[x,y]...], "text": str, "confidence": float}],
               "full_text": str}``
        """
        bgr = _read_image(image)
        h, w = bgr.shape[:2]

        polys = self.detector.detect(bgr)

        items: List[Dict[str, Any]] = []
        if polys:
            expand_ratio = float(self.config.pipeline.get("expand_ratio", 0.05))
            y_tol = int(self.config.pipeline.get("reading_order_y_tolerance", 10))
            batch_size = int(self.config.recognizer.get("batch_size", 16))

            order = sort_polygons_reading_order(polys, y_tolerance=y_tol)
            polys_sorted = [polys[i] for i in order]

            crops: List[Image.Image] = []
            for poly in polys_sorted:
                poly_exp = expand_polygon(poly, ratio=expand_ratio)
                warped = four_point_transform(bgr, poly_exp)
                # VietOCR expects PIL RGB.
                crops.append(Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)))

            rec_results: List = []
            for start in range(0, len(crops), batch_size):
                batch = crops[start : start + batch_size]
                rec_results.extend(self.recognizer.recognize_batch(batch))

            for poly, (text, conf) in zip(polys_sorted, rec_results):
                items.append(
                    {
                        "bbox": poly.astype(float).tolist(),
                        "axis_aligned_bbox": polygon_to_bbox(poly),
                        "text": text,
                        "confidence": float(conf),
                    }
                )

        full_text = "\n".join(it["text"] for it in items if it["text"])

        result: Dict[str, Any] = {
            "image_size": [int(w), int(h)],
            "num_boxes": len(items),
            "items": items,
            "full_text": full_text,
        }

        do_viz = (
            save_visualization
            if save_visualization is not None
            else bool(self.config.pipeline.get("save_visualization", False))
        )
        if do_viz:
            out_dir = Path(output_dir or self.config.output.get("dir", "data/outputs"))
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = output_stem or (
                Path(image).stem if isinstance(image, (str, os.PathLike)) else "image"
            )
            viz = draw_ocr_result(bgr, items)
            viz_path = out_dir / f"{stem}_viz.png"
            viz.save(viz_path)
            result["visualization_path"] = str(viz_path)

        return result

    def save_json(self, result: Dict[str, Any], path: Union[str, os.PathLike]) -> None:
        """Save the pipeline result to a JSON file (UTF-8, indented)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
