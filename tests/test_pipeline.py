"""Unit + smoke tests for Phase 1 modules.

Heavy model downloads (PaddleOCR + VietOCR weights ~300 MB) are mocked so
these tests stay fast and offline-friendly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.image import (  # noqa: E402
    _order_points,
    expand_polygon,
    four_point_transform,
    polygon_to_bbox,
    sort_polygons_reading_order,
)


# ----------------------------------------------------------- image utilities

def test_order_points_orients_quadrilateral():
    pts = np.array([[10, 10], [50, 50], [50, 10], [10, 50]], dtype=np.float32)
    ordered = _order_points(pts)
    assert ordered.tolist() == [[10, 10], [50, 10], [50, 50], [10, 50]]


def test_four_point_transform_preserves_size_for_axis_aligned_box():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[10:40, 20:80] = 255
    poly = np.array([[20, 10], [80, 10], [80, 40], [20, 40]], dtype=np.float32)
    crop = four_point_transform(img, poly)
    assert crop.shape == (30, 60, 3)
    # Interior should be fully white (boundary pixels may sample outside).
    assert (crop[5:25, 5:55] == 255).all()


def test_expand_polygon_makes_box_larger():
    poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    expanded = expand_polygon(poly, ratio=0.2)
    bbox = polygon_to_bbox(expanded)
    assert bbox[0] < 0
    assert bbox[1] < 0
    assert bbox[2] > 10
    assert bbox[3] > 10


def test_expand_polygon_zero_ratio_is_noop():
    poly = np.array([[1, 1], [3, 1], [3, 4], [1, 4]], dtype=np.float32)
    np.testing.assert_array_equal(expand_polygon(poly, ratio=0.0), poly)


def test_sort_reading_order_groups_rows_then_columns():
    # Two rows, two columns each.
    polys = [
        np.array([[100, 0], [120, 0], [120, 20], [100, 20]], dtype=np.float32),  # row 0, right
        np.array([[0, 100], [20, 100], [20, 120], [0, 120]], dtype=np.float32),   # row 1, left
        np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=np.float32),         # row 0, left
        np.array([[100, 100], [120, 100], [120, 120], [100, 120]], dtype=np.float32),  # row 1, right
    ]
    order = sort_polygons_reading_order(polys, y_tolerance=10)
    assert order == [2, 0, 1, 3]


def test_sort_reading_order_empty_input():
    assert sort_polygons_reading_order([]) == []


def test_polygon_to_bbox_returns_axis_aligned_corners():
    poly = np.array([[5, 7], [25, 9], [27, 30], [4, 28]], dtype=np.float32)
    assert polygon_to_bbox(poly) == [4, 7, 27, 30]


# ----------------------------------------------------------------- pipeline

class _FakeDetector:
    def __init__(self, polys):
        self._polys = polys

    def detect(self, image):
        return list(self._polys)


class _FakeRecognizer:
    def __init__(self, texts):
        self._texts = texts

    def recognize_batch(self, images):
        out = []
        for i, _ in enumerate(images):
            t = self._texts[i] if i < len(self._texts) else "..."
            out.append((t, 0.9))
        return out


def test_pipeline_run_with_mocked_components(tmp_path):
    from src.pipeline import HandwritingOCRPipeline, PipelineConfig

    img = np.full((200, 400, 3), 255, dtype=np.uint8)
    polys = [
        np.array([[10, 10], [120, 10], [120, 40], [10, 40]], dtype=np.float32),
        np.array([[10, 80], [200, 80], [200, 120], [10, 120]], dtype=np.float32),
    ]
    pipe = HandwritingOCRPipeline(
        detector=_FakeDetector(polys),
        recognizer=_FakeRecognizer(["xin chao", "tieng viet"]),
        config=PipelineConfig(),
    )

    result = pipe.run(img)

    assert result["num_boxes"] == 2
    assert result["image_size"] == [400, 200]
    assert [it["text"] for it in result["items"]] == ["xin chao", "tieng viet"]
    assert result["full_text"] == "xin chao\ntieng viet"
    for it in result["items"]:
        assert len(it["bbox"]) == 4
        assert len(it["axis_aligned_bbox"]) == 4
        assert 0.0 <= it["confidence"] <= 1.0


def test_pipeline_run_handles_no_detections():
    from src.pipeline import HandwritingOCRPipeline, PipelineConfig

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    pipe = HandwritingOCRPipeline(
        detector=_FakeDetector([]),
        recognizer=_FakeRecognizer([]),
        config=PipelineConfig(),
    )
    result = pipe.run(img)
    assert result["num_boxes"] == 0
    assert result["items"] == []
    assert result["full_text"] == ""


def test_pipeline_save_visualization(tmp_path):
    from src.pipeline import HandwritingOCRPipeline, PipelineConfig

    img = np.full((150, 300, 3), 200, dtype=np.uint8)
    polys = [np.array([[10, 10], [100, 10], [100, 40], [10, 40]], dtype=np.float32)]
    pipe = HandwritingOCRPipeline(
        detector=_FakeDetector(polys),
        recognizer=_FakeRecognizer(["hello"]),
        config=PipelineConfig(output={"dir": str(tmp_path)}),
    )

    result = pipe.run(img, save_visualization=True, output_dir=tmp_path, output_stem="t")
    viz_path = Path(result["visualization_path"])
    assert viz_path.is_file()
    assert viz_path.suffix == ".png"


def test_pipeline_save_json_writes_utf8(tmp_path):
    from src.pipeline import HandwritingOCRPipeline, PipelineConfig

    pipe = HandwritingOCRPipeline(
        detector=_FakeDetector([]),
        recognizer=_FakeRecognizer([]),
        config=PipelineConfig(),
    )
    out = tmp_path / "out.json"
    pipe.save_json({"text": "ti\u1ebfng vi\u1ec7t"}, out)
    content = out.read_text(encoding="utf-8")
    assert "ti\u1ebfng vi\u1ec7t" in content


# ------------------------------------------------------------------- config

def test_pipeline_config_loads_default_yaml():
    from src.pipeline import PipelineConfig

    cfg = PipelineConfig.from_yaml(ROOT / "configs" / "default.yaml")
    assert cfg.detector["lang"] == "vi"
    assert cfg.recognizer["model"] in {"vgg_transformer", "vgg_seq2seq"}
