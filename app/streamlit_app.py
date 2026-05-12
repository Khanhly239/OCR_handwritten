"""Streamlit web UI for the Vietnamese handwriting OCR pipeline.

Run with::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Allow `import src.*` when launched from any cwd.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detector import TextDetector  # noqa: E402
from src.recognizer import TextRecognizer  # noqa: E402
from src.pipeline import HandwritingOCRPipeline, PipelineConfig  # noqa: E402
from src.utils.viz import draw_ocr_result  # noqa: E402


st.set_page_config(
    page_title="Vietnamese Handwriting OCR",
    page_icon="\u270d\ufe0f",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading PaddleOCR detector...")
def get_detector(
    det_db_box_thresh: float,
    det_db_unclip_ratio: float,
    use_angle_cls: bool,
    use_gpu: bool,
) -> TextDetector:
    return TextDetector(
        lang="vi",
        use_angle_cls=use_angle_cls,
        use_gpu=use_gpu,
        det_db_box_thresh=det_db_box_thresh,
        det_db_unclip_ratio=det_db_unclip_ratio,
    )


@st.cache_resource(show_spinner="Loading VietOCR recognizer...")
def get_recognizer(model: str, device: str) -> TextRecognizer:
    rec = TextRecognizer(model=model, device=device)
    # Trigger weight download up-front so the first inference is faster.
    rec._load()  # noqa: SLF001
    return rec


def _bytes_to_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


# --------------------------------------------------------------------- sidebar

st.sidebar.title("Settings")

rec_model = st.sidebar.selectbox(
    "VietOCR backbone",
    options=["vgg_transformer", "vgg_seq2seq"],
    index=0,
    help="transformer = more accurate; seq2seq = faster.",
)

device = st.sidebar.selectbox(
    "Device",
    options=["auto", "cpu", "cuda:0"],
    index=0,
)

use_angle_cls = st.sidebar.checkbox("Angle classifier (0/180)", value=True)

det_db_box_thresh = st.sidebar.slider(
    "Detection box threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
)
det_db_unclip_ratio = st.sidebar.slider(
    "Box unclip ratio (diacritics)",
    min_value=1.0,
    max_value=3.5,
    value=2.0,
    step=0.1,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Phase 1: PaddleOCR (detect) + VietOCR (recognize). "
    "Pretrained weights are aimed at printed text -- handwriting accuracy "
    "will improve after fine-tuning in Phase 1.5."
)


# ---------------------------------------------------------------------- main

st.title("Vietnamese Handwriting OCR")
st.write(
    "Upload an image of Vietnamese handwriting. "
    "PaddleOCR detects the text regions, then VietOCR reads each one."
)

uploaded = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    accept_multiple_files=False,
)

if uploaded is None:
    st.info("Upload an image to start. Sample images live in `data/samples/`.")
    st.stop()

raw_bytes = uploaded.read()
try:
    bgr = _bytes_to_bgr(raw_bytes)
except ValueError:
    st.error("Could not decode the uploaded file as an image.")
    st.stop()

resolved_device = device
if device == "auto":
    try:
        import torch

        resolved_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        resolved_device = "cpu"
use_gpu = resolved_device.startswith("cuda")

detector = get_detector(
    det_db_box_thresh=det_db_box_thresh,
    det_db_unclip_ratio=det_db_unclip_ratio,
    use_angle_cls=use_angle_cls,
    use_gpu=use_gpu,
)
recognizer = get_recognizer(model=rec_model, device=resolved_device)

pipe = HandwritingOCRPipeline(
    detector=detector,
    recognizer=recognizer,
    config=PipelineConfig(
        device=resolved_device,
        detector={"det_db_box_thresh": det_db_box_thresh,
                   "det_db_unclip_ratio": det_db_unclip_ratio},
        recognizer={"model": rec_model, "batch_size": 16},
        pipeline={"expand_ratio": 0.05, "reading_order_y_tolerance": 10},
    ),
)

with st.spinner("Running OCR..."):
    result = pipe.run(bgr)

viz_img = draw_ocr_result(bgr, result["items"])

col_left, col_right = st.columns(2)
with col_left:
    st.subheader("Input")
    st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
with col_right:
    st.subheader(f"Result ({result['num_boxes']} regions)")
    st.image(viz_img, use_container_width=True)

st.subheader("Recognized text")
if result["items"]:
    df = pd.DataFrame(
        [
            {
                "idx": i,
                "text": it["text"],
                "confidence": round(it["confidence"], 3),
                "bbox": it["axis_aligned_bbox"],
            }
            for i, it in enumerate(result["items"])
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.warning("No text regions were detected.")

st.subheader("Full text (reading order)")
st.text_area("full_text", value=result["full_text"], height=180)

json_bytes = json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
viz_bytes_io = io.BytesIO()
viz_img.save(viz_bytes_io, format="PNG")

col_a, col_b = st.columns(2)
with col_a:
    st.download_button(
        "Download JSON",
        data=json_bytes,
        file_name=f"{Path(uploaded.name).stem}_ocr.json",
        mime="application/json",
    )
with col_b:
    st.download_button(
        "Download visualization (PNG)",
        data=viz_bytes_io.getvalue(),
        file_name=f"{Path(uploaded.name).stem}_viz.png",
        mime="image/png",
    )
