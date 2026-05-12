"""Visualization helpers: draw polygons + Vietnamese text onto an image."""

from __future__ import annotations

import os
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Candidate Unicode fonts (ordered by likelihood of being installed).
_FONT_CANDIDATES = [
    # Windows
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    # Linux (Colab uses these)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    # macOS
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _load_font(size: int) -> ImageFont.ImageFont:
    """Return a Unicode-capable PIL font, falling back to default if none found."""
    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _to_pil(image) -> Image.Image:
    """Accept a numpy BGR/RGB array or PIL.Image and return a PIL RGB image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.ndim == 2:
        return Image.fromarray(arr).convert("RGB")
    if arr.shape[2] == 4:
        return Image.fromarray(arr).convert("RGB")
    # Assume OpenCV BGR.
    return Image.fromarray(arr[:, :, ::-1]).convert("RGB")


def draw_ocr_result(
    image,
    items: Sequence[Mapping],
    *,
    box_color: tuple = (255, 64, 64),
    text_color: tuple = (255, 255, 255),
    text_bg: tuple = (0, 0, 0),
    font_size: int = 16,
    line_width: int = 2,
    show_text: bool = True,
) -> Image.Image:
    """Draw polygons and (optionally) recognized text on ``image``.

    Parameters
    ----------
    image : np.ndarray | PIL.Image.Image
        Source image (OpenCV BGR or PIL RGB).
    items : sequence of mapping
        Each item must have keys ``bbox`` (4x2 polygon or 4-tuple xyxy) and
        ``text``. Optional key: ``confidence``.
    """
    pil_img = _to_pil(image).copy()
    draw = ImageDraw.Draw(pil_img)
    font = _load_font(font_size)

    for item in items:
        bbox = np.asarray(item["bbox"], dtype=np.float32)
        if bbox.shape == (4,):
            x_min, y_min, x_max, y_max = bbox.tolist()
            poly = [
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max),
            ]
        else:
            poly = [tuple(map(float, p)) for p in bbox.reshape(-1, 2)]

        draw.line(poly + [poly[0]], fill=box_color, width=line_width)

        if not show_text:
            continue

        text = item.get("text", "")
        conf = item.get("confidence")
        if conf is not None:
            label = f"{text}  ({conf:.2f})"
        else:
            label = text
        if not label:
            continue

        anchor = min(poly, key=lambda p: (p[1], p[0]))
        tx, ty = anchor[0], max(anchor[1] - font_size - 4, 0)

        try:
            left, top, right, bottom = draw.textbbox((tx, ty), label, font=font)
        except AttributeError:
            text_w, text_h = font.getsize(label) if hasattr(font, "getsize") else (len(label) * font_size // 2, font_size)
            left, top, right, bottom = tx, ty, tx + text_w, ty + text_h

        draw.rectangle([left - 2, top - 1, right + 2, bottom + 1], fill=text_bg)
        draw.text((tx, ty), label, fill=text_color, font=font)

    return pil_img
