"""Generate a synthetic sample image so the pipeline can be smoke-tested
without bringing your own handwriting photo.

Run from the repository root::

    python scripts/make_sample_image.py
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
]


def _font(size: int) -> ImageFont.ImageFont:
    for p in _FONT_CANDIDATES:
        if os.path.isfile(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def main(out: Path = Path("data/samples/sample.png")) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)

    width, height = 900, 320
    img = Image.new("RGB", (width, height), color=(252, 252, 248))
    draw = ImageDraw.Draw(img)
    font = _font(40)

    lines = [
        "Xin ch\u00e0o, \u0111\u00e2y l\u00e0 demo OCR.",
        "Nh\u1eadn di\u1ec7n ch\u1eef Ti\u1ebfng Vi\u1ec7t c\u00f3 d\u1ea5u.",
        "PaddleOCR + VietOCR pipeline.",
    ]
    y = 30
    for line in lines:
        draw.text((40, y), line, fill=(20, 20, 20), font=font)
        y += 80

    img.save(out)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
