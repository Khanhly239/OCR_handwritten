"""Text detection wrapper around PaddleOCR (DBNet, detection only).

PaddleOCR has both a detector and a recognizer. We only want the detector
because Vietnamese handwriting recognition is handled by VietOCR. Setting
``rec=False`` skips the recognition step.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


class TextDetector:
    """Detect text regions in an image and return their polygon corners.

    Parameters
    ----------
    lang : str
        Language hint for PaddleOCR's detector. ``'vi'`` falls back to a
        Latin-script detector that works well for Vietnamese.
    use_angle_cls : bool
        Use PaddleOCR's angle classifier to handle 180-degree rotated text.
    use_gpu : bool
        Run on GPU if PaddlePaddle was built with CUDA support.
    det_db_thresh, det_db_box_thresh, det_db_unclip_ratio :
        DBNet hyperparameters. Higher ``unclip_ratio`` produces larger boxes
        (useful for Vietnamese diacritics).
    det_limit_side_len : int
        Resize images so the longest/shortest side (per ``det_limit_type``)
        does not exceed this value. Speeds up detection on large scans.
    det_limit_type : str
        ``"max"`` or ``"min"``.
    show_log : bool
        Forward to PaddleOCR. Default ``False`` keeps the console clean.
    """

    def __init__(
        self,
        lang: str = "vi",
        *,
        use_angle_cls: bool = True,
        use_gpu: bool = False,
        det_db_thresh: float = 0.3,
        det_db_box_thresh: float = 0.5,
        det_db_unclip_ratio: float = 2.0,
        det_limit_side_len: int = 960,
        det_limit_type: str = "max",
        show_log: bool = False,
    ) -> None:
        # Import lazily so that the module can be imported without PaddleOCR
        # installed (useful for unit tests that mock the detector).
        from paddleocr import PaddleOCR

        self._kwargs = dict(
            lang=lang,
            use_angle_cls=use_angle_cls,
            use_gpu=use_gpu,
            det_db_thresh=det_db_thresh,
            det_db_box_thresh=det_db_box_thresh,
            det_db_unclip_ratio=det_db_unclip_ratio,
            det_limit_side_len=det_limit_side_len,
            det_limit_type=det_limit_type,
            show_log=show_log,
        )
        self._ocr = PaddleOCR(**self._kwargs)

    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        """Detect text regions in ``image`` (BGR ``np.ndarray``).

        Returns
        -------
        list of np.ndarray
            Each element is a ``(4, 2)`` float32 polygon (TL, TR, BR, BL order
            is *not* guaranteed; downstream code should reorder if needed).
        """
        if image is None:
            raise ValueError("image is None")
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        # PaddleOCR's high-level API: det=True, rec=False -> only polygons.
        result = self._ocr.ocr(image, det=True, rec=False, cls=False)

        # API quirks: result can be ``[None]`` for empty pages or
        # ``[[poly1, poly2, ...]]`` for a single page.
        if not result or result[0] is None:
            return []
        polys_raw = result[0]
        polys: List[np.ndarray] = []
        for p in polys_raw:
            arr = np.asarray(p, dtype=np.float32).reshape(-1, 2)
            if arr.shape[0] >= 4:
                polys.append(arr[:4])
        return polys

    @property
    def config(self) -> dict:
        """Return the constructor kwargs (useful for logging/serialization)."""
        return dict(self._kwargs)
