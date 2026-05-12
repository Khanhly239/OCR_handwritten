"""Image utilities: perspective warp, polygon expansion, reading-order sort."""

from __future__ import annotations

from typing import List, Sequence

import cv2
import numpy as np


Polygon = np.ndarray  # shape (4, 2), float32


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Return polygon points in TL, TR, BR, BL order."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError(f"Expected 4 points, got {pts.shape[0]}")

    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()

    ordered[0] = pts[np.argmin(s)]      # top-left has smallest x+y
    ordered[2] = pts[np.argmax(s)]      # bottom-right has largest x+y
    ordered[1] = pts[np.argmin(diff)]   # top-right has smallest y-x
    ordered[3] = pts[np.argmax(diff)]   # bottom-left has largest y-x
    return ordered


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Warp the quadrilateral region defined by ``pts`` into an axis-aligned crop.

    Parameters
    ----------
    image : np.ndarray
        Source image (BGR or grayscale).
    pts : np.ndarray
        Array of 4 polygon corners.

    Returns
    -------
    np.ndarray
        Cropped, perspective-corrected image. Always at least 1x1 in size.
    """
    rect = _order_points(pts)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(round(max(width_a, width_b))), 1)

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(round(max(height_a, height_b))), 1)

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def expand_polygon(poly: np.ndarray, ratio: float = 0.05) -> np.ndarray:
    """Expand a polygon outward from its centroid by ``ratio`` of its size.

    Useful to avoid clipping Vietnamese diacritics that PaddleOCR's detector
    sometimes excludes from the box.
    """
    if ratio <= 0:
        return np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    center = pts.mean(axis=0, keepdims=True)
    expanded = center + (pts - center) * (1.0 + ratio)
    return expanded


def sort_polygons_reading_order(
    polys: Sequence[np.ndarray], y_tolerance: int = 10
) -> List[int]:
    """Return indices that sort polygons in natural reading order.

    Polygons are grouped into rows based on the y-coordinate of their centroid
    (within ``y_tolerance`` pixels), then sorted left-to-right within each row.
    """
    if not polys:
        return []

    centers = np.array(
        [np.asarray(p, dtype=np.float32).reshape(-1, 2).mean(axis=0) for p in polys]
    )
    indices = list(range(len(polys)))
    indices.sort(key=lambda i: centers[i, 1])

    rows: List[List[int]] = []
    for idx in indices:
        cy = centers[idx, 1]
        placed = False
        for row in rows:
            row_cy = np.mean([centers[j, 1] for j in row])
            if abs(cy - row_cy) <= y_tolerance:
                row.append(idx)
                placed = True
                break
        if not placed:
            rows.append([idx])

    ordered: List[int] = []
    for row in rows:
        row.sort(key=lambda i: centers[i, 0])
        ordered.extend(row)
    return ordered


def polygon_to_bbox(poly: np.ndarray) -> List[int]:
    """Return ``[x_min, y_min, x_max, y_max]`` axis-aligned bbox for a polygon."""
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    return [int(round(x_min)), int(round(y_min)), int(round(x_max)), int(round(y_max))]
