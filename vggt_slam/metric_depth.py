"""Metric scale for each submap from measured (RGB-D) depth.

Enabled by ``main.py --metric_depth_folder <dir>``; without it nothing here
runs. The folder holds one depth image per input image, with the image's
base name (``frame_000123.png`` -> ``<dir>/frame_000123.png``), as integer
depth units (``--metric_depth_scale`` per metre, 1000 = millimetres, 0 = no
depth). Images without a depth file are skipped.

A submap's scale is the median, over its confident pixels, of measured depth
over VGGT's predicted depth. That is the scale the *points* need to be in
metres, and it is taken in preference to the odometry's displacement-based
scale (``Odometry.submap_scale``): VGGT can badly underestimate how far the
camera moved while its depth stays right -- on a wheeled robot whose camera
sees part of the robot, by 10x for tens of seconds -- and a translation-based
scale then inflates every point of the submap by the same factor. Both scales
are printed, so a disagreement shows in the log.
"""

from __future__ import annotations

import os

import cv2
import numpy as np


class MetricDepth:
    def __init__(self, folder: str, depth_scale: float = 1000.0, min_depth: float = 0.3,
                 max_depth: float = 6.0, min_pixels: int = 2000) -> None:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"--metric_depth_folder {folder} is not a directory")
        self.folder = folder
        self.depth_scale = float(depth_scale)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.min_pixels = int(min_pixels)
        self.scales: list[tuple[int, float | None, int]] = []

    def load(self, image_name: str, shape: tuple[int, int]) -> np.ndarray | None:
        """Measured depth in metres on VGGT's ``(H, W)`` grid, 0 where unknown.

        Mirrors ``vggt.utils.load_fn.load_and_preprocess_images`` in "crop"
        mode: resize to width 518 with a multiple-of-14 height, then centre
        crop rows if taller than 518. Nearest-neighbour, so no depth is
        invented across edges.
        """

        path = os.path.join(self.folder, os.path.basename(image_name))
        if not os.path.isfile(path):
            return None
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            return None
        height, width = raw.shape[:2]
        new_width = 518
        new_height = round(height * (new_width / width) / 14) * 14
        depth = cv2.resize(raw.astype(np.float32), (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        if new_height > 518:
            start = (new_height - 518) // 2
            depth = depth[start:start + 518]
        if depth.shape != tuple(shape):
            return None
        depth = depth / self.depth_scale
        depth[(depth < self.min_depth) | (depth > self.max_depth)] = 0.0
        return depth

    def submap_scale(self, submap_id: int, img_names, points_cam: np.ndarray, conf: np.ndarray,
                     conf_threshold: float) -> float | None:
        """Metres per VGGT unit for one submap's points, or ``None`` with too little depth."""

        ratios = []
        for index, name in enumerate(img_names):
            predicted = points_cam[index, ..., 2]
            measured = self.load(name, predicted.shape)
            if measured is None:
                continue
            valid = (measured > 0) & (predicted > 1e-6) & (conf[index] > conf_threshold)
            ratios.append(measured[valid] / predicted[valid])
        pixels = int(sum(len(r) for r in ratios))
        scale = float(np.median(np.concatenate(ratios))) if pixels >= self.min_pixels else None
        self.scales.append((submap_id, scale, pixels))
        print(f"Depth scale for submap {submap_id}: {scale} from {pixels} pixels")
        return scale
