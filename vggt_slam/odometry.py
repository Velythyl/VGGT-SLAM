"""Wheel (or any metric) odometry for VGGT-SLAM.

Enabled by ``main.py --odometry <file>``; without that flag nothing in this
module runs and VGGT-SLAM behaves exactly as upstream.

The file holds one camera pose per image, in a metric odometry frame:

    <frame number> tx ty tz qx qy qz qw

``frame number`` is the number in the image's file name (the same number
``Submap.set_frame_ids`` extracts, e.g. ``frame_000123.png`` -> 123), and the
pose is camera-to-odometry (``T_odom_camera``) for the OpenCV optical frame. A
caller that has base odometry and a base-to-camera mount composes them
(``T_odom_base @ T_base_camera``) before writing the file. Frames without a
row are allowed; they simply get no odometry constraint.

What the odometry is used for (see ``README.md``, "Odometry fusion"):

1. keyframing -- a frame becomes a keyframe when the camera has moved
   ``kf_dist`` metres or turned ``kf_angle_deg`` degrees since the last one,
   in addition to the optical-flow test, so that driving straight down a
   corridor (tiny flow near the image centre) no longer leaves tens of metres
   between keyframes;
2. metric scale per submap -- VGGT's submap is rescaled so its camera
   displacements match the odometry's, which replaces the pairwise scale
   estimate between two metric submaps and makes the whole map metric;
3. relative-pose factors between consecutive frames of a submap;
4. the gauge -- the first node is anchored at its odometry pose, so the map
   is published in the odometry frame;
5. loop-closure gating -- a retrieved loop whose two frames the odometry
   places more than ``lc_max_dist`` apart is rejected.
"""

from __future__ import annotations

import os
import re

import numpy as np
from gtsam import noiseModel
from scipy.spatial.transform import Rotation


def frame_number(path: str) -> int:
    """The number in an image's file name, as ``Submap.set_frame_ids`` reads it."""

    match = re.search(r"\d+(?:\.\d+)?", os.path.basename(path))
    if match is None:
        raise ValueError(f"No number found in image name: {path}")
    return int(round(float(match.group())))


class Odometry:
    def __init__(
        self,
        poses: dict[int, np.ndarray],
        *,
        kf_dist: float = 0.5,
        kf_angle_deg: float = 15.0,
        sigma: float = 0.05,
        lc_max_dist: float = 3.0,
        min_scale_motion: float = 0.05,
        inner: str = "fuse",
    ) -> None:
        if not poses:
            raise ValueError("The odometry file holds no poses.")
        self.poses = poses
        self.kf_dist = float(kf_dist)
        self.kf_angle = np.radians(float(kf_angle_deg))
        self.lc_max_dist = float(lc_max_dist)
        if inner not in ("fuse", "factor"):
            raise ValueError(f"inner must be 'fuse' or 'factor', got {inner!r}")
        self.inner = inner
        self.min_scale_motion = float(min_scale_motion)
        self.noise = noiseModel.Diagonal.Sigmas(float(sigma) * np.ones(15, dtype=float))
        self._last_keyframe: np.ndarray | None = None
        self.scales: list[tuple[int, float | None, int]] = []  # (submap id, scale, pairs), for the log

    @classmethod
    def load(cls, path: str, **kwargs) -> "Odometry":
        poses: dict[int, np.ndarray] = {}
        with open(path, encoding="utf-8") as handle:
            for raw in handle:
                values = raw.split()
                if len(values) != 8 or raw.lstrip().startswith("#"):
                    continue
                number = [float(value) for value in values]
                pose = np.eye(4)
                pose[:3, :3] = Rotation.from_quat(number[4:8]).as_matrix()
                pose[:3, 3] = number[1:4]
                poses[int(round(number[0]))] = pose
        print(f"Loaded {len(poses)} odometry camera poses from {path}")
        return cls(poses, **kwargs)

    def pose(self, frame: float) -> np.ndarray | None:
        return self.poses.get(int(round(frame)))

    def relative(self, first: float, second: float) -> np.ndarray | None:
        """``T_first_second`` from the odometry, or ``None`` if either frame has no pose."""

        a, b = self.pose(first), self.pose(second)
        if a is None or b is None:
            return None
        return np.linalg.inv(a) @ b

    def fuse_motion(self, first: float, second: float, vggt_motion: np.ndarray) -> np.ndarray | None:
        """VGGT's rotation with the odometry's translation for one frame pair, or ``None``.

        The two sources are complementary on a wheeled robot: wheel odometry
        measures distance well but under-reports spin turns (wheel slip),
        while VGGT gets rotation right but can badly underestimate
        translation (a camera that sees part of the robot sees something
        that never moves). Fusing them into *one* SE(3) measurement, instead
        of two disagreeing BetweenFactorSL4s, matters in an SL(4) graph: the
        optimizer reconciles disagreeing rigid measurements through the
        homographies' scale and projective components, which leave the
        decomposed camera poses looking right while scaling each frame's
        points.
        """

        odometry = self.relative(first, second)
        if odometry is None:
            return None
        fused = np.eye(4)
        fused[:3, :3] = vggt_motion[:3, :3]
        fused[:3, 3] = odometry[:3, 3]
        return fused

    # ---------------------------------------------------------- keyframing

    def moved_since_keyframe(self, frame: float) -> bool:
        pose = self.pose(frame)
        if pose is None or self._last_keyframe is None:
            return False
        motion = np.linalg.inv(self._last_keyframe) @ pose
        angle = np.linalg.norm(Rotation.from_matrix(motion[:3, :3]).as_rotvec())
        return bool(np.linalg.norm(motion[:3, 3]) >= self.kf_dist or angle >= self.kf_angle)

    def mark_keyframe(self, frame: float) -> None:
        pose = self.pose(frame)
        if pose is not None:
            self._last_keyframe = pose

    # --------------------------------------------------------------- scale

    def submap_scale(self, submap_id: int, frame_ids, world_to_cam: np.ndarray) -> float | None:
        """Metres per VGGT unit for one submap, or ``None`` if the odometry cannot tell.

        Every pair of the submap's frames that the odometry moved apart by at
        least ``min_scale_motion`` (and by at least a quarter of the largest
        such displacement, so short noisy baselines do not dominate) gives a
        ratio of odometry displacement to VGGT displacement; the median is the
        scale. Fewer than three pairs -- a parked robot -- gives ``None``, and
        the caller falls back to upstream's pairwise scale estimate.
        """

        centres_vggt = np.linalg.inv(world_to_cam)[:, :3, 3]
        poses = [self.pose(frame) for frame in frame_ids]
        pairs = []
        for i in range(len(frame_ids)):
            for j in range(i + 1, len(frame_ids)):
                if poses[i] is None or poses[j] is None:
                    continue
                odom = float(np.linalg.norm(poses[j][:3, 3] - poses[i][:3, 3]))
                vggt = float(np.linalg.norm(centres_vggt[j] - centres_vggt[i]))
                if odom >= self.min_scale_motion and vggt > 1e-9:
                    pairs.append((odom, odom / vggt))
        scale = None
        if pairs:
            longest = max(odom for odom, _ in pairs)
            ratios = [ratio for odom, ratio in pairs if odom >= 0.25 * longest]
            if len(ratios) >= 3:
                scale = float(np.median(ratios))
        self.scales.append((submap_id, scale, len(pairs)))
        print(f"Odometry scale for submap {submap_id}: {scale} from {len(pairs)} pairs")
        return scale

    # --------------------------------------------------------- loop closure

    def loop_plausible(self, first: float, second: float) -> bool:
        a, b = self.pose(first), self.pose(second)
        if a is None or b is None:
            return True
        return bool(np.linalg.norm(a[:3, 3] - b[:3, 3]) <= self.lc_max_dist)
