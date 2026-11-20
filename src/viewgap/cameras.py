"""Pinhole cameras on a hemisphere band, and the rays they cast.

Conventions (the same everywhere in this package):

* world frame: right-handed, z up;
* camera frame: OpenCV style, x right, y down, z forward;
* a pose is a 4x4 camera-to-world matrix ``c2w``;
* a pixel (u, v) with u to the right and v down maps to the camera-frame
  direction ((u - cx) / f, (v - cy) / f, 1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RADIUS = 3.0            # distance from every camera to the look-at point
TARGET = np.array([0.0, 0.0, 0.25])
FOV_DEG = 42.0          # full field of view, square images
ELEV_RANGE_DEG = (15.0, 70.0)


@dataclass(frozen=True)
class Intrinsics:
    size: int       # images are size x size
    focal: float    # in pixels

    @property
    def K(self) -> np.ndarray:
        c = self.size / 2.0
        return np.array([[self.focal, 0, c], [0, self.focal, c], [0, 0, 1.0]])


def intrinsics(size: int) -> Intrinsics:
    return Intrinsics(size, 0.5 * size / np.tan(np.radians(FOV_DEG) / 2))


def look_at(position: np.ndarray, target: np.ndarray = TARGET) -> np.ndarray:
    """Camera-to-world matrix for a camera at ``position`` looking at ``target``, z up."""
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, forward, position
    return c2w


def hemisphere_cameras(n: int) -> np.ndarray:
    """``n`` camera-to-world poses, quasi-uniform over a band of the upper hemisphere.

    A Fibonacci lattice restricted to the elevation band, so the cameras have no
    ordering that a split could exploit by accident.
    """
    lo, hi = (np.sin(np.radians(e)) for e in ELEV_RANGE_DEG)
    golden = (1 + 5 ** 0.5) / 2
    i = np.arange(n)
    z = lo + (hi - lo) * (i + 0.5) / n                # uniform in sin(elevation) = uniform in area
    az = 2 * np.pi * i / golden
    r = np.sqrt(1 - z ** 2)
    dirs = np.stack([r * np.cos(az), r * np.sin(az), z], axis=1)
    return np.stack([look_at(TARGET + RADIUS * d) for d in dirs])


def view_directions(c2w: np.ndarray) -> np.ndarray:
    """Unit vectors from the look-at point to each camera centre."""
    d = c2w[:, :3, 3] - TARGET
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def angle_to_nearest(c2w_query: np.ndarray, c2w_ref: np.ndarray) -> np.ndarray:
    """For each query camera, the angle in degrees to the nearest reference camera,
    measured between their viewing directions as seen from the look-at point."""
    cos = view_directions(c2w_query) @ view_directions(c2w_ref).T
    return np.degrees(np.arccos(np.clip(cos.max(axis=1), -1, 1)))


def rays(c2w: np.ndarray, intr: Intrinsics) -> tuple[np.ndarray, np.ndarray]:
    """Origins and unit directions for every pixel centre: two (size, size, 3) arrays."""
    s = intr.size
    u, v = np.meshgrid(np.arange(s) + 0.5, np.arange(s) + 0.5)
    d_cam = np.stack([(u - s / 2) / intr.focal, (v - s / 2) / intr.focal, np.ones_like(u)], axis=-1)
    d = d_cam @ c2w[:3, :3].T
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    o = np.broadcast_to(c2w[:3, 3], d.shape).copy()
    return o, d
