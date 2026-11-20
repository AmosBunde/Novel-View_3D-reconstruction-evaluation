"""Rendered image sets, cached on disk so each scene is rendered once per resolution."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .cameras import hemisphere_cameras, intrinsics, rays
from .scenes import make_scene, render_set

N_CAMERAS = 120
FIELD_SIZE = 128        # resolution the radiance field is trained and scored at
SFM_SIZE = 384          # resolution structure from motion sees
CACHE = Path("data")


def load(scene: str, size: int) -> dict[str, np.ndarray]:
    """Images, exact depth maps and true poses for one scene at one resolution."""
    path = CACHE / f"{scene}_{size}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    c2w = hemisphere_cameras(N_CAMERAS)
    rgb, depth = render_set(make_scene(scene), c2w, intrinsics(size), supersample=3 if size <= 160 else 2)
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(path, rgb=rgb, depth=depth, c2w=c2w)
    return {"rgb": rgb, "depth": depth, "c2w": c2w}


def ray_bundle(c2w: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Ray origins and directions for a set of cameras: two (n, size, size, 3) arrays."""
    intr = intrinsics(size)
    pairs = [rays(c, intr) for c in c2w]
    return np.stack([o for o, _ in pairs]), np.stack([d for _, d in pairs])
