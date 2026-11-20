"""Procedural scenes with exact ground truth.

Each scene is a square table with spheres and boxes on it, textured with a
non-repeating 3D pattern. Rays are intersected analytically, so every rendered
pixel comes with its exact depth. Nothing is downloaded and nothing is licensed
by a third party: the dataset is this file plus a seed.

Shading is Lambertian with hard shadows from one directional light, plus a
Blinn-Phong highlight on the spheres so that appearance depends on the viewing
direction, as it does in real scenes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cameras import Intrinsics, rays

SCENES = ("blocks", "orbs", "clutter")
BACKGROUND = 1.0                         # white
LIGHT = np.array([0.45, -0.35, 0.82])
LIGHT = LIGHT / np.linalg.norm(LIGHT)
EPS = 1e-4


@dataclass(frozen=True)
class Scene:
    name: str
    box_min: np.ndarray       # (B, 3)
    box_max: np.ndarray       # (B, 3)
    sph_c: np.ndarray         # (S, 3)
    sph_r: np.ndarray         # (S,)
    colour: np.ndarray        # (B + S, 3) base colours, boxes first
    tex_table: np.ndarray     # random values that the texture lattice is hashed into


def make_scene(name: str) -> Scene:
    if name not in SCENES:
        raise ValueError(f"unknown scene {name!r}; choose from {SCENES}")
    rng = np.random.default_rng(1000 + SCENES.index(name))
    n_box, n_sph = {"blocks": (6, 1), "orbs": (1, 6), "clutter": (5, 5)}[name]

    boxes_min = [np.array([-0.9, -0.9, -0.08])]      # the table
    boxes_max = [np.array([0.9, 0.9, 0.0])]
    spheres_c, spheres_r = [], []
    placed: list[tuple[float, float, float]] = []    # x, y, footprint radius

    def place(radius: float) -> tuple[float, float]:
        for _ in range(2000):
            x, y = rng.uniform(-0.72, 0.72, 2)
            if all(np.hypot(x - px, y - py) > radius + pr + 0.03 for px, py, pr in placed):
                placed.append((x, y, radius))
                return x, y
        raise RuntimeError("could not place object")

    for _ in range(n_box):
        hx, hy = rng.uniform(0.09, 0.2, 2)
        h = rng.uniform(0.15, 0.6)
        x, y = place(float(np.hypot(hx, hy)))
        boxes_min.append(np.array([x - hx, y - hy, 0.0]))
        boxes_max.append(np.array([x + hx, y + hy, h]))
    for _ in range(n_sph):
        r = rng.uniform(0.11, 0.24)
        x, y = place(r)
        spheres_c.append(np.array([x, y, r]))
        spheres_r.append(r)

    n = len(boxes_min) + len(spheres_c)
    hue = (rng.uniform() + np.arange(n) * 0.381966) % 1.0
    colour = np.stack([_hsv(h, 0.55, 0.9) for h in hue])
    colour[0] = np.array([0.78, 0.76, 0.72])         # the table is a warm grey

    return Scene(
        name=name,
        box_min=np.stack(boxes_min), box_max=np.stack(boxes_max),
        sph_c=np.stack(spheres_c) if spheres_c else np.zeros((0, 3)),
        sph_r=np.array(spheres_r),
        colour=colour, tex_table=rng.uniform(0, 1, 4096),
    )


def _hsv(h: float, s: float, v: float) -> np.ndarray:
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return np.array([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i])


OCTAVES = ((0.30, 0.40), (0.11, 0.35), (0.04, 0.25))    # (cell size in scene units, weight)


def _value_noise(table: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Random values on an integer lattice, blended smoothly between lattice points."""
    base = np.floor(p).astype(np.int64)
    f = p - base
    f = f * f * (3 - 2 * f)
    out = np.zeros(p.shape[0])
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                c = base + np.array([dx, dy, dz])
                h = (c[:, 0] * 73856093) ^ (c[:, 1] * 19349663) ^ (c[:, 2] * 83492791)
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                out += w * table[h % table.size]
    return out


def _texture(scene: Scene, p: np.ndarray) -> np.ndarray:
    """A non-repeating pattern in [0, 1] at points p (N, 3), with detail at three scales.

    The finest scale is what gives a feature detector something to hold on to.
    """
    v = sum(w * _value_noise(scene.tex_table, p / cell + 17.0 * k) for k, (cell, w) in enumerate(OCTAVES))
    return np.clip(0.5 + 1.6 * (v - 0.5), 0, 1)


def intersect(scene: Scene, o: np.ndarray, d: np.ndarray):
    """Nearest hit along each ray. Returns (t, primitive index, normal); t is inf on a miss."""
    n_rays = o.shape[0]
    t_best = np.full(n_rays, np.inf)
    idx = np.full(n_rays, -1)
    normal = np.zeros((n_rays, 3))

    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
    for b in range(scene.box_min.shape[0]):
        t0 = (scene.box_min[b] - o) * inv
        t1 = (scene.box_max[b] - o) * inv
        tn, tf = np.minimum(t0, t1), np.maximum(t0, t1)
        axis = np.nanargmax(tn, axis=1)
        t_near, t_far = np.nanmax(tn, axis=1), np.nanmin(tf, axis=1)
        hit = (t_near < t_far) & (t_near > EPS) & (t_near < t_best)
        t_best[hit], idx[hit] = t_near[hit], b
        nrm = np.zeros((int(hit.sum()), 3))
        nrm[np.arange(nrm.shape[0]), axis[hit]] = -np.sign(d[hit, axis[hit]])
        normal[hit] = nrm
    n_box = scene.box_min.shape[0]
    for s in range(scene.sph_c.shape[0]):
        oc = o - scene.sph_c[s]
        b_half = (oc * d).sum(axis=1)
        disc = b_half ** 2 - ((oc * oc).sum(axis=1) - scene.sph_r[s] ** 2)
        t = -b_half - np.sqrt(np.where(disc > 0, disc, np.nan))
        hit = (disc > 0) & (t > EPS) & (t < t_best)
        t_best[hit], idx[hit] = t[hit], n_box + s
        normal[hit] = (oc[hit] + t[hit, None] * d[hit]) / scene.sph_r[s]
    return t_best, idx, normal


def shade(scene: Scene, o: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Colour (N, 3) and depth along the ray (N,), inf where the ray leaves the scene."""
    t, idx, n = intersect(scene, o, d)
    rgb = np.full((o.shape[0], 3), BACKGROUND)
    hit = np.isfinite(t)
    if not hit.any():
        return rgb, t
    p = o[hit] + t[hit, None] * d[hit]
    nh, dh, ih = n[hit], d[hit], idx[hit]
    albedo = scene.colour[ih] * (0.25 + 0.75 * _texture(scene, p))[:, None]
    t_shadow, _, _ = intersect(scene, p + nh * 1e-3, np.broadcast_to(LIGHT, p.shape).copy())
    lit = (~np.isfinite(t_shadow)).astype(float)
    diffuse = np.clip(nh @ LIGHT, 0, None) * lit
    half = LIGHT - dh
    half = half / np.linalg.norm(half, axis=1, keepdims=True)
    is_sphere = (ih >= scene.box_min.shape[0]).astype(float)
    specular = 0.35 * is_sphere * lit * np.clip((nh * half).sum(axis=1), 0, None) ** 40
    rgb[hit] = np.clip(albedo * (0.32 + 0.68 * diffuse)[:, None] + specular[:, None], 0, 1)
    return rgb, t


def render(scene: Scene, c2w: np.ndarray, intr: Intrinsics, supersample: int = 3):
    """Antialiased image and depth map for one camera.

    Colour is averaged over ``supersample``^2 sub-pixel rays. Depth is the exact
    value at the pixel centre, with inf for background, because an average of
    depths across an object edge is not the depth of anything.
    """
    from .cameras import intrinsics  # local import keeps the module graph simple

    s = intr.size
    hi = intrinsics(s * supersample)
    o, d = rays(c2w, hi)
    rgb, _ = shade(scene, o.reshape(-1, 3), d.reshape(-1, 3))
    rgb = rgb.reshape(s, supersample, s, supersample, 3).mean(axis=(1, 3))
    o, d = rays(c2w, intr)
    t, _, _ = intersect(scene, o.reshape(-1, 3), d.reshape(-1, 3))
    return rgb.astype(np.float32), t.reshape(s, s).astype(np.float32)


def render_set(scene: Scene, c2ws: np.ndarray, intr: Intrinsics, supersample: int = 3):
    out = [render(scene, c, intr, supersample) for c in c2ws]
    return np.stack([r for r, _ in out]), np.stack([t for _, t in out])
