"""A compact voxel-grid radiance field, trained by volume rendering.

This is deliberately a small member of a well-known family (dense feature grids
with a shallow decoder, as in DVGO and Plenoxels), not a new method and not a
state-of-the-art one. It is sized to train in a few minutes on a CPU so that the
study can afford dozens of runs. The subject of the study is the test design,
and the same model is used on every side of every comparison.

Model: two feature grids at different resolutions are sampled trilinearly and
concatenated. A small network turns the features into density and a geometry
code; a second one turns the geometry code and the encoded viewing direction
into colour. Rendering is standard emission-absorption quadrature over
stratified samples inside the scene's bounding box, composited on white.

Speed comes from an occupancy grid, as in Instant-NGP and DVGO: every few
hundred steps the density is read out on a coarse lattice, and samples that fall
in cells the model itself considers empty are skipped. The grid is derived from
the model, never from the ground-truth scene.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BOUND_MIN = torch.tensor([-1.0, -1.0, -0.15])
BOUND_MAX = torch.tensor([1.0, 1.0, 0.85])


@dataclass(frozen=True)
class FieldConfig:
    coarse_res: int = 40
    fine_res: int = 96
    coarse_ch: int = 8
    fine_ch: int = 4
    hidden: int = 48
    n_samples: int = 64
    steps: int = 1000
    batch_rays: int = 2048
    occ_res: int = 64
    occ_every: int = 100
    occ_threshold: float = 1.0
    tv_every: int = 10
    lr_grid: float = 0.03
    lr_net: float = 0.004
    tv_weight: float = 1e-4


class Field(nn.Module):
    def __init__(self, cfg: FieldConfig):
        super().__init__()
        self.cfg = cfg
        self.coarse = nn.Parameter(0.05 * torch.randn(1, cfg.coarse_ch, *(cfg.coarse_res,) * 3))
        self.fine = nn.Parameter(0.05 * torch.randn(1, cfg.fine_ch, *(cfg.fine_res,) * 3))
        ch = cfg.coarse_ch + cfg.fine_ch
        self.geo = nn.Sequential(nn.Linear(ch, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, 1 + 15))
        self.col = nn.Sequential(nn.Linear(15 + 9, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, 3))
        self.register_buffer("occ", torch.ones((cfg.occ_res,) * 3, dtype=torch.bool))

    def occupied(self, x: torch.Tensor) -> torch.Tensor:
        """Whether each point (N, 3) lies in a cell the model currently treats as non-empty."""
        r = self.cfg.occ_res
        i = ((x - BOUND_MIN) / (BOUND_MAX - BOUND_MIN) * r).long().clamp(0, r - 1)
        return self.occ[i[:, 2], i[:, 1], i[:, 0]]

    @torch.no_grad()
    def update_occupancy(self) -> float:
        """Recompute the occupancy grid from the current density; returns the occupied fraction."""
        r = self.cfg.occ_res
        c = (torch.arange(r) + 0.5) / r
        z, y, x = torch.meshgrid(c, c, c, indexing="ij")
        pts = torch.stack([x, y, z], dim=-1).view(-1, 3) * (BOUND_MAX - BOUND_MIN) + BOUND_MIN
        sigma = F.softplus(self.geo(self.features(pts))[:, 0] - 1.0) * 25.0
        dense = (sigma > self.cfg.occ_threshold).view(1, 1, r, r, r).float()
        self.occ.copy_(F.max_pool3d(dense, 3, stride=1, padding=1).view(r, r, r) > 0)
        return float(self.occ.float().mean())

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Trilinear features at world points x (N, 3)."""
        g = (2 * (x - BOUND_MIN) / (BOUND_MAX - BOUND_MIN) - 1).view(1, -1, 1, 1, 3)
        # grid_sample reads the last dimension as (x, y, z) -> (W, H, D); the grids are
        # stored (D, H, W) = (z, y, x), which is what that ordering expects.
        a = F.grid_sample(self.coarse, g, mode="bilinear", align_corners=True)
        b = F.grid_sample(self.fine, g, mode="bilinear", align_corners=True)
        return torch.cat([a, b], dim=1).view(a.shape[1] + b.shape[1], -1).t()

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.geo(self.features(x))
        sigma = F.softplus(h[:, 0] - 1.0) * 25.0
        rgb = torch.sigmoid(self.col(torch.cat([h[:, 1:], _sh2(d)], dim=1)))
        return sigma, rgb

    def tv(self) -> torch.Tensor:
        g = self.fine
        return ((g[:, :, 1:] - g[:, :, :-1]) ** 2).mean() + ((g[:, :, :, 1:] - g[:, :, :, :-1]) ** 2).mean() \
            + ((g[..., 1:] - g[..., :-1]) ** 2).mean()


def _sh2(d: torch.Tensor) -> torch.Tensor:
    """Real spherical harmonics up to degree 2 (9 values), unnormalised."""
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    return torch.stack([torch.ones_like(x), x, y, z, x * y, y * z, 3 * z * z - 1, x * z, x * x - y * y], dim=1)


def _box_span(o: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
    t0, t1 = (BOUND_MIN - o) * inv, (BOUND_MAX - o) * inv
    near = torch.minimum(t0, t1).amax(dim=1).clamp(min=0.05)
    far = torch.maximum(t0, t1).amin(dim=1)
    return near, torch.maximum(far, near + 1e-3)


def render_rays(model: Field, o: torch.Tensor, d: torch.Tensor, jitter: bool):
    """Colour, expected depth and accumulated opacity for a batch of rays."""
    n = model.cfg.n_samples
    near, far = _box_span(o, d)
    u = (torch.arange(n) + (torch.rand(o.shape[0], n) if jitter else 0.5)) / n
    t = near[:, None] + (far - near)[:, None] * u
    x = o[:, None, :] + t[..., None] * d[:, None, :]
    x_flat, d_flat = x.reshape(-1, 3), d[:, None, :].expand_as(x).reshape(-1, 3)
    keep = model.occupied(x_flat)
    sigma = torch.zeros(x_flat.shape[0])
    rgb = torch.zeros(x_flat.shape[0], 3)
    if keep.any():
        s_keep, c_keep = model(x_flat[keep], d_flat[keep])
        sigma = sigma.index_put((keep,), s_keep)
        rgb = rgb.index_put((keep,), c_keep)
    sigma, rgb = sigma.view(-1, n), rgb.view(-1, n, 3)
    delta = ((far - near) / n)[:, None]
    alpha = 1 - torch.exp(-sigma * delta)
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-10], dim=1), dim=1)[:, :-1]
    w = alpha * trans
    acc = w.sum(dim=1)
    colour = (w[..., None] * rgb).sum(dim=1) + (1 - acc)[:, None]
    depth = (w * t).sum(dim=1) / acc.clamp(min=1e-6)
    return colour, depth, acc


def train_field(o: np.ndarray, d: np.ndarray, rgb: np.ndarray, cfg: FieldConfig, seed: int,
                log_every: int = 0) -> Field:
    """Fit a field to rays (N, 3), (N, 3) with colours (N, 3)."""
    torch.manual_seed(seed)
    model = Field(cfg)
    opt = torch.optim.Adam([
        {"params": [model.coarse, model.fine], "lr": cfg.lr_grid},
        {"params": list(model.geo.parameters()) + list(model.col.parameters()), "lr": cfg.lr_net},
    ], eps=1e-15)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.05 ** (1 / cfg.steps))
    o_t, d_t, c_t = (torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)) for a in (o, d, rgb))
    gen = torch.Generator().manual_seed(seed)
    for step in range(cfg.steps):
        pick = torch.randint(0, o_t.shape[0], (cfg.batch_rays,), generator=gen)
        colour, _, _ = render_rays(model, o_t[pick], d_t[pick], jitter=True)
        loss = F.mse_loss(colour, c_t[pick])
        total = loss
        if step % cfg.tv_every == 0:
            total = total + cfg.tv_every * cfg.tv_weight * model.tv()
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        sched.step()
        if (step + 1) % cfg.occ_every == 0:
            model.update_occupancy()
        if log_every and (step + 1) % log_every == 0:
            print(f"  step {step + 1:5d}  psnr {-10 * np.log10(loss.item()):.2f}", flush=True)
    return model


@torch.no_grad()
def render_image(model: Field, o: np.ndarray, d: np.ndarray, chunk: int = 16384):
    """Render full images from ray arrays of shape (..., 3). Returns colour, depth, opacity."""
    shape = o.shape[:-1]
    o_t = torch.from_numpy(np.ascontiguousarray(o.reshape(-1, 3), dtype=np.float32))
    d_t = torch.from_numpy(np.ascontiguousarray(d.reshape(-1, 3), dtype=np.float32))
    parts = [render_rays(model, o_t[i:i + chunk], d_t[i:i + chunk], jitter=False)
             for i in range(0, o_t.shape[0], chunk)]
    colour, depth, acc = (torch.cat([p[k] for p in parts]).numpy() for k in range(3))
    return colour.reshape(*shape, 3), depth.reshape(shape), acc.reshape(shape)
