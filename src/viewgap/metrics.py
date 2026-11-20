"""Image and geometry metrics. Images are float arrays in [0, 1], shape (H, W, 3)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    return float(-10 * np.log10(max(mse, 1e-12)))


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Structural similarity (Wang et al., 2004): Gaussian window, sigma 1.5, mean over channels."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    out = []
    for ch in range(pred.shape[-1]):
        x, y = pred[..., ch].astype(np.float64), target[..., ch].astype(np.float64)
        blur = lambda a: gaussian_filter(a, 1.5, truncate=3.5)  # noqa: E731
        mx, my = blur(x), blur(y)
        vx, vy, cxy = blur(x * x) - mx ** 2, blur(y * y) - my ** 2, blur(x * y) - mx * my
        s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
        out.append(s.mean())
    return float(np.mean(out))


def depth_mae(pred_depth: np.ndarray, true_depth: np.ndarray) -> float:
    """Mean absolute depth error over pixels where the true scene has a surface.

    In scene units; the table is 1.8 units wide and the cameras are 3 units away.
    """
    fg = np.isfinite(true_depth)
    return float(np.abs(pred_depth[fg] - true_depth[fg]).mean())


def silhouette_iou(opacity: np.ndarray, true_depth: np.ndarray) -> float:
    pred, true = opacity > 0.5, np.isfinite(true_depth)
    return float((pred & true).sum() / max((pred | true).sum(), 1))
