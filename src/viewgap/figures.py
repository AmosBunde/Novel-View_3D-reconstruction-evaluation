"""Every figure in the README and the report, drawn from the files in ``results/``."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import data  # noqa: E402
from .cameras import view_directions  # noqa: E402
from .scenes import SCENES  # noqa: E402
from .splits import DESIGNS, split  # noqa: E402
from .suite import N_TEST, N_TRAIN_FULL, RESULTS, load_views  # noqa: E402

OUT = Path("docs/img")
BLUE, RED, INK, MUTED, GRID = "#2F5D9B", "#BA3D1A", "#1F2E43", "#5A6678", "#E3E7ED"
COLOUR = {"random": BLUE, "cap": RED}
LABEL = {"random": "Random views held out", "cap": "A whole patch held out"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.dpi": 160, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "legend.frameon": False,
})


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)


def fig_scenes() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.6))
    for k, sc in enumerate(SCENES):
        d = data.load(sc, data.FIELD_SIZE)
        v = 70
        axes[0, k].imshow(d["rgb"][v], interpolation="nearest")
        axes[0, k].set_title(sc, fontsize=11, loc="left")
        depth = np.where(np.isfinite(d["depth"][v]), d["depth"][v], np.nan)
        axes[1, k].imshow(depth, cmap="viridis_r", interpolation="nearest")
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0, 0].set_ylabel("rendered image")
    axes[1, 0].set_ylabel("exact depth")
    _save(fig, "scenes.png")


def fig_splits() -> None:
    c2w = data.load(SCENES[0], data.FIELD_SIZE)["c2w"]
    dirs = view_directions(c2w)
    az = np.arctan2(dirs[:, 1], dirs[:, 0])
    zenith = 90 - np.degrees(np.arcsin(dirs[:, 2]))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.2), subplot_kw={"projection": "polar"})
    for ax, design in zip(axes, DESIGNS):
        train, test = split(c2w, design, N_TEST, seed=0)
        ax.scatter(az[train], zenith[train], s=16, color="#B8C0CC", label="training camera")
        ax.scatter(az[test], zenith[test], s=30, color=COLOUR[design], label="test camera", zorder=3)
        ax.set_ylim(0, 80), ax.set_yticks([20, 45, 75]), ax.set_yticklabels([]), ax.set_xticklabels([])
        ax.set_title(LABEL[design], fontsize=11, pad=10)
    axes[1].legend(loc="lower center", bbox_to_anchor=(-0.12, -0.2), ncol=2)
    fig.suptitle("The 120 cameras seen from above the scene", fontsize=10, color=MUTED, y=0.02)
    _save(fig, "splits.png")


def fig_headline(s: dict) -> None:
    views = load_views()
    t = views[(views.role == "test") & (views.n_train == N_TRAIN_FULL) & (views.poses == "true")]
    runs = t.groupby(["scene", "design", "seed"], as_index=False)[["psnr", "depth_mae"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
    for ax, metric, label, scale in ((axes[0], "psnr", "PSNR on test views (dB, higher is better)", 1.0),
                                     (axes[1], "depth_mae", "Depth error on test views (% of camera distance, lower is better)", 100 / 3.0)):
        for k, sc in enumerate(SCENES):
            for seed in sorted(runs.seed.unique()):
                pair = [runs[(runs.scene == sc) & (runs.design == de) & (runs.seed == seed)][metric].mean() * scale
                        for de in DESIGNS]
                ax.plot(pair, [k + 0.12, k - 0.12], color="#C5CBD6", lw=1, zorder=1)
            for de, off in zip(DESIGNS, (0.12, -0.12)):
                vals = runs[(runs.scene == sc) & (runs.design == de)][metric].values * scale
                ax.scatter(vals, np.full(len(vals), k + off), s=34, color=COLOUR[de], zorder=3,
                           label=LABEL[de] if k == 0 else None, edgecolor="white", linewidth=0.8)
        ax.set_yticks(range(len(SCENES)), SCENES)
        ax.set_ylim(-0.6, len(SCENES) - 0.4)
        ax.set_xlabel(label, fontsize=9)
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.08, 1.02), ncol=2)
    _save(fig, "headline.png")


def fig_angle(s: dict) -> None:
    views = load_views()
    t = views[(views.role == "test") & (views.n_train == N_TRAIN_FULL) & (views.poses == "true")]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharex=True)
    for ax, metric, label, scale in ((axes[0], "psnr", "PSNR (dB)", 1.0),
                                     (axes[1], "depth_mae", "Depth error (% of camera distance)", 100 / 3.0)):
        for de in DESIGNS:
            g = t[t.design == de]
            ax.scatter(g.angle_nearest, g[metric] * scale, s=9, color=COLOUR[de], alpha=0.35, linewidth=0,
                       label=LABEL[de])
        curve = s["angle_curve"]
        ax.plot([c["angle_mean"] for c in curve], [c[metric] * scale for c in curve], color=INK, lw=2,
                marker="o", ms=4, label="mean of each angle band")
        ax.set_xlabel("Angle from the test camera to the nearest training camera (degrees)", fontsize=9)
        ax.set_ylabel(label)
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.08, 1.02), ncol=3, fontsize=9)
    _save(fig, "angle.png")


def fig_sparse(s: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), sharex=True)
    n = [r["n_train"] for r in s["e2"]]
    for ax, metric, label, scale in ((axes[0], "psnr", "PSNR on test views (dB)", 1.0),
                                     (axes[1], "depth_mae", "Depth error (% of camera distance)", 100 / 3.0)):
        for de in DESIGNS:
            y = [r[de][metric] * scale for r in s["e2"]]
            ax.plot(n, y, color=COLOUR[de], lw=2, marker="o", ms=5, label=LABEL[de])
            ax.annotate(f"{y[-1]:.1f}", (n[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9)
        ax.set_xscale("log", base=2)
        ax.set_xticks(n, [str(v) for v in n])
        ax.set_xlabel("Training views")
        ax.set_ylabel(label)
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.08, 1.02), ncol=2)
    _save(fig, "sparse.png")


def fig_examples() -> None:
    fig, axes = plt.subplots(len(SCENES), 6, figsize=(10.4, 5.5))
    for r, sc in enumerate(SCENES):
        for c, de in enumerate(DESIGNS):
            z = np.load(RESULTS / "examples" / f"{sc}-{de}-s0-n{N_TRAIN_FULL}-true.npz")
            true, pred = z["median_true"].astype(float), z["median_pred"].astype(float)
            err = np.abs(true - pred).mean(axis=-1)
            for k, (im, kw) in enumerate(((true, {}), (pred, {}), (err, {"cmap": "magma", "vmin": 0, "vmax": 0.25}))):
                ax = axes[r, 3 * c + k]
                ax.imshow(np.clip(im, 0, 1) if im.ndim == 3 else im, interpolation="nearest", **kw)
                ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if r == 0:
                    ax.set_title(("truth", "rendered", "error")[k], fontsize=9)
        axes[r, 0].set_ylabel(sc)
    fig.text(0.31, 0.96, LABEL["random"], ha="center", color=BLUE, fontsize=11, weight="bold")
    fig.text(0.71, 0.96, LABEL["cap"], ha="center", color=RED, fontsize=11, weight="bold")
    _save(fig, "examples.png")


def fig_sfm() -> None:
    fig, axes = plt.subplots(1, len(SCENES), figsize=(9.6, 3.5))
    for ax, sc in zip(axes, SCENES):
        path = RESULTS / "sfm" / f"{sc}.npz"
        if not path.exists():
            continue
        z = np.load(path)
        true = data.load(sc, data.FIELD_SIZE)["c2w"][:, :3, 3]
        pts = z["points"]
        pts = pts[(np.abs(pts) < 1.3).all(axis=1)]
        ax.scatter(pts[:, 0], pts[:, 1], s=1.2, color="#8792A2", linewidth=0, label="recovered points")
        ax.scatter(true[:, 0], true[:, 1], s=30, facecolor="none", edgecolor=BLUE, linewidth=1, label="true camera")
        est = z["c2w"][z["ok"], :3, 3]
        ax.scatter(est[:, 0], est[:, 1], s=7, color=RED, label="recovered camera", zorder=3)
        ax.set_aspect("equal"), ax.set_title(sc, fontsize=11, loc="left")
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3, markerscale=1.6)
    _save(fig, "sfm.png")


def make_all() -> None:
    s = json.loads((RESULTS / "summary.json").read_text())
    fig_scenes()
    fig_splits()
    fig_headline(s)
    fig_angle(s)
    fig_sparse(s)
    fig_examples()
    fig_sfm()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))
