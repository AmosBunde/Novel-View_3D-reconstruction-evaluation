"""The experiment matrix, the runner and the summary statistics.

Every run trains one radiance field on one scene and scores it on held-out
cameras. Runs differ only in which cameras are held out (``design``), how many
training cameras are kept (``n_train``), the seed, and whether the camera poses
are the true ones or the ones recovered by structure from motion (``poses``).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, metrics
from .cameras import angle_to_nearest
from .field import FieldConfig, render_image, train_field
from .scenes import SCENES
from .splits import DESIGNS, split, subsample

N_TEST = 24
N_TRAIN_FULL = data.N_CAMERAS - N_TEST
SEEDS = (0, 1, 2)
SPARSE_COUNTS = (12, 24, 48)
RESULTS = Path("results")


@dataclass(frozen=True)
class Run:
    scene: str
    design: str
    seed: int
    n_train: int = N_TRAIN_FULL
    poses: str = "true"

    @property
    def run_id(self) -> str:
        return f"{self.scene}-{self.design}-s{self.seed}-n{self.n_train}-{self.poses}"


def matrix() -> list[Run]:
    runs = [Run(sc, de, se) for sc in SCENES for de in DESIGNS for se in SEEDS]                 # E1
    runs += [Run(sc, de, 0, n) for sc in SCENES for de in DESIGNS for n in SPARSE_COUNTS]       # E2
    runs += [Run(sc, de, 0, poses="sfm") for sc in SCENES for de in DESIGNS]                    # E3
    return runs


def cameras_for(run: Run) -> tuple[np.ndarray, np.ndarray]:
    """Train and test camera indices for a run. The split never depends on the pose source."""
    d = data.load(run.scene, data.FIELD_SIZE)
    train, test = split(d["c2w"], run.design, N_TEST, seed=100 * SCENES.index(run.scene) + run.seed)
    return subsample(train, run.n_train, run.seed), test


def run_one(run: Run, cfg: FieldConfig = FieldConfig(), threads: int = 4) -> pd.DataFrame:
    import torch

    torch.set_num_threads(threads)
    d = data.load(run.scene, data.FIELD_SIZE)
    true_c2w = d["c2w"]
    train, test = cameras_for(run)
    c2w = true_c2w
    if run.poses == "sfm":
        z = np.load(RESULTS / "sfm" / f"{run.scene}.npz")
        c2w = z["c2w"]
        train, test = train[z["ok"][train]], test[z["ok"][test]]
    o, dirs = data.ray_bundle(c2w, data.FIELD_SIZE)

    t0 = time.time()
    model = train_field(o[train].reshape(-1, 3), dirs[train].reshape(-1, 3),
                        d["rgb"][train].reshape(-1, 3), cfg, seed=run.seed)
    seconds = time.time() - t0

    angle = angle_to_nearest(true_c2w[test], true_c2w[train])
    rows, renders = [], {}
    fit_views = train[:: max(1, len(train) // 4)][:4]
    for role, views in (("test", test), ("train", fit_views)):
        for k, v in enumerate(views):
            rgb, depth, acc = render_image(model, o[v], dirs[v])
            rgb = np.clip(rgb, 0, 1)
            rows.append({
                **asdict(run), "run_id": run.run_id, "role": role, "view": int(v),
                "angle_nearest": float(angle[k]) if role == "test" else 0.0,
                "psnr": metrics.psnr(rgb, d["rgb"][v]), "ssim": metrics.ssim(rgb, d["rgb"][v]),
                "depth_mae": metrics.depth_mae(depth, d["depth"][v]),
                "iou": metrics.silhouette_iou(acc, d["depth"][v]), "train_seconds": seconds,
            })
            if role == "test":
                renders[int(v)] = (rgb, depth, acc)
    df = pd.DataFrame(rows)

    if run.seed == 0 and run.n_train == N_TRAIN_FULL and run.poses == "true":
        t = df[df.role == "test"].sort_values("psnr")
        picks = {"worst": int(t.view.iloc[0]), "median": int(t.view.iloc[len(t) // 2])}
        out = RESULTS / "examples"
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"{run.run_id}.npz", **{
            f"{name}_{key}": val for name, v in picks.items() for key, val in (
                ("pred", renders[v][0].astype(np.float16)), ("true", d["rgb"][v].astype(np.float16)),
                ("pred_depth", renders[v][1].astype(np.float16)), ("true_depth", d["depth"][v].astype(np.float16)),
                ("opacity", renders[v][2].astype(np.float16)), ("view", np.array(v)))})

    out = RESULTS / "runs"
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"{run.run_id}.tmp"
    df.to_parquet(tmp)
    tmp.rename(out / f"{run.run_id}.parquet")       # a run counts as done only once its file is complete
    return df


def _worker(run: Run) -> str:
    t0 = time.time()
    df = run_one(run)
    psnr = df[df.role == "test"].psnr.mean()
    return f"{run.run_id}: test PSNR {psnr:.2f} dB in {time.time() - t0:.0f} s"


def run_suite(workers: int = 2, only: str | None = None) -> None:
    todo = [r for r in matrix() if not (RESULTS / "runs" / f"{r.run_id}.parquet").exists()
            and (only is None or only in r.run_id)]
    for sc in SCENES:                                # render in the parent, so workers only read the cache
        data.load(sc, data.FIELD_SIZE)
    print(f"{len(todo)} runs to do, {len(matrix()) - len(todo)} already done", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool:
        for k, line in enumerate(pool.imap_unordered(_worker, todo), 1):
            print(f"[{k}/{len(todo)}] {line}", flush=True)


# ---------------------------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------------------------

METRICS = ("psnr", "ssim", "depth_mae", "iou")


def load_views() -> pd.DataFrame:
    files = sorted((RESULTS / "runs").glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _boot_ci(values: np.ndarray, rng: np.random.Generator, n: int = 5000) -> list[float]:
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), (n, len(values)))
    lo, hi = np.percentile(values[idx].mean(axis=1), [2.5, 97.5])
    return [float(lo), float(hi)]


def summarise() -> dict:
    """Aggregate every finished run into the numbers the report quotes.

    The unit of analysis is the run, not the view: views in one run share a model.
    Intervals are bootstrap intervals over (scene, seed) pairs, with the two designs
    kept paired inside each resample.
    """
    rng = np.random.default_rng(0)
    views = load_views()
    test = views[views.role == "test"]
    per_run = test.groupby(["scene", "design", "seed", "n_train", "poses"], as_index=False)[list(METRICS)].mean()
    out: dict = {"n_runs": int(views.run_id.nunique()), "n_test_views": int(len(test)),
                 "train_seconds_mean": float(views.groupby("run_id").train_seconds.first().mean())}

    # E1: the two designs at full training density, true poses.
    e1 = per_run[(per_run.n_train == N_TRAIN_FULL) & (per_run.poses == "true")]
    wide = e1.pivot(index=["scene", "seed"], columns="design", values=list(METRICS))
    out["e1"] = {"pairs": int(len(wide)), "designs": {}, "gap": {}, "per_scene_gap": {}}
    for de in DESIGNS:
        out["e1"]["designs"][de] = {m: {"mean": float(wide[m][de].mean()), "ci": _boot_ci(wide[m][de].values, rng)}
                                    for m in METRICS}
    for m in METRICS:
        diff = (wide[m]["random"] - wide[m]["cap"]).values
        out["e1"]["gap"][m] = {"mean": float(diff.mean()), "ci": _boot_ci(diff, rng),
                               "pairs_same_sign": int((np.sign(diff) == np.sign(diff.mean())).sum())}
    for sc in SCENES:
        if sc in wide.index.get_level_values("scene"):
            w = wide.xs(sc, level="scene")
            out["e1"]["per_scene_gap"][sc] = {m: float((w[m]["random"] - w[m]["cap"]).mean()) for m in METRICS}
    ratio = (wide["depth_mae"]["cap"] / wide["depth_mae"]["random"]).values
    mse = lambda p: 10 ** (-p / 10)  # noqa: E731
    ratio_mse = (mse(wide["psnr"]["cap"]) / mse(wide["psnr"]["random"])).values
    out["e1"]["depth_ratio"] = {"mean": float(ratio.mean()), "ci": _boot_ci(ratio, rng)}
    out["e1"]["image_mse_ratio"] = {"mean": float(ratio_mse.mean()), "ci": _boot_ci(ratio_mse, rng)}

    # Training-view fit, to show the two designs fit their training data equally well.
    fit = views[(views.role == "train") & (views.n_train == N_TRAIN_FULL) & (views.poses == "true")]
    out["e1"]["train_psnr"] = {de: float(fit[fit.design == de].psnr.mean()) for de in DESIGNS}

    # Error against angle to the nearest training camera, pooled over E1 test views.
    t1 = test[(test.n_train == N_TRAIN_FULL) & (test.poses == "true")]
    edges = [0, 9, 12, 16, 20, 26, 90]
    t1 = t1.assign(bin=pd.cut(t1.angle_nearest, edges))
    out["angle_curve"] = [
        {"lo": float(b.left), "hi": float(b.right), "n": int(len(g)), "angle_mean": float(g.angle_nearest.mean()),
         **{m: float(g[m].mean()) for m in METRICS}}
        for b, g in t1.groupby("bin", observed=True)]
    rho = [g[["angle_nearest", "psnr"]].corr(method="spearman").iloc[0, 1]
           for _, g in t1[t1.design == "cap"].groupby("run_id")]
    out["angle_spearman_cap"] = {"mean": float(np.mean(rho)), "ci": _boot_ci(np.array(rho), rng),
                                 "runs_negative": int(np.sum(np.array(rho) < 0)), "runs": len(rho)}
    out["angle_range"] = {de: [float(t1[t1.design == de].angle_nearest.min()),
                               float(t1[t1.design == de].angle_nearest.median()),
                               float(t1[t1.design == de].angle_nearest.max())] for de in DESIGNS}
    out["psnr_depth_spearman"] = float(t1[["psnr", "depth_mae"]].corr(method="spearman").iloc[0, 1])

    # E2: fewer training views (seed 0 only), with the full-density seed-0 runs as the last point.
    e2 = per_run[(per_run.seed == 0) & (per_run.poses == "true")]
    out["e2"] = [{"n_train": int(n), **{de: {m: float(g[g.design == de][m].mean()) for m in METRICS} for de in DESIGNS},
                  "gap_psnr": float(g[g.design == "random"].psnr.mean() - g[g.design == "cap"].psnr.mean()),
                  "per_scene_gap_psnr": {sc: float(g[(g.design == "random") & (g.scene == sc)].psnr.mean()
                                                   - g[(g.design == "cap") & (g.scene == sc)].psnr.mean())
                                         for sc in SCENES}}
                 for n, g in e2.groupby("n_train")]

    # E3: recovered poses against true poses, same split and seed.
    e3 = per_run[(per_run.seed == 0) & (per_run.n_train == N_TRAIN_FULL)]
    if (e3.poses == "sfm").any():
        w3 = e3.pivot(index=["scene", "design"], columns="poses", values=list(METRICS))
        out["e3"] = {"rows": [{"scene": sc, "design": de, **{f"{m}_{p}": float(w3[m][p][(sc, de)])
                                                              for m in METRICS for p in ("true", "sfm")}}
                              for sc, de in w3.index],
                     "psnr_cost": {"mean": float((w3["psnr"]["true"] - w3["psnr"]["sfm"]).mean()),
                                   "max": float((w3["psnr"]["true"] - w3["psnr"]["sfm"]).max())},
                     "sfm": {sc: json.loads((RESULTS / "sfm" / f"{sc}.json").read_text()) for sc in SCENES
                             if (RESULTS / "sfm" / f"{sc}.json").exists()}}
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=2))
    per_run.to_csv(RESULTS / "per_run.csv", index=False)
    return out
