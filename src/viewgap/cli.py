"""Command line for the study. Run ``viewgap --help``."""
from __future__ import annotations

import argparse
import json

import numpy as np


def cmd_render(args) -> None:
    from . import data
    from .scenes import SCENES
    for sc in SCENES:
        for size in (data.FIELD_SIZE, data.SFM_SIZE):
            d = data.load(sc, size)
            print(f"{sc} at {size} px: {d['rgb'].shape[0]} views, "
                  f"{np.isfinite(d['depth']).mean():.0%} of pixels on a surface")


def cmd_sfm(args) -> None:
    from . import data, sfm
    from .cameras import intrinsics
    from .scenes import SCENES
    from .suite import RESULTS
    out = RESULTS / "sfm"
    out.mkdir(parents=True, exist_ok=True)
    for sc in SCENES:
        d = data.load(sc, data.SFM_SIZE)
        rec = sfm.reconstruct(d["rgb"], intrinsics(data.SFM_SIZE).K)
        est, ok = rec.c2w(d["c2w"].shape[0])
        aligned = sfm.align(est, d["c2w"], ok)
        errors = sfm.pose_errors(aligned, d["c2w"], ok)
        s, R, t = sfm.umeyama(est[ok, :3, 3], d["c2w"][ok, :3, 3])
        points = np.array(list(rec.points.values())) @ R.T * s + t
        errors["points"] = int(len(points))
        np.savez_compressed(out / f"{sc}.npz", c2w=aligned, ok=ok, points=points.astype(np.float32))
        (out / f"{sc}.json").write_text(json.dumps(errors, indent=2))
        print(sc, json.dumps(errors))


def cmd_one(args) -> None:
    from .suite import Run, run_one
    df = run_one(Run(args.scene, args.design, args.seed, args.n_train, args.poses), threads=args.threads)
    print(df[df.role == "test"][["psnr", "ssim", "depth_mae", "iou"]].mean().round(4).to_string())


def cmd_suite(args) -> None:
    from .suite import run_suite
    run_suite(workers=args.workers, only=args.only)


def cmd_summarise(args) -> None:
    from .suite import summarise
    s = summarise()
    g = s["e1"]["gap"]["psnr"]
    print(f"{s['n_runs']} runs. PSNR, random minus cap: {g['mean']:.2f} dB "
          f"(95% interval {g['ci'][0]:.2f} to {g['ci'][1]:.2f})")


def cmd_figures(args) -> None:
    from .figures import make_all
    make_all()


def main() -> None:
    from .scenes import SCENES
    from .splits import DESIGNS
    from .suite import N_TRAIN_FULL
    p = argparse.ArgumentParser(prog="viewgap", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("render", help="render and cache every scene (about 15 minutes, once)").set_defaults(fn=cmd_render)
    sub.add_parser("sfm", help="recover camera poses from the images and score them").set_defaults(fn=cmd_sfm)
    one = sub.add_parser("one", help="train and score a single radiance field")
    one.add_argument("--scene", choices=SCENES, default="clutter")
    one.add_argument("--design", choices=DESIGNS, default="random")
    one.add_argument("--seed", type=int, default=0)
    one.add_argument("--n-train", type=int, default=N_TRAIN_FULL)
    one.add_argument("--poses", choices=("true", "sfm"), default="true")
    one.add_argument("--threads", type=int, default=8)
    one.set_defaults(fn=cmd_one)
    suite = sub.add_parser("suite", help="run every experiment that has not been run yet")
    suite.add_argument("--workers", type=int, default=2)
    suite.add_argument("--only", help="only runs whose id contains this text")
    suite.set_defaults(fn=cmd_suite)
    sub.add_parser("summarise", help="aggregate finished runs into results/summary.json").set_defaults(fn=cmd_summarise)
    sub.add_parser("figures", help="draw every figure in docs/img").set_defaults(fn=cmd_figures)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
