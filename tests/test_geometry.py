"""Checks on the pieces the study's numbers rest on: cameras, ground truth, splits, metrics, SfM maths."""
import numpy as np
import pytest

from viewgap import metrics, sfm
from viewgap.cameras import RADIUS, TARGET, angle_to_nearest, hemisphere_cameras, intrinsics, rays
from viewgap.scenes import SCENES, intersect, make_scene, render
from viewgap.splits import split, subsample


def test_cameras_sit_on_the_sphere_and_look_at_the_target():
    c2w = hemisphere_cameras(40)
    centres = c2w[:, :3, 3]
    assert np.allclose(np.linalg.norm(centres - TARGET, axis=1), RADIUS)
    forward = c2w[:, :3, 2]
    assert np.allclose(forward, (TARGET - centres) / RADIUS, atol=1e-9)
    for m in c2w:                                   # proper rotations
        assert np.allclose(m[:3, :3] @ m[:3, :3].T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(m[:3, :3]), 1.0)


def test_centre_pixel_ray_points_along_the_optical_axis():
    c2w = hemisphere_cameras(5)[2]
    intr = intrinsics(64)
    _, d = rays(c2w, intr)
    centre = d[31:33, 31:33].mean(axis=(0, 1))
    assert np.allclose(centre / np.linalg.norm(centre), c2w[:3, 2], atol=1e-3)


def test_depth_reprojects_to_the_same_world_point_from_two_cameras():
    """Ground-truth depth is exact: a surface point seen by camera A lands, in camera B,
    on a pixel whose own depth puts it at the same world point."""
    scene = make_scene("blocks")
    intr = intrinsics(96)
    a, b = hemisphere_cameras(60)[[10, 23]]
    _, depth_a = render(scene, a, intr, supersample=1)
    o, d = rays(a, intr)
    pts = (o + depth_a[..., None] * d)[np.isfinite(depth_a)]
    cam = (pts - b[:3, 3]) @ b[:3, :3]
    t_b, _, _ = intersect(scene, np.broadcast_to(b[:3, 3], pts.shape).copy(),
                          (pts - b[:3, 3]) / np.linalg.norm(pts - b[:3, 3], axis=1, keepdims=True))
    dist = np.linalg.norm(pts - b[:3, 3], axis=1)
    visible = np.abs(t_b - dist) < 1e-3              # not occluded from camera B
    assert visible.mean() > 0.4
    assert (cam[visible, 2] > 0).all()


@pytest.mark.parametrize("name", SCENES)
def test_scenes_are_deterministic_and_fit_the_bounding_box(name):
    a, b = make_scene(name), make_scene(name)
    assert np.array_equal(a.box_min, b.box_min) and np.array_equal(a.tex_table, b.tex_table)
    assert a.box_min.min() >= -1 and a.box_max.max() <= 1
    if a.sph_c.size:
        assert (a.sph_c[:, 2] + a.sph_r).max() <= 0.85


@pytest.mark.parametrize("design", ["random", "cap"])
def test_splits_are_disjoint_and_the_same_size(design):
    c2w = hemisphere_cameras(120)
    train, test = split(c2w, design, 24, seed=3)
    assert len(test) == 24 and len(train) == 96
    assert not set(train) & set(test)
    assert np.array_equal(split(c2w, design, 24, seed=3)[1], test)      # reproducible


def test_cap_test_cameras_are_further_from_training_cameras_than_random_ones():
    c2w = hemisphere_cameras(120)
    far = []
    for design in ("random", "cap"):
        train, test = split(c2w, design, 24, seed=0)
        far.append(np.median(angle_to_nearest(c2w[test], c2w[train])))
    assert far[1] > far[0] + 3


def test_subsample_keeps_a_subset():
    train = np.arange(96)
    assert set(subsample(train, 12, 0)) <= set(train) and len(subsample(train, 12, 0)) == 12
    assert np.array_equal(subsample(train, 200, 0), train)


def test_metrics_on_known_cases():
    rng = np.random.default_rng(0)
    im = rng.uniform(size=(32, 32, 3))
    assert metrics.psnr(im, im) > 100
    assert np.isclose(metrics.psnr(im, np.clip(im + 0.1, 0, 1.1)), 20.0, atol=0.01)
    assert np.isclose(metrics.ssim(im, im), 1.0)
    assert metrics.ssim(im, rng.uniform(size=im.shape)) < 0.2
    depth = np.full((4, 4), 2.0)
    depth[0] = np.inf
    assert np.isclose(metrics.depth_mae(np.full((4, 4), 2.5), depth), 0.5)
    assert np.isclose(metrics.silhouette_iou(np.isfinite(depth).astype(float), depth), 1.0)


def test_triangulation_recovers_a_known_point():
    intr = intrinsics(200)
    c2w = hemisphere_cameras(30)[[0, 7, 15]]
    X = np.array([0.2, -0.1, 0.3])
    Rs = [c[:3, :3].T for c in c2w]
    ts = [-c[:3, :3].T @ c[:3, 3] for c in c2w]
    uv = [sfm._project(R, t, X[None], intr.K)[0][0] for R, t in zip(Rs, ts)]
    assert np.allclose(sfm.triangulate(Rs, ts, uv, intr.K), X, atol=1e-8)


def test_umeyama_recovers_a_known_similarity():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(20, 3))
    R = sfm.Rotation.from_rotvec([0.3, -0.2, 0.9]).as_matrix()
    dst = 2.5 * src @ R.T + np.array([1.0, -2.0, 0.5])
    s, R_hat, t_hat = sfm.umeyama(src, dst)
    assert np.isclose(s, 2.5) and np.allclose(R_hat, R) and np.allclose(t_hat, [1.0, -2.0, 0.5])


def test_bundle_adjustment_pulls_perturbed_cameras_back():
    intr = intrinsics(300)
    rng = np.random.default_rng(2)
    c2w = hemisphere_cameras(40)[:6]
    X = rng.uniform(-0.5, 0.5, size=(60, 3)) + np.array([0, 0, 0.3])
    Rs = [c[:3, :3].T for c in c2w]
    ts = [-c[:3, :3].T @ c[:3, 3] for c in c2w]
    kps = [sfm._project(R, t, X, intr.K)[0] for R, t in zip(Rs, ts)]
    tracks = [{c: p for c in range(6)} for p in range(60)]
    rec = sfm.Reconstruction(
        R={c: (sfm.Rotation.from_rotvec(rng.normal(scale=0.01, size=3)).as_matrix() @ Rs[c]) if c else Rs[c] for c in range(6)},
        t={c: ts[c] + (rng.normal(scale=0.01, size=3) if c else 0) for c in range(6)},
        points={p: X[p] + rng.normal(scale=0.01, size=3) for p in range(60)}, tracks=tracks, keypoints=kps)
    rms = sfm.bundle_adjust(rec, intr.K, fixed=0, max_nfev=50)
    assert rms < 0.05
