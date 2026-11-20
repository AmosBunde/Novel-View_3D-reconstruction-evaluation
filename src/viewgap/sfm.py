"""Incremental structure from motion with known intrinsics.

Camera poses and a sparse point cloud are recovered from the images alone:

1. SIFT keypoints and descriptors in every image;
2. ratio-test matching between every pair of images, kept only if an essential
   matrix fitted with RANSAC explains enough of the matches;
3. the verified matches are chained into tracks (one track = one 3D point);
4. two views with many matches and enough parallax start the reconstruction;
5. the remaining views are added one at a time by PnP against the points
   already triangulated, each followed by triangulation of the tracks it
   completes, with bundle adjustment at intervals and at the end.

OpenCV supplies the feature detector and the minimal solvers (essential matrix,
PnP). The track building, the incremental loop, the triangulation and the bundle
adjustment are written here. The result is in an arbitrary frame and scale;
``align`` maps it onto ground truth for evaluation.

Poses inside this module are world-to-camera: x_cam = R @ x_world + t.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

MIN_PAIR_INLIERS = 20
REPROJ_PX = 2.0
MIN_PARALLAX_DEG = 2.0


@dataclass
class Reconstruction:
    R: dict[int, np.ndarray]            # image index -> 3x3 world-to-camera rotation
    t: dict[int, np.ndarray]            # image index -> translation
    points: dict[int, np.ndarray]       # track id -> 3D point
    tracks: list[dict[int, int]]        # track id -> {image index: keypoint index}
    keypoints: list[np.ndarray]         # per image, (N, 2) pixel coordinates

    def c2w(self, n_images: int) -> tuple[np.ndarray, np.ndarray]:
        """Camera-to-world matrices (n, 4, 4) and a mask of which images were registered."""
        out = np.tile(np.eye(4), (n_images, 1, 1))
        ok = np.zeros(n_images, dtype=bool)
        for i, R in self.R.items():
            out[i, :3, :3], out[i, :3, 3], ok[i] = R.T, -R.T @ self.t[i], True
        return out, ok


def detect(images: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """SIFT on float RGB images in [0, 1]. Returns keypoint coordinates and descriptors."""
    sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.015)
    kps, descs = [], []
    for im in images:
        grey = cv2.cvtColor((im * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        kp, d = sift.detectAndCompute(grey, None)
        kps.append(np.array([k.pt for k in kp], dtype=np.float64).reshape(-1, 2))
        descs.append(d if d is not None else np.zeros((0, 128), np.float32))
    return kps, descs


def _essential(p1: np.ndarray, p2: np.ndarray, K: np.ndarray):
    """Essential matrix by five-point RANSAC. OpenCV's solver can abort on a degenerate
    sample; such a pair is treated as having no geometry rather than stopping the run."""
    try:
        E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0, maxIters=400)
    except cv2.error:
        return None, None
    if E is None or E.shape != (3, 3):
        return None, None
    return E, mask


def _match_one(args):
    i, j, kp_i, kp_j, d_i, d_j, K = args
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d_i, d_j, k=2)
    good = [(m.queryIdx, m.trainIdx) for m, s in (p for p in knn if len(p) == 2) if m.distance < 0.75 * s.distance]
    if len(good) < MIN_PAIR_INLIERS:
        return None
    good = np.array(good)
    _, uniq = np.unique(good[:, 1], return_index=True)      # one match per keypoint in j
    good = good[np.sort(uniq)]
    if len(good) < MIN_PAIR_INLIERS:
        return None
    E, mask = _essential(kp_i[good[:, 0]], kp_j[good[:, 1]], K)
    if E is None or mask is None or int(mask.sum()) < MIN_PAIR_INLIERS:
        return None
    return (i, j), good[mask.ravel() > 0]


def match_pairs(kps, descs, K: np.ndarray, threads: int = 8) -> dict[tuple[int, int], np.ndarray]:
    """Geometrically verified matches for every image pair: (i, j) -> (M, 2) keypoint indices.

    Every pair is tried; no pose or ordering information is used to choose which.
    """
    from concurrent.futures import ThreadPoolExecutor

    n = len(kps)
    jobs = [(i, j, kps[i], kps[j], descs[i], descs[j], K) for i in range(n) for j in range(i + 1, n)
            if len(descs[i]) >= 2 and len(descs[j]) >= 2]
    with ThreadPoolExecutor(threads) as pool:                # OpenCV releases the GIL
        return dict(r for r in pool.map(_match_one, jobs) if r is not None)


def build_tracks(pairs: dict[tuple[int, int], np.ndarray]) -> list[dict[int, int]]:
    """Chain pairwise matches into tracks with union-find; drop tracks that visit an image twice."""
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(a):
        root = a
        while parent.setdefault(root, root) != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    for (i, j), m in pairs.items():
        for a, b in m:
            ra, rb = find((i, int(a))), find((j, int(b)))
            if ra != rb:
                parent[ra] = rb
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for node in list(parent):
        groups.setdefault(find(node), []).append(node)
    tracks = []
    for members in groups.values():
        images = [im for im, _ in members]
        if len(members) >= 2 and len(set(images)) == len(images):
            tracks.append({im: kp for im, kp in members})
    return tracks


def _project(R, t, X, K):
    x = X @ R.T + t
    return (x[:, :2] / x[:, 2:3]) * np.array([K[0, 0], K[1, 1]]) + K[:2, 2], x[:, 2]


def triangulate(Rs, ts, pts2d, K) -> np.ndarray:
    """Linear (DLT) triangulation of one point seen in two or more views."""
    A = []
    for R, t, (u, v) in zip(Rs, ts, pts2d):
        P = K @ np.hstack([R, t[:, None]])
        A += [u * P[2] - P[0], v * P[2] - P[1]]
    X = np.linalg.svd(np.array(A))[2][-1]
    return X[:3] / X[3]


def _parallax_deg(Rs, ts, X) -> float:
    centres = [-R.T @ t for R, t in zip(Rs, ts)]
    rays = [(X - c) / np.linalg.norm(X - c) for c in centres]
    best = 0.0
    for a in range(len(rays)):
        for b in range(a + 1, len(rays)):
            best = max(best, np.degrees(np.arccos(np.clip(rays[a] @ rays[b], -1, 1))))
    return best


def _try_triangulate(rec: Reconstruction, track_id: int, K) -> None:
    track = rec.tracks[track_id]
    views = [i for i in track if i in rec.R]
    if len(views) < 2:
        return
    Rs, ts = [rec.R[i] for i in views], [rec.t[i] for i in views]
    pts = [rec.keypoints[i][track[i]] for i in views]
    X = triangulate(Rs, ts, pts, K)
    if _parallax_deg(Rs, ts, X) < MIN_PARALLAX_DEG:
        return
    for R, t, p in zip(Rs, ts, pts):
        uv, z = _project(R, t, X[None], K)
        if z[0] <= 0 or np.linalg.norm(uv[0] - p) > REPROJ_PX:
            return
    rec.points[track_id] = X


def _max_reproj(rec: Reconstruction, track_id: int, K) -> float:
    track, X = rec.tracks[track_id], rec.points[track_id]
    worst = 0.0
    for c, kp in track.items():
        if c in rec.R:
            uv, z = _project(rec.R[c], rec.t[c], X[None], K)
            worst = max(worst, np.inf if z[0] <= 0 else float(np.linalg.norm(uv[0] - rec.keypoints[c][kp])))
    return worst


def bundle_adjust(rec: Reconstruction, K, fixed: int, max_nfev: int = 25) -> float:
    """Jointly refine all registered cameras and points; returns the RMS reprojection error in pixels.

    One camera is held fixed to remove the rotation and translation freedom. Scale is
    left free: it cannot be observed, and the solver has no reason to move along it.
    """
    # Points that are already far off are dropped first, so the solve itself can use a plain
    # squared loss, which converges in a handful of iterations where a robust loss takes dozens.
    for p in [p for p in rec.points if _max_reproj(rec, p, K) > 4 * REPROJ_PX]:
        rec.points.pop(p)
    cams = sorted(rec.R)
    free = [c for c in cams if c != fixed]
    cam_slot = {c: k for k, c in enumerate(free)}
    pids = sorted(rec.points)
    pt_slot = {p: k for k, p in enumerate(pids)}
    obs = [(c, p, rec.keypoints[c][rec.tracks[p][c]]) for p in pids for c in rec.tracks[p] if c in rec.R]
    oc = np.array([cam_slot.get(c, -1) for c, _, _ in obs])
    op = np.array([pt_slot[p] for _, p, _ in obs])
    uv = np.array([o for _, _, o in obs])
    n_c, n_p = len(free), len(pids)
    R_fixed, t_fixed = rec.R[fixed], rec.t[fixed]

    x0 = np.concatenate(
        [np.concatenate([Rotation.from_matrix(rec.R[c]).as_rotvec(), -rec.R[c].T @ rec.t[c]]) for c in free]
        + [np.array([rec.points[p] for p in pids]).ravel()])

    def unpack(x):
        # Each camera is a rotation vector and a centre. Optimising the centre rather than the
        # translation keeps a small rotation from swinging the camera around the world origin,
        # which makes the problem much better conditioned.
        cam = x[:6 * n_c].reshape(n_c, 6)
        Rm = Rotation.from_rotvec(cam[:, :3]).as_matrix()
        return Rm, -np.einsum("nij,nj->ni", Rm, cam[:, 3:]), x[6 * n_c:].reshape(n_p, 3)

    def residual(x):
        Rm, tm, X = unpack(x)
        is_fixed = oc < 0
        R_obs = np.where(is_fixed[:, None, None], R_fixed, Rm[np.maximum(oc, 0)])
        t_obs = np.where(is_fixed[:, None], t_fixed, tm[np.maximum(oc, 0)])
        xc = np.einsum("nij,nj->ni", R_obs, X[op]) + t_obs
        proj = xc[:, :2] / xc[:, 2:3] * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]
        return (proj - uv).ravel()

    sparsity = lil_matrix((2 * len(obs), x0.size), dtype=int)
    rows = np.arange(len(obs))
    for k in range(6):
        ok = oc >= 0
        sparsity[2 * rows[ok], 6 * oc[ok] + k] = 1
        sparsity[2 * rows[ok] + 1, 6 * oc[ok] + k] = 1
    for k in range(3):
        sparsity[2 * rows, 6 * n_c + 3 * op + k] = 1
        sparsity[2 * rows + 1, 6 * n_c + 3 * op + k] = 1

    sol = least_squares(residual, x0, jac_sparsity=sparsity, loss="linear",
                        x_scale="jac", method="trf", max_nfev=max_nfev,
                        tr_solver="lsmr", tr_options={"atol": 1e-8, "btol": 1e-8})
    Rm, tm, X = unpack(sol.x)
    for c in free:
        rec.R[c], rec.t[c] = Rm[cam_slot[c]], tm[cam_slot[c]]
    err = np.linalg.norm(residual(sol.x).reshape(-1, 2), axis=1)
    bad = set(op[err > 2 * REPROJ_PX])
    for p in pids:
        if pt_slot[p] in bad:
            rec.points.pop(p)
        else:
            rec.points[p] = X[pt_slot[p]]
    return float(np.sqrt(np.mean(err[err <= 2 * REPROJ_PX] ** 2)))


def reconstruct(images: np.ndarray, K: np.ndarray, ba_every: int = 12, log=print) -> Reconstruction:
    """Run the whole pipeline on float RGB images (N, H, W, 3) with shared intrinsics K."""
    kps, descs = detect(images)
    pairs = match_pairs(kps, descs, K)
    tracks = build_tracks(pairs)
    log(f"sfm: {sum(len(k) for k in kps) // len(kps)} keypoints per image, "
        f"{len(pairs)} verified pairs, {len(tracks)} tracks")
    rec = Reconstruction({}, {}, {}, tracks, kps)
    by_image: dict[int, list[int]] = {}
    for tid, tr in enumerate(tracks):
        for im in tr:
            by_image.setdefault(im, []).append(tid)

    # Initial pair: the most matches among pairs whose points triangulate with real parallax.
    first = None
    for (i, j), m in sorted(pairs.items(), key=lambda kv: -len(kv[1]))[:60]:
        p1, p2 = kps[i][m[:, 0]], kps[j][m[:, 1]]
        E, _ = _essential(p1, p2, K)
        if E is None:
            continue
        n_good, R, t, _ = cv2.recoverPose(E, p1, p2, K)
        if n_good < MIN_PAIR_INLIERS:
            continue
        Rs, ts = [np.eye(3), R], [np.zeros(3), t.ravel()]
        angles = [_parallax_deg(Rs, ts, triangulate(Rs, ts, [a, b], K)) for a, b in zip(p1[:80], p2[:80])]
        if np.median(angles) > 6.0:
            first = (i, j, R, t.ravel())
            break
    if first is None:
        raise RuntimeError("no image pair with enough matches and parallax to start from")
    i, j, R, t = first
    rec.R[i], rec.t[i], rec.R[j], rec.t[j] = np.eye(3), np.zeros(3), R, t
    for tid in set(by_image[i]) & set(by_image[j]):
        _try_triangulate(rec, tid, K)
    bundle_adjust(rec, K, fixed=i)

    since_ba = 0
    while len(rec.R) < len(images):
        counts = {im: sum(1 for tid in tids if tid in rec.points)
                  for im, tids in by_image.items() if im not in rec.R}
        if not counts:
            break
        im = max(counts, key=counts.get)
        if counts[im] < 12:
            break
        tids = [tid for tid in by_image[im] if tid in rec.points]
        obj = np.array([rec.points[tid] for tid in tids])
        img = np.array([kps[im][tracks[tid][im]] for tid in tids])
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, reprojectionError=REPROJ_PX,
                                                 iterationsCount=300, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < 12:
            by_image.pop(im)            # cannot be registered; leave it out and carry on
            continue
        _, rvec, tvec = cv2.solvePnP(obj[inl.ravel()], img[inl.ravel()], K, None, rvec, tvec,
                                     useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        rec.R[im], rec.t[im] = cv2.Rodrigues(rvec)[0], tvec.ravel()
        for tid in by_image[im]:
            if tid not in rec.points:
                _try_triangulate(rec, tid, K)
        since_ba += 1
        if since_ba >= ba_every:
            bundle_adjust(rec, K, fixed=i, max_nfev=12)
            since_ba = 0
    rms = bundle_adjust(rec, K, fixed=i, max_nfev=40)
    for tid in range(len(tracks)):
        if tid not in rec.points:
            _try_triangulate(rec, tid, K)
    rms = bundle_adjust(rec, K, fixed=i, max_nfev=40)
    log(f"sfm: registered {len(rec.R)} of {len(images)} images, {len(rec.points)} points, "
        f"reprojection RMS {rms:.2f} px")
    return rec


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity (scale, R, t) with dst ~ scale * R @ src + t (Umeyama, 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / xs.var(axis=0).sum()
    return float(scale), R, mu_d - scale * R @ mu_s


def align(c2w_est: np.ndarray, c2w_true: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Map estimated poses into the ground-truth frame with one similarity transform.

    This fixes the frame and scale, which structure from motion cannot observe. It does
    not correct any individual camera: all poses move together.
    """
    s, R, t = umeyama(c2w_est[ok, :3, 3], c2w_true[ok, :3, 3])
    out = c2w_est.copy()
    out[:, :3, 3] = s * c2w_est[:, :3, 3] @ R.T + t
    out[:, :3, :3] = R @ c2w_est[:, :3, :3]
    return out


def pose_errors(c2w_aligned: np.ndarray, c2w_true: np.ndarray, ok: np.ndarray) -> dict[str, float]:
    pos = np.linalg.norm(c2w_aligned[ok, :3, 3] - c2w_true[ok, :3, 3], axis=1)
    rel = np.einsum("nij,nkj->nik", c2w_aligned[ok, :3, :3], c2w_true[ok, :3, :3])
    ang = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    return {"registered": int(ok.sum()), "position_rmse": float(np.sqrt((pos ** 2).mean())),
            "position_max": float(pos.max()), "rotation_mean_deg": float(ang.mean()),
            "rotation_max_deg": float(ang.max())}
