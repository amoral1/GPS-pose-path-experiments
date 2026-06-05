"""
Gaussian Splat export (.splat format for antimatter15.com/splat viewer).

Two exporters:
  export_naive_splat    — fixed scale/opacity (baseline, matches splatv1)
  export_improved_splat — kNN-adaptive scale + confidence-weighted opacity (splatv2)

Format: 32 bytes per Gaussian
  xyz       12 bytes  (3x float32)
  scale     12 bytes  (3x float32, isotropic)
  rgba       4 bytes  (uint8 x4)
  rotation   4 bytes  (uint8 quaternion, identity = [255,128,128,128])

Usage:
    from mapanything_pipeline.splat import export_improved_splat, subsample
"""

import os
import numpy as np
from pathlib import Path


def subsample(
    pts: np.ndarray,
    cols: np.ndarray,
    confs: np.ndarray,
    max_gaussians: int = 4_000_000,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Subsample pts/cols/confs with a single shared index.

    Always index all three arrays together — using separate indices causes
    shape mismatches in export_improved_splat and inflates/deflates opacity.

    Args:
        max_gaussians: browser sweet spot is 3-5M (96-160 MB).
                       32.5M Gaussians (~992 MB) causes Load Failed in browser.
    """
    n = len(pts)
    if n <= max_gaussians:
        return pts, cols, confs

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, max_gaussians, replace=False)
    print(f"Subsampled {n:,} -> {max_gaussians:,} Gaussians")
    return pts[idx], cols[idx], confs[idx]


def export_naive_splat(
    path: str,
    pts: np.ndarray,
    cols: np.ndarray,
    fixed_scale: float = 0.05,
    opacity: int = 200,
) -> None:
    """
    Baseline .splat export — fixed scale and opacity per Gaussian.
    Matches the splatv1 output. Useful as a comparison baseline.
    """
    n      = len(pts)
    xyz    = pts.astype(np.float32)
    scales = np.full((n, 3), fixed_scale, dtype=np.float32)

    rgba_u8 = np.zeros((n, 4), dtype=np.uint8)
    rgba_u8[:, :3] = (np.clip(cols, 0, 1) * 255).astype(np.uint8)
    rgba_u8[:, 3]  = opacity

    rot_u8 = np.full((n, 4), 128, dtype=np.uint8)
    rot_u8[:, 0] = 255

    buf = np.zeros((n, 32), dtype=np.uint8)
    buf[:, 0:12]  = xyz.view(np.uint8).reshape(n, 12)
    buf[:, 12:24] = scales.view(np.uint8).reshape(n, 12)
    buf[:, 24:28] = rgba_u8
    buf[:, 28:32] = rot_u8

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(buf.tobytes())

    size_mb = os.path.getsize(path) / 1024 ** 2
    print(f"Naive splat saved: {path}  ({size_mb:.0f} MB, {n:,} Gaussians)")


def export_improved_splat(
    path: str,
    pts: np.ndarray,
    cols: np.ndarray,
    confs: np.ndarray | None = None,
    k_neighbors: int = 4,
) -> None:
    """
    Improved .splat export with adaptive scale and confidence-driven opacity.

    Improvements over naive:
      - Per-point isotropic scale from mean kNN distance (not fixed 0.05).
        Gaussians in dense regions are smaller; sparse regions are larger.
      - Confidence-driven opacity: conf in [0,1] -> opacity in [120, 255].
        Low-confidence points (sky, reflections) are more transparent.

    Args:
        pts:         (N, 3) float — 3D point positions
        cols:        (N, 3) float — RGB colors in [0, 1]
        confs:       (N,)   float — per-point confidence in [0, 1], or None
        k_neighbors: number of nearest neighbours for scale estimation
    """
    from scipy.spatial import cKDTree

    n   = len(pts)
    xyz = pts.astype(np.float32)

    # Adaptive scale from local point density
    tree       = cKDTree(pts)
    dists, _   = tree.query(pts, k=k_neighbors + 1)  # +1 includes self (dist=0)
    nn_dist    = dists[:, 1:].mean(axis=1)
    scale_vals = np.clip(nn_dist, 0.005, 0.5).astype(np.float32)
    scales     = np.stack([scale_vals, scale_vals, scale_vals], axis=1)

    rgba_u8 = np.zeros((n, 4), dtype=np.uint8)
    rgba_u8[:, :3] = (np.clip(cols, 0, 1) * 255).astype(np.uint8)
    if confs is not None:
        rgba_u8[:, 3] = np.clip((120 + confs * 135), 0, 255).astype(np.uint8)
    else:
        rgba_u8[:, 3] = 200

    # Identity quaternion — isotropic spheres
    rot_u8 = np.full((n, 4), 128, dtype=np.uint8)
    rot_u8[:, 0] = 255

    buf = np.zeros((n, 32), dtype=np.uint8)
    buf[:, 0:12]  = xyz.view(np.uint8).reshape(n, 12)
    buf[:, 12:24] = scales.view(np.uint8).reshape(n, 12)
    buf[:, 24:28] = rgba_u8
    buf[:, 28:32] = rot_u8

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(buf.tobytes())

    size_mb = os.path.getsize(path) / 1024 ** 2
    print(f"Improved splat saved: {path}  ({size_mb:.0f} MB)")
    print(f"  {n:,} Gaussians | scale {scale_vals.min():.4f} - {scale_vals.max():.4f}")
    print("  View at: https://antimatter15.com/splat/")
