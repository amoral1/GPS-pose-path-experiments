"""
Point cloud extraction from MapAnything / VGGT-Long inference predictions.

extract_reconstruction() is the canonical way to pull pts, colors, confidence,
poses, and intrinsics from a list of per-frame prediction dicts.

Usage:
    from mapanything_pipeline.reconstruction import extract_reconstruction, write_colored_ply
"""

import struct
import numpy as np
from pathlib import Path


def extract_reconstruction(predictions: list[dict]) -> tuple:
    """
    Merge all per-frame MapAnything predictions into unified arrays.

    Key design decisions vs naive extraction:
      - Colors from pred["img_no_norm"]: the model's internal preprocessed image,
        pixel-aligned with pts3d. Avoids JPEG re-loading and color drift.
      - Confidence mask at 15th percentile per frame (or pred["mask"] if present).
      - conf normalised per-frame to [0,1] using min/max (ptp removed in NumPy 2.0).
      - 3 empty lists initialised to avoid unpack errors on first iteration.

    Args:
        predictions: list of dicts from model.infer(), one per input frame.
                     Expected keys: pts3d, img_no_norm, conf, camera_poses,
                     and optionally mask, intrinsics / camera_intrinsics.

    Returns:
        pts    (N, 3)  float32 — 3D points in world coordinates
        cols   (N, 3)  float32 — RGB colors in [0, 1]
        confs  (N,)    float32 — per-point confidence in [0, 1]
        poses  (F, 4, 4) float64 — cam-to-world matrices, one per input frame
        cam_K  (F, 3, 3) or None — recovered pinhole intrinsics per frame
    """
    all_pts, all_cols, all_confs = [], [], []
    poses, intrinsics = [], []

    for pred in predictions:
        pts3d   = pred["pts3d"][0].cpu().numpy()             # (H, W, 3)
        img_rgb = pred["img_no_norm"][0].cpu().numpy()       # (H, W, 3) in [0,1]
        conf    = pred["conf"][0].squeeze(-1).cpu().numpy()  # (H, W)

        # Prefer model mask if available, otherwise fall back to confidence percentile
        if "mask" in pred:
            mask = pred["mask"][0, :, :, 0].cpu().numpy() > 0.5
        else:
            mask = conf > np.percentile(conf, 15)

        conf_masked = conf[mask]
        # Normalise per-frame — ptp() removed in NumPy 2.0
        conf_norm = (
            (conf_masked - conf_masked.min())
            / (conf_masked.max() - conf_masked.min() + 1e-8)
        )

        all_pts.append(pts3d[mask])
        all_cols.append(np.clip(img_rgb[mask], 0, 1))
        all_confs.append(conf_norm.astype(np.float32))
        poses.append(pred["camera_poses"][0].cpu().numpy())

        # Handle both key names seen in MapAnything versions
        K_key = "intrinsics" if "intrinsics" in pred else "camera_intrinsics"
        if K_key in pred:
            intrinsics.append(pred[K_key][0].cpu().numpy())

    return (
        np.concatenate(all_pts),
        np.concatenate(all_cols),
        np.concatenate(all_confs),
        np.array(poses),
        np.array(intrinsics) if intrinsics else None,
    )


def write_colored_ply(path: str, pts: np.ndarray, cols: np.ndarray) -> None:
    """
    Write a binary PLY file with XYZ + RGB per vertex.

    Args:
        path: output file path (.ply)
        pts:  (N, 3) float — 3D coordinates
        cols: (N, 3) float — RGB in [0, 1]
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pts  = pts.astype(np.float32)
    cols = (np.clip(cols, 0, 1) * 255).astype(np.uint8)
    n    = len(pts)

    header = (
        f"ply\n"
        f"format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        f"property float x\n"
        f"property float y\n"
        f"property float z\n"
        f"property uchar red\n"
        f"property uchar green\n"
        f"property uchar blue\n"
        f"end_header\n"
    ).encode()

    # Interleave xyz + rgb as structured array
    dt  = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                    ("r", "u1"),  ("g", "u1"),  ("b", "u1")])
    buf = np.empty(n, dtype=dt)
    buf["x"], buf["y"], buf["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    buf["r"], buf["g"], buf["b"] = cols[:, 0], cols[:, 1], cols[:, 2]

    with open(path, "wb") as f:
        f.write(header)
        f.write(buf.tobytes())

    size_mb = path.stat().st_size / 1024 ** 2
    print(f"PLY saved: {path}  ({size_mb:.1f} MB, {n:,} points)")
