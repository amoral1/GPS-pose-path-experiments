"""
GPS comparison and trajectory alignment metrics for DROID-SLAM runs.

Functions:
  load_gps_segment      — read GPS CSV, optionally convert to UTM
  compute_arc_scale     — approximate scale factor from arc-length ratio
  horn_transform        — closed-form SVD similarity transform (scale+R+t)
  apply_horn_transform  — apply Horn result to DROID-SLAM XYZ
  gtsam_gps_fusion      — GPS-only factor-graph fusion via GTSAM
  motion_onset_frame    — first frame where consistent motion is detected

Usage:
    from models.droid_slam.metrics import (
        load_gps_segment, compute_arc_scale,
        horn_transform, apply_horn_transform,
        gtsam_gps_fusion, motion_onset_frame,
    )

    gps_df    = load_gps_segment(gps_csv)
    scale     = compute_arc_scale(droid_xyz, gps_utm)
    s, R, t   = horn_transform(droid_xy, gps_utm)
    aligned   = apply_horn_transform(droid_xyz, s, R, t)
    fused_xyz = gtsam_gps_fusion(aligned, gps_utm, gps_sigma=3.0)
    onset     = motion_onset_frame(image_paths, displacement_px_thresh=25)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ── GPS reference ─────────────────────────────────────────────────────────────

def load_gps_segment(
    csv_path: str,
    convert_to_utm: bool = True,
    epsg_in: int = 4326,
    epsg_out: int = 32654,
) -> pd.DataFrame:
    """
    Load a GPS segment CSV and optionally convert lon/lat to UTM.

    Args:
        csv_path:       path to the GPS CSV
        convert_to_utm: if True, add utm_x / utm_y columns
        epsg_in:        input CRS (default WGS84)
        epsg_out:       output UTM EPSG (default 32654 = WGS84 zone 54N)

    Returns:
        DataFrame with columns including utm_x, utm_y (if converted)
    """
    df = pd.read_csv(csv_path)

    if convert_to_utm and ("utm_x" not in df.columns or "utm_y" not in df.columns):
        try:
            from pyproj import Transformer
        except ImportError:
            raise ImportError("pip install pyproj")

        transformer = Transformer.from_crs(
            f"EPSG:{epsg_in}", f"EPSG:{epsg_out}", always_xy=True
        )
        utm_x, utm_y = transformer.transform(df["longitude"].values, df["latitude"].values)
        df["utm_x"] = utm_x
        df["utm_y"] = utm_y
        print(f"Converted to UTM EPSG:{epsg_out}")

    print(f"GPS segment: {len(df)} points from {csv_path}")
    return df


# ── Scale estimation ──────────────────────────────────────────────────────────

def compute_arc_scale(
    droid_xyz: np.ndarray,
    gps_utm: np.ndarray,
) -> float:
    """
    Estimate metric scale as GPS arc length / DROID-SLAM arc length.

    Args:
        droid_xyz: (N, 2 or 3) DROID-SLAM XZ or XYZ positions
        gps_utm:   (M, 2) GPS UTM [utm_x, utm_y] positions

    Returns:
        scale factor (float)
    """
    if droid_xyz.shape[1] >= 2:
        droid_xy = droid_xyz[:, [0, 2]] if droid_xyz.shape[1] == 3 else droid_xyz[:, :2]
    else:
        droid_xy = droid_xyz

    droid_arc = float(np.sum(np.linalg.norm(np.diff(droid_xy, axis=0), axis=1)))
    gps_arc   = float(np.sum(np.linalg.norm(np.diff(gps_utm,  axis=0), axis=1)))

    if droid_arc < 1e-6:
        raise ValueError("DROID-SLAM arc length near zero — check trajectory.")

    scale = gps_arc / droid_arc
    print(f"Arc-length scale: GPS={gps_arc:.1f} m  DROID={droid_arc:.1f} m  scale={scale:.4f}")
    return scale


# ── Horn transform (SVD similarity) ──────────────────────────────────────────

def horn_transform(
    src: np.ndarray,
    tgt: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Closed-form similarity transform: tgt ≈ scale * R @ src + t

    Finds optimal scale, rotation, and translation between corresponding
    2D point sets using Horn's SVD method.

    Args:
        src: (N, 2) source points (DROID-SLAM XZ)
        tgt: (N, 2) target points (GPS UTM xy)

    Returns:
        scale : float
        R     : (2, 2) rotation matrix
        t     : (2,)   translation vector
    """
    assert src.shape == tgt.shape, "src and tgt must have the same shape"
    n = len(src)

    src_c = src - src.mean(axis=0)
    tgt_c = tgt - tgt.mean(axis=0)

    # Scale
    var_src = float((src_c ** 2).sum() / n)
    if var_src < 1e-12:
        raise ValueError("Source points have zero variance.")

    # Rotation via SVD
    H = src_c.T @ tgt_c / n
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, d])
    R = Vt.T @ D @ U.T

    scale = float((S * D.diagonal()).sum() / var_src)
    t = tgt.mean(axis=0) - scale * R @ src.mean(axis=0)

    print(f"Horn transform: scale={scale:.4f}  |t|={np.linalg.norm(t):.1f} m")
    return scale, R, t


def apply_horn_transform(
    droid_xyz: np.ndarray,
    scale: float,
    R: np.ndarray,
    t: np.ndarray,
    use_xz: bool = True,
) -> np.ndarray:
    """
    Apply a Horn similarity transform to DROID-SLAM positions.

    Args:
        droid_xyz: (N, 3) DROID-SLAM XYZ from TUM trajectory
        scale:     from horn_transform()
        R:         (2, 2) rotation matrix
        t:         (2,) translation
        use_xz:    if True, transform XZ plane (Y up); else transform XY

    Returns:
        (N, 2) aligned XY positions in GPS UTM frame
    """
    src = droid_xyz[:, [0, 2]] if use_xz else droid_xyz[:, :2]
    aligned = (scale * (R @ src.T).T) + t
    print(f"Horn transform applied to {len(aligned)} poses.")
    return aligned


# ── GTSAM GPS fusion ──────────────────────────────────────────────────────────

def gtsam_gps_fusion(
    aligned_xy: np.ndarray,
    gps_utm: np.ndarray,
    gps_sigma: float = 3.0,
) -> np.ndarray:
    """
    GPS-only GTSAM factor-graph fusion.

    Places a PriorFactor2 at every pose node using GPS UTM measurements.
    Uses Horn-aligned DROID-SLAM positions as the initial estimate.

    Args:
        aligned_xy: (N, 2) Horn-aligned DROID-SLAM XY positions
        gps_utm:    (M, 2) GPS UTM reference (resampled to match N if M≠N)
        gps_sigma:  GPS noise sigma in metres (fixed urban prior; no HDOP available)

    Returns:
        (N, 2) fused XY positions
    """
    try:
        import gtsam
    except ImportError:
        raise ImportError("pip install gtsam")

    n = min(len(aligned_xy), len(gps_utm))
    src  = aligned_xy[:n]
    tgt  = gps_utm[:n]

    graph  = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    noise  = gtsam.noiseModel.Isotropic.Sigma(2, gps_sigma)

    for i in range(n):
        key   = gtsam.symbol("x", i)
        prior = gtsam.Point2(tgt[i, 0], tgt[i, 1])
        graph.add(gtsam.PriorFactorPoint2(key, prior, noise))
        values.insert(key, gtsam.Point2(src[i, 0], src[i, 1]))

    params = gtsam.LevenbergMarquardtParams()
    result = gtsam.LevenbergMarquardtOptimizer(graph, values, params).optimize()

    fused = np.array([[result.atPoint2(gtsam.symbol("x", i))[0],
                       result.atPoint2(gtsam.symbol("x", i))[1]]
                      for i in range(n)])

    print(f"GTSAM GPS fusion complete: {n} poses  sigma={gps_sigma} m")
    return fused


# ── Motion onset detection ────────────────────────────────────────────────────

def motion_onset_frame(
    image_paths: list[str],
    min_px: float = 25.0,
    max_px: float = 150.0,
    window: int = 5,
) -> int:
    """
    Find the first frame index where consistent inter-frame optical flow
    is within the valid range [min_px, max_px] for DROID-SLAM initialisation.

    Args:
        image_paths: ordered list of image paths
        min_px:      minimum mean displacement (below → too little parallax)
        max_px:      maximum mean displacement (above → insufficient match overlap)
        window:      number of consecutive frames that must satisfy the condition

    Returns:
        Frame index of motion onset, or 0 if not determinable
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("pip install opencv-python")

    n = len(image_paths)
    consecutive = 0

    for i in range(1, n):
        prev = cv2.imread(image_paths[i - 1], cv2.IMREAD_GRAYSCALE)
        curr = cv2.imread(image_paths[i],     cv2.IMREAD_GRAYSCALE)
        if prev is None or curr is None:
            continue

        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mean_disp = float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean())

        if min_px <= mean_disp <= max_px:
            consecutive += 1
            if consecutive >= window:
                onset = i - window + 1
                print(f"Motion onset: frame {onset}  (disp={mean_disp:.1f} px)")
                return onset
        else:
            consecutive = 0

    print("Motion onset not found — check SKIP or sequence content.")
    return 0
