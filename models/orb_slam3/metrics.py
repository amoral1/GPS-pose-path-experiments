"""
Trajectory metrics for ORB-SLAM3 KeyFrameTrajectory outputs.

Functions:
  inter_frame_displacement  — per-keyframe displacement diagnostic
  compute_trajectory_stats  — path length, mean step, max step
  align_keyframes_to_gps    — Horn-align ORB-SLAM3 to GPS UTM
  compute_ate_tum           — ATE against a GPS/GT reference

ORB-SLAM3 outputs only keyframes (not every input frame), so alignment
uses keyframe-to-GPS nearest-timestamp matching.

Usage:
    from models.orb_slam3.metrics import (
        inter_frame_displacement,
        compute_trajectory_stats,
        align_keyframes_to_gps,
        compute_ate_tum,
    )

    disp = inter_frame_displacement(traj)
    stats = compute_trajectory_stats(traj)
    aligned_xy, scale, R, t = align_keyframes_to_gps(traj, gps_df)
    ate_m = compute_ate_tum(aligned_xy, gps_utm_matched)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Inter-frame displacement ──────────────────────────────────────────────────

def inter_frame_displacement(
    traj: np.ndarray,
) -> dict[str, float]:
    """
    Compute per-keyframe 3D displacement statistics.

    Args:
        traj: (N, 8) TUM trajectory [ts tx ty tz qx qy qz qw]

    Returns:
        dict with keys: mean_m, median_m, max_m, total_m, n_keyframes
    """
    xyz = traj[:, 1:4]
    dists = np.linalg.norm(np.diff(xyz, axis=0), axis=1)

    stats = {
        "mean_m":      float(dists.mean()),
        "median_m":    float(np.median(dists)),
        "max_m":       float(dists.max()),
        "total_m":     float(dists.sum()),
        "n_keyframes": len(traj),
    }
    print(
        f"Displacement: mean={stats['mean_m']:.3f} m  "
        f"max={stats['max_m']:.3f} m  "
        f"total={stats['total_m']:.1f} m  "
        f"({stats['n_keyframes']} keyframes)"
    )
    return stats


# ── Trajectory statistics ─────────────────────────────────────────────────────

def compute_trajectory_stats(traj: np.ndarray) -> dict[str, float]:
    """
    Path length, mean and max step size, and duration from TUM trajectory.

    Args:
        traj: (N, 8) TUM trajectory

    Returns:
        dict with keys: path_m, mean_step_m, max_step_m, duration_s, n_keyframes
    """
    xyz  = traj[:, 1:4]
    ts   = traj[:, 0]
    dists = np.linalg.norm(np.diff(xyz, axis=0), axis=1)

    stats = {
        "path_m":       float(dists.sum()),
        "mean_step_m":  float(dists.mean()),
        "max_step_m":   float(dists.max()),
        "duration_s":   float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0,
        "n_keyframes":  len(traj),
    }
    print(
        f"Trajectory: path={stats['path_m']:.1f} m  "
        f"dur={stats['duration_s']:.1f} s  "
        f"n={stats['n_keyframes']}"
    )
    return stats


# ── GPS alignment ─────────────────────────────────────────────────────────────

def align_keyframes_to_gps(
    traj: np.ndarray,
    gps_df: pd.DataFrame,
    utm_x_col: str = "utm_x",
    utm_y_col: str = "utm_y",
    ts_col: str = "captured_at_ms",
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Horn-align ORB-SLAM3 XZ positions to GPS UTM using nearest-timestamp matching.

    Args:
        traj:      (N, 8) TUM KeyFrameTrajectory
        gps_df:    DataFrame with utm_x, utm_y, and a timestamp column
        utm_x_col: UTM X column name in gps_df
        utm_y_col: UTM Y column name in gps_df
        ts_col:    timestamp column (ms) — converted to seconds for matching

    Returns:
        aligned_xy : (N, 2) ORB-SLAM3 positions in GPS UTM frame
        scale      : Horn similarity scale
        R          : (2, 2) rotation matrix
        t          : (2,) translation vector
    """
    from models.droid_slam.metrics import horn_transform, apply_horn_transform

    kf_ts  = traj[:, 0]          # seconds
    gps_ts = gps_df[ts_col].values / 1000.0  # ms → seconds

    # Match each keyframe to nearest GPS timestamp
    matched_gps = []
    for ts in kf_ts:
        idx = int(np.argmin(np.abs(gps_ts - ts)))
        matched_gps.append([gps_df.iloc[idx][utm_x_col], gps_df.iloc[idx][utm_y_col]])

    gps_utm = np.array(matched_gps)
    orb_xz  = traj[:, [1, 3]]   # X, Z (horizontal plane)

    scale, R, t = horn_transform(orb_xz, gps_utm)
    aligned_xy  = apply_horn_transform(
        np.column_stack([traj[:, 1], traj[:, 3]]),  # (N, 2) XZ
        scale, R, t, use_xz=False,
    )
    return aligned_xy, scale, R, t


# ── ATE computation ───────────────────────────────────────────────────────────

def compute_ate_tum(
    estimated_xy: np.ndarray,
    reference_xy: np.ndarray,
) -> float:
    """
    Compute Absolute Trajectory Error (RMSE) between estimated and reference XY.

    Args:
        estimated_xy: (N, 2) estimated positions (already aligned)
        reference_xy: (N, 2) reference GPS UTM positions

    Returns:
        ATE in metres (RMSE)
    """
    n = min(len(estimated_xy), len(reference_xy))
    errors = np.linalg.norm(estimated_xy[:n] - reference_xy[:n], axis=1)
    ate_m  = float(np.sqrt((errors ** 2).mean()))
    print(f"ATE (RMSE): {ate_m:.3f} m  ({n} keyframes)")
    return ate_m
