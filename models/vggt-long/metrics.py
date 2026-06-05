"""
GPS reference loading and trajectory metrics for VGGT-Long runs.

Wraps the shared mapanything_pipeline.metrics functions and adds:
  - GPS reference CSV loading (trajectory.csv or gps_reference.csv)
  - TensorFlow tf.summary pose tracking (step distance, rotation, chunk overlap)
  - Matplotlib summary figure of per-step metrics

Usage:
    from models.vggt_long.metrics import (
        load_gps_reference,
        compute_pose_summary,
        log_tf_summaries,
        compute_all,
    )

    gps_pts, gps_df = load_gps_reference(traj_csv)
    summary = compute_pose_summary(cam_poses)
    log_tf_summaries(summary, tb_log_dir)
    metrics = compute_all(gps_df)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Re-export shared metric functions so callers can import from one place
from mapanything_pipeline.metrics import (
    align_to_gps,
    compute_all,
    compute_ate,
    compute_dtw,
    compute_max_drift,
    compute_rpe,
    extract_translation,
    haversine,
    haversine_vectorised,
    per_frame_errors,
)

__all__ = [
    # shared
    "haversine",
    "haversine_vectorised",
    "compute_ate",
    "compute_rpe",
    "compute_max_drift",
    "compute_dtw",
    "per_frame_errors",
    "compute_all",
    "extract_translation",
    "align_to_gps",
    # VGGT-Long specific
    "load_gps_reference",
    "compute_pose_summary",
    "log_tf_summaries",
    "plot_pose_summary",
]


# ── GPS reference ─────────────────────────────────────────────────────────────

def load_gps_reference(
    csv_path: str,
    lat_col: str | None = None,
    lon_col: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load GPS reference from trajectory.csv or gps_reference.csv.

    Prefers lat_sfm / lon_sfm (SfM-refined) when available,
    falls back to lat_raw / lon_raw.

    Args:
        csv_path: path to the CSV file
        lat_col:  override latitude column name
        lon_col:  override longitude column name

    Returns:
        gps_pts : (N, 2) float array [lat, lon]
        df      : full DataFrame with all columns
    """
    df = pd.read_csv(csv_path)

    if lat_col is None:
        lat_col = "lat_sfm" if "lat_sfm" in df.columns and df["lat_sfm"].notna().any() else "lat_raw"
    if lon_col is None:
        lon_col = "lon_sfm" if "lon_sfm" in df.columns and df["lon_sfm"].notna().any() else "lon_raw"

    df = df.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)
    gps_pts = df[[lat_col, lon_col]].values.astype(np.float64)
    print(f"GPS reference: {len(gps_pts)} points  ({lat_col}, {lon_col})")
    return gps_pts, df


# ── Per-pose summary ──────────────────────────────────────────────────────────

def _rotation_angle_deg(R: np.ndarray) -> float:
    """Angle of rotation matrix R in degrees."""
    cos_val = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def compute_pose_summary(cam_poses: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute per-frame step distances, rotation angles, and cumulative path
    from (N, 4, 4) cam-to-world matrices.

    Args:
        cam_poses: (N, 4, 4) cam-to-world float array

    Returns:
        dict with keys:
            step_dists   — (N-1,) inter-frame translation distances (metres)
            rot_angles   — (N-1,) inter-frame rotation angles (degrees)
            cum_path     — (N-1,) cumulative path length (metres)
            translations — (N, 3) camera centres in world coords
    """
    translations = np.array([
        extract_translation(p) for p in cam_poses
    ])

    step_dists = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    cum_path   = np.cumsum(step_dists)

    rot_angles = np.array([
        _rotation_angle_deg(cam_poses[i + 1, :3, :3] @ cam_poses[i, :3, :3].T)
        for i in range(len(cam_poses) - 1)
    ])

    total_m = float(cum_path[-1]) if len(cum_path) else 0.0
    print(f"Pose summary: {len(cam_poses)} frames, path={total_m:.1f} m")
    return {
        "step_dists":   step_dists,
        "rot_angles":   rot_angles,
        "cum_path":     cum_path,
        "translations": translations,
    }


# ── TensorFlow tf.summary logging ────────────────────────────────────────────

def log_tf_summaries(
    summary: dict[str, np.ndarray],
    tb_log_dir: str,
    chunk_size: int = 1,
) -> None:
    """
    Log per-step pose metrics to TensorBoard via tf.summary (CPU-only).

    Logged scalars:
        pose/step_dist_m    — inter-frame translation
        pose/rot_angle_deg  — inter-frame rotation
        pose/cum_path_m     — cumulative path length

    Args:
        summary:     output of compute_pose_summary()
        tb_log_dir:  TensorBoard log directory
        chunk_size:  step scaling (for chunk-level aggregation)
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("[metrics] tensorflow not installed — tf.summary skipped.")
        return

    os.makedirs(tb_log_dir, exist_ok=True)
    writer = tf.summary.create_file_writer(tb_log_dir)

    with writer.as_default():
        for i, (sd, ra, cp) in enumerate(zip(
            summary["step_dists"],
            summary["rot_angles"],
            summary["cum_path"],
        )):
            step = i * chunk_size
            tf.summary.scalar("pose/step_dist_m",   sd, step=step)
            tf.summary.scalar("pose/rot_angle_deg", ra, step=step)
            tf.summary.scalar("pose/cum_path_m",    cp, step=step)
        writer.flush()

    print(f"TF summaries written to {tb_log_dir}")
    print(f"  View: tensorboard --logdir {tb_log_dir}")


# ── Matplotlib summary figure ─────────────────────────────────────────────────

def plot_pose_summary(
    summary: dict[str, np.ndarray],
    save_path: str | None = None,
) -> "matplotlib.figure.Figure":
    """
    Four-panel matplotlib figure: step distance, rotation angle,
    cumulative path, and XZ top-down trajectory.

    Args:
        summary:   output of compute_pose_summary()
        save_path: if given, save figure to this PNG path

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))

    axes[0, 0].plot(summary["step_dists"], lw=0.7, color="steelblue")
    axes[0, 0].set_title("Step distance (m)")

    axes[0, 1].plot(summary["rot_angles"], lw=0.7, color="darkorange")
    axes[0, 1].set_title("Rotation angle (°)")

    axes[1, 0].plot(summary["cum_path"], lw=0.8, color="forestgreen")
    axes[1, 0].set_title("Cumulative path (m)")

    t = summary["translations"]
    colors = cm.plasma(np.linspace(0, 1, len(t)))
    axes[1, 1].scatter(t[:, 0], t[:, 2], c=colors, s=1)
    axes[1, 1].set_title("Top-down (X-Z)")
    axes[1, 1].set_aspect("equal")

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Pose summary figure saved: {save_path}")

    return fig
