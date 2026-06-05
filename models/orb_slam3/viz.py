"""
Visualization utilities for ORB-SLAM3 KeyFrameTrajectory outputs.

Functions:
  plot_keyframe_trajectory  — XZ top-down + XY scatter of keyframe positions
  plot_gps_overlay          — GPS UTM vs aligned ORB-SLAM3 trajectory overlay
  plot_displacement         — per-keyframe displacement bar chart (parallax check)

Usage:
    from models.orb_slam3.viz import (
        plot_keyframe_trajectory, plot_gps_overlay, plot_displacement,
    )

    plot_keyframe_trajectory(traj, title="ORB-SLAM3 — s2")
    plot_gps_overlay(gps_utm, aligned_xy)
    plot_displacement(traj)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ── Keyframe trajectory ───────────────────────────────────────────────────────

def plot_keyframe_trajectory(
    traj: np.ndarray,
    title: str = "ORB-SLAM3 Keyframe Trajectory",
    save_path: str | None = None,
) -> "matplotlib.figure.Figure":
    """
    Two-panel figure: XZ top-down (ground plane) and XY lateral view,
    colored by keyframe index.

    Args:
        traj:      (N, 8) TUM KeyFrameTrajectory
        title:     figure title
        save_path: if given, save to PNG

    Returns:
        matplotlib Figure
    """
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    x, y, z = traj[:, 1], traj[:, 2], traj[:, 3]
    colors = cm.plasma(np.linspace(0, 1, len(traj)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(x, z, c=colors, s=6)
    axes[0].scatter(x[0],  z[0],  color="green", s=80, zorder=5, label="Start")
    axes[0].scatter(x[-1], z[-1], color="red",   s=80, zorder=5, label="End")
    axes[0].set_title(f"{title} — XZ (top-down)")
    axes[0].set_xlabel("X (m)"); axes[0].set_ylabel("Z (m)")
    axes[0].set_aspect("equal")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(x, y, c=colors, s=6)
    axes[1].set_title(f"{title} — XY (lateral)")
    axes[1].set_xlabel("X (m)"); axes[1].set_ylabel("Y (m)")
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Trajectory figure saved: {save_path}")

    return fig


# ── GPS overlay ───────────────────────────────────────────────────────────────

def plot_gps_overlay(
    gps_utm: np.ndarray,
    aligned_xy: np.ndarray,
    title: str = "GPS vs ORB-SLAM3",
    save_path: str | None = None,
) -> "matplotlib.figure.Figure":
    """
    Overlay GPS UTM track and Horn-aligned ORB-SLAM3 keyframe positions.

    Args:
        gps_utm:    (N, 2) GPS UTM [utm_x, utm_y]
        aligned_xy: (M, 2) aligned ORB-SLAM3 XY in UTM frame
        title:      figure title
        save_path:  if given, save to PNG

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(gps_utm[:, 0], gps_utm[:, 1],
            label="GPS (UTM)", linewidth=1.2, color="steelblue")
    ax.scatter(aligned_xy[:, 0], aligned_xy[:, 1],
               label="ORB-SLAM3 (aligned)", s=8, color="darkorange", zorder=3)

    ax.set_title(title)
    ax.set_xlabel("UTM X (m)")
    ax.set_ylabel("UTM Y (m)")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"GPS overlay saved: {save_path}")

    return fig


# ── Displacement chart ────────────────────────────────────────────────────────

def plot_displacement(
    traj: np.ndarray,
    min_px_ref: float | None = None,
    max_px_ref: float | None = None,
    save_path: str | None = None,
) -> "matplotlib.figure.Figure":
    """
    Bar chart of per-keyframe 3D displacement with optional valid-range shading.

    Args:
        traj:        (N, 8) TUM KeyFrameTrajectory
        min_px_ref:  lower bound reference line (optional)
        max_px_ref:  upper bound reference line (optional)
        save_path:   if given, save to PNG

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt

    xyz   = traj[:, 1:4]
    dists = np.linalg.norm(np.diff(xyz, axis=0), axis=1)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(dists)), dists, color="steelblue", alpha=0.7, width=1.0)

    if min_px_ref is not None:
        ax.axhline(min_px_ref, color="green", linestyle="--", label=f"min ref ({min_px_ref})")
    if max_px_ref is not None:
        ax.axhline(max_px_ref, color="red",   linestyle="--", label=f"max ref ({max_px_ref})")

    ax.set_title("ORB-SLAM3 — per-keyframe displacement (m)")
    ax.set_xlabel("Keyframe pair index")
    ax.set_ylabel("Displacement (m)")
    if min_px_ref or max_px_ref:
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Displacement chart saved: {save_path}")

    return fig
