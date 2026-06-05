"""
Visualization utilities for DROID-SLAM trajectory outputs.

Functions:
  plot_xz_plane     — top-down XZ trajectory (horizontal ground plane)
  plot_three_axes   — X, Y, Z components vs time
  plot_gps_fusion   — GPS vs Horn-aligned DROID-SLAM vs GTSAM-fused overlay

Usage:
    from models.droid_slam.viz import plot_xz_plane, plot_three_axes, plot_gps_fusion

    plot_xz_plane(tum_path, out_tag=out_tag)
    plot_three_axes(tum_path)
    plot_gps_fusion(gps_utm, src_aligned, fused_xy)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ── XZ plane (horizontal ground plane) ───────────────────────────────────────

def plot_xz_plane(
    tum_path: str,
    out_tag: str = "",
    save_path: str | None = None,
) -> "matplotlib.figure.Figure":
    """
    Plot the XZ plane (horizontal ground plane) of a TUM-format trajectory.

    Args:
        tum_path:  path to the .txt TUM trajectory file
        out_tag:   label for the figure title
        save_path: if given, save figure to this PNG path

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt

    data = np.loadtxt(tum_path)   # (N, 8): ts tx ty tz qx qy qz qw
    x, z = data[:, 1], data[:, 3]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x, z, linewidth=0.8, color="steelblue", label="DROID-SLAM")
    ax.scatter(x[0],  z[0],  color="green", s=60, zorder=5, label="Start")
    ax.scatter(x[-1], z[-1], color="red",   s=60, zorder=5, label="End")
    ax.set_title(f"DROID-SLAM — XZ plane  {out_tag}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"XZ plot saved: {save_path}")

    return fig


# ── Three-axis plot ───────────────────────────────────────────────────────────

def plot_three_axes(
    tum_path: str,
    out_tag: str = "",
    save_path: str | None = None,
) -> "matplotlib.figure.Figure":
    """
    Plot X, Y, Z translation components vs frame index.

    Args:
        tum_path:  path to the .txt TUM trajectory file
        out_tag:   label for the figure title
        save_path: if given, save figure to this PNG path

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt

    data = np.loadtxt(tum_path)
    ts = data[:, 0]
    x, y, z = data[:, 1], data[:, 2], data[:, 3]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].plot(x, z, linewidth=0.8, color="steelblue")
    axes[0].set_title("X-Z (top-down)"); axes[0].set_aspect("equal")

    axes[1].plot(ts, y, linewidth=0.8, color="darkorange")
    axes[1].set_title("Y (height) vs time")

    axes[2].plot(ts, np.sqrt(x**2 + z**2), linewidth=0.8, color="forestgreen")
    axes[2].set_title("Horizontal distance vs time")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"DROID-SLAM trajectory  {out_tag}")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Three-axis plot saved: {save_path}")

    return fig


# ── GPS fusion overlay ────────────────────────────────────────────────────────

def plot_gps_fusion(
    gps_utm: np.ndarray,
    src_aligned: np.ndarray,
    fused_xy: np.ndarray | None = None,
    save_path: str | None = None,
) -> "matplotlib.figure.Figure":
    """
    Overlay GPS, Horn-aligned DROID-SLAM, and (optionally) GTSAM-fused result
    in UTM coordinates.

    Args:
        gps_utm:     (N, 2) raw GPS [utm_x, utm_y]
        src_aligned: (N, 2) Horn-aligned DROID-SLAM positions
        fused_xy:    (N, 2) GTSAM-fused positions (optional)
        save_path:   if given, save figure to this PNG path

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(gps_utm[:, 0], gps_utm[:, 1],
            label="Raw GPS", linewidth=1.2, color="steelblue")
    ax.plot(src_aligned[:, 0], src_aligned[:, 1],
            label="DROID-SLAM (Horn)", linewidth=0.8, color="darkorange", linestyle="--")

    if fused_xy is not None:
        ax.plot(fused_xy[:, 0], fused_xy[:, 1],
                label="GTSAM fused", linewidth=1.0, color="forestgreen")

    ax.set_title("GPS vs DROID-SLAM Alignment")
    ax.set_xlabel("UTM X (m)")
    ax.set_ylabel("UTM Y (m)")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"GPS fusion plot saved: {save_path}")

    return fig
