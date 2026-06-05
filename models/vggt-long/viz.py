"""
Visualization utilities for VGGT-Long outputs.

Functions:
  plotly_pointcloud      — interactive 3D scatter (reuses shared logic)
  export_geojson         — write camera trajectory as GeoJSON LineString
  plot_trajectory        — matplotlib XY / XZ scatter + GPS overlay
  depth_preview          — imported from reconstruction for convenience

Usage:
    from models.vggt_long.viz import (
        plotly_pointcloud, export_geojson, plot_trajectory,
    )

    fig = plotly_pointcloud(pts, cols, title="VGGT-Long — seq_B")
    export_geojson(aligned_xyz, utm_anchor, geojson_path)
    plot_trajectory(translations, gps_pts=gps_pts, aligned_xyz=aligned_xyz)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ── Plotly 3D point cloud ─────────────────────────────────────────────────────

def plotly_pointcloud(
    pts: np.ndarray,
    cols: np.ndarray,
    max_pts: int = 30_000,
    sigma_clip: float = 3.0,
    title: str = "VGGT-Long Point Cloud",
):
    """
    Interactive Plotly 3D scatter of the reconstructed point cloud.

    Delegates to mapanything_pipeline.viz.plotly_pointcloud with a
    VGGT-Long default title — import once and reuse.

    Args:
        pts:        (N, 3) float — 3D points
        cols:       (N, 3) float — RGB in [0, 1]
        max_pts:    subsample to this many points for browser performance
        sigma_clip: remove outliers beyond this many std deviations
        title:      figure title

    Returns:
        plotly.graph_objects.Figure
    """
    from mapanything_pipeline.viz import plotly_pointcloud as _pc
    return _pc(pts, cols, max_pts=max_pts, sigma_clip=sigma_clip, title=title)


# ── GeoJSON trajectory export ─────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def export_geojson(
    aligned_xyz: np.ndarray,
    utm_anchor: tuple[float, float],
    out_path: str,
    epsg_utm: int = 32654,
) -> None:
    """
    Export the aligned VGGT-Long trajectory as a GeoJSON LineString.

    Converts UTM offsets back to WGS-84 lon/lat using pyproj.

    Args:
        aligned_xyz:  (N, 3) XYZ in the UTM anchor frame from align_to_gps()
        utm_anchor:   (utm_x, utm_y) of the GPS anchor point
        out_path:     output .geojson file path
        epsg_utm:     UTM zone EPSG code (default 32654 = Tokyo / WGS84 zone 54N)
    """
    try:
        from pyproj import Transformer
    except ImportError:
        raise ImportError("pip install pyproj")

    utm_x, utm_y = utm_anchor
    to_wgs = Transformer.from_crs(f"EPSG:{epsg_utm}", "EPSG:4326", always_xy=True)

    coords = []
    for xyz in aligned_xyz:
        lon, lat = to_wgs.transform(utm_x + xyz[0], utm_y + xyz[1])
        coords.append([lon, lat])

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords,
                },
                "properties": {"source": "vggt_long"},
            }
        ],
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(geojson, f, cls=_NumpyEncoder)
    print(f"GeoJSON saved: {out_path}  ({len(coords)} points)")


# ── Trajectory visualization ──────────────────────────────────────────────────

def plot_trajectory(
    translations: np.ndarray,
    gps_pts: np.ndarray | None = None,
    aligned_xyz: np.ndarray | None = None,
    utm_anchor: tuple[float, float] | None = None,
    epsg_utm: int = 32654,
    save_path: str | None = None,
) -> "matplotlib.figure.Figure":
    """
    Two-panel matplotlib figure:
      Left  — top-down (X-Z) scatter of VGGT-Long camera centres
      Right — GPS vs VGGT-Long overlay in lat/lon space (requires aligned_xyz)

    Args:
        translations:  (N, 3) camera centres from compute_pose_summary()
        gps_pts:       (M, 2) [lat, lon] GPS reference (optional)
        aligned_xyz:   (N, 3) Horn-aligned VGGT-Long XYZ in UTM frame (optional)
        utm_anchor:    (utm_x, utm_y) GPS anchor for re-projection (optional)
        epsg_utm:      UTM EPSG code for reprojection
        save_path:     if given, write figure to this PNG path

    Returns:
        matplotlib Figure
    """
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    n_panels = 2 if (gps_pts is not None and aligned_xyz is not None and utm_anchor is not None) else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # ── Left: top-down XZ ────────────────────────────────────────────────────
    t_col = cm.plasma(np.linspace(0, 1, len(translations)))
    axes[0].scatter(translations[:, 0], translations[:, 2], c=t_col, s=1)
    axes[0].set_title("VGGT-Long — Top-down (X-Z)")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Z (m)")

    # ── Right: GPS overlay ───────────────────────────────────────────────────
    if n_panels == 2:
        try:
            from pyproj import Transformer
            utm_x, utm_y = utm_anchor  # type: ignore
            to_wgs = Transformer.from_crs(f"EPSG:{epsg_utm}", "EPSG:4326", always_xy=True)

            v_lon = [to_wgs.transform(utm_x + xyz[0], utm_y + xyz[1])[0] for xyz in aligned_xyz]  # type: ignore
            v_lat = [to_wgs.transform(utm_x + xyz[0], utm_y + xyz[1])[1] for xyz in aligned_xyz]  # type: ignore

            axes[1].plot(gps_pts[:, 1], gps_pts[:, 0], "b.", ms=1.5, label="GPS (sfm)")  # type: ignore
            axes[1].plot(v_lon, v_lat, "r.", ms=1.5, label="VGGT-Long")
            axes[1].set_title("GPS vs VGGT-Long")
            axes[1].set_xlabel("Longitude")
            axes[1].set_ylabel("Latitude")
            axes[1].legend(fontsize=8)
            axes[1].set_aspect("equal")
        except ImportError:
            axes[1].text(0.5, 0.5, "pyproj not installed", ha="center", va="center")

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Trajectory figure saved: {save_path}")

    return fig
