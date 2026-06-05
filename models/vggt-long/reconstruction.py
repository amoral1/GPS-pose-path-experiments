"""
3D reconstruction utilities for VGGT-Long outputs.

Functions:
  aggregate_world_points  — merge world_points/images/depth across chunk_records
  write_ply               — write Open3D colored point cloud to .ply
  depth_preview           — matplotlib preview of per-frame depth maps
  tsdf_fusion             — optional Open3D TSDF mesh fusion from depth records

VGGT-Long chunk_records schema (per chunk):
    world_points  : (H, W, 3) or (N, 3) float — 3D points in world frame
    images        : (H, W, 3) float [0, 1]     — RGB colors aligned to world_points
    depth         : (H, W)    float             — metric depth (metres)
    extrinsics    : (4, 4) or (N, 4, 4) float  — cam-to-world poses

Usage:
    from models.vggt_long.reconstruction import (
        aggregate_world_points, write_ply, depth_preview, tsdf_fusion,
    )

    pts, cols, confs = aggregate_world_points(chunk_records, conf_thr=0.65)
    write_ply(ply_path, pts, cols)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ── Point cloud aggregation ───────────────────────────────────────────────────

def aggregate_world_points(
    chunk_records: list[dict],
    conf_thr: float = 0.65,
    sample_ratio: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge world_points, images, and confidence across all chunk_records.

    Points below conf_thr are discarded. sample_ratio sub-samples the remainder
    for memory management.

    Args:
        chunk_records: list of chunk dicts from run_inference() or load_checkpoint()
        conf_thr:      minimum confidence to keep (0–1 scale)
        sample_ratio:  fraction of passing points to retain (1.0 = all)

    Returns:
        pts   : (N, 3) float32 — world-frame 3D points
        cols  : (N, 3) float32 — RGB in [0, 1]
        confs : (N,)   float32 — raw confidence values
    """
    all_pts, all_rgb, all_conf = [], [], []

    for i, rec in enumerate(chunk_records):
        wp  = np.asarray(rec["world_points"]).reshape(-1, 3).astype(np.float32)
        rgb = np.asarray(rec["images"]).reshape(-1, 3).astype(np.float32)

        # confidence may live under "conf" or "confidence"
        conf_key = "conf" if "conf" in rec else "confidence"
        if conf_key in rec:
            c = np.asarray(rec[conf_key]).reshape(-1).astype(np.float32)
        else:
            c = np.ones(len(wp), dtype=np.float32)

        mask = c >= conf_thr
        if not mask.any():
            continue

        if sample_ratio < 1.0:
            idx = np.where(mask)[0]
            n_keep = max(1, int(len(idx) * sample_ratio))
            chosen = np.random.choice(idx, n_keep, replace=False)
            mask = np.zeros(len(wp), dtype=bool)
            mask[chosen] = True

        all_pts.append(wp[mask])
        all_rgb.append(np.clip(rgb[mask], 0.0, 1.0))
        all_conf.append(c[mask])

    if not all_pts:
        raise ValueError("No points survived confidence threshold — lower conf_thr.")

    pts   = np.concatenate(all_pts)
    cols  = np.concatenate(all_rgb)
    confs = np.concatenate(all_conf)

    print(
        f"Aggregated {len(pts):,} points from {len(chunk_records)} chunks "
        f"(conf≥{conf_thr}, sample={sample_ratio:.2f})"
    )
    return pts, cols, confs


# ── PLY export ────────────────────────────────────────────────────────────────

def write_ply(path: str, pts: np.ndarray, cols: np.ndarray) -> None:
    """
    Write a colored point cloud to .ply using Open3D.

    Args:
        path: output file path (.ply)
        pts:  (N, 3) float — XYZ positions
        cols: (N, 3) float — RGB in [0, 1]
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("pip install open3d")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(cols, 0, 1).astype(np.float64))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)
    size_mb = Path(path).stat().st_size / 1024 ** 2
    print(f"PLY saved: {path}  ({len(pts):,} pts  {size_mb:.0f} MB)")


# ── Depth preview ─────────────────────────────────────────────────────────────

def depth_preview(
    chunk_records: list[dict],
    n: int = 4,
    save_path: str | None = None,
) -> "matplotlib.figure.Figure | None":
    """
    Show a 1×n matplotlib grid of per-frame depth maps from the first n chunks.

    Args:
        chunk_records: list of chunk dicts with a "depth" key
        n:             number of depth frames to display
        save_path:     if given, write figure to this PNG path

    Returns:
        matplotlib Figure, or None if no depth records found
    """
    import matplotlib.pyplot as plt

    depth_records = [r for r in chunk_records if "depth" in r]
    if not depth_records:
        print("[reconstruction] No depth records found in chunk_records.")
        return None

    n_show = min(n, len(depth_records))
    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
    if n_show == 1:
        axes = [axes]

    for ax, rec in zip(axes, depth_records[:n_show]):
        d = np.asarray(rec["depth"])
        if d.ndim == 3:
            d = d[0]
        vmin, vmax = np.percentile(d, 2), np.percentile(d, 98)
        ax.imshow(d, cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(f"depth (p2={vmin:.1f}m, p98={vmax:.1f}m)")
        ax.axis("off")

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Depth preview saved: {save_path}")

    return fig


# ── TSDF fusion ───────────────────────────────────────────────────────────────

def tsdf_fusion(
    depth_records: list[dict],
    cam_poses: np.ndarray,
    cam_K: np.ndarray | None,
    voxel_m: float = 0.05,
    trunc_m: float = 0.20,
    img_w: int = 256,
    img_h: int = 256,
) -> "open3d.geometry.TriangleMesh":
    """
    Fuse per-frame depth maps into a mesh using Open3D ScalableTSDFVolume.

    Args:
        depth_records: chunk_records entries that contain a "depth" key
        cam_poses:     (N, 4, 4) cam-to-world matrices from load_poses()
        cam_K:         (N, 4) [fx, fy, cx, cy] or None (falls back to 60-deg FoV)
        voxel_m:       TSDF voxel size in metres
        trunc_m:       truncation distance in metres
        img_w, img_h:  image dimensions for the TSDF intrinsic

    Returns:
        Open3D TriangleMesh
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("pip install open3d")

    if cam_K is not None and cam_K.ndim == 2 and cam_K.shape[-1] >= 4:
        fx, fy, cx, cy = cam_K[0, :4]
        img_w = int(round(cx * 2))
        img_h = int(round(cy * 2))
    else:
        fx = fy = (img_w / 2) / np.tan(np.radians(30.0))
        cx, cy = img_w / 2.0, img_h / 2.0

    intrinsic = o3d.camera.PinholeCameraIntrinsic(img_w, img_h, fx, fy, cx, cy)
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_m,
        sdf_trunc=trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for i, rec in enumerate(depth_records):
        if i >= len(cam_poses):
            break

        depth = np.asarray(rec["depth"])
        if depth.ndim == 3:
            depth = depth[0]

        rgb = np.asarray(rec.get("images", np.zeros((*depth.shape, 3))))
        if rgb.ndim == 3 and rgb.shape[-1] == 3:
            rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        else:
            rgb_u8 = np.zeros((*depth.shape, 3), dtype=np.uint8)

        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
        color_o3d = o3d.geometry.Image(rgb_u8)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0,
            depth_trunc=10.0,
            convert_rgb_to_intensity=False,
        )

        extrinsic = np.linalg.inv(cam_poses[i])  # world-to-cam for TSDF
        vol.integrate(rgbd, intrinsic, extrinsic)

    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    print(f"TSDF mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} faces")
    return mesh
