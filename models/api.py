"""
DROID-SLAM pipeline setup and data I/O helpers.

Handles:
  - Sequence directory resolution and sequence config
  - rgb.txt / timestamp list generation from {timestamp_ms}_{image_id}.jpg files
  - Intrinsics loading (from ORB-SLAM3 settings YAML or manual)
  - TUM-format trajectory load and save
  - Per-frame outlier detection and optional interpolation

DROID-SLAM trajectory format (TUM):
    timestamp tx ty tz qx qy qz qw  (one row per frame)

Usage:
    from models.droid_slam.api import (
        resolve_sequence, build_image_list, load_intrinsics,
        load_tum_trajectory, save_tum_trajectory,
        detect_outliers, interpolate_outliers,
    )

    seq_dir, img_dir, out_tag = resolve_sequence(project_root, "s1_segment")
    timestamps, image_paths = build_image_list(img_dir, skip=1, frame_start=0, frame_end=-1)
    fx, fy, cx, cy, w, h = load_intrinsics()
    traj = load_tum_trajectory(tum_path)
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np


# ── Sequence registry ─────────────────────────────────────────────────────────

SEQUENCES: dict[str, dict] = {
    "s1_segment": {
        "subdir":      "jn2K5gYsXOk46FtZrWAwMH/segments/phase1_homogeneous",
        "description": "S1 phase1_homogeneous — 74 frames, dense urban",
        "skip_default": 1,
    },
    "s1_full": {
        "subdir":      "jn2K5gYsXOk46FtZrWAwMH",
        "description": "S1 full — 4571 frames, Tokyo urban",
        "skip_default": 1,
    },
    "s2": {
        "subdir":      "rO5jKtQfFyvpEqGHYPLAb2",
        "description": "S2 — seq_B, Mapillary sequence",
        "skip_default": 1,
    },
}

# Default Mapillary intrinsics derived from EXIF (2048×1536, verified)
DEFAULT_INTRINSICS = dict(fx=1090.7, fy=1090.7, cx=1024.0, cy=768.0, w=2048, h=1536)


def resolve_sequence(
    project_root: str,
    sequence: str = "s1_segment",
    frame_start: int = 0,
    frame_end: int = -1,
) -> tuple[str, str, str]:
    """
    Resolve sequence directory, image directory, and output tag.

    Args:
        project_root: Drive root (e.g. /content/drive/MyDrive/P1)
        sequence:     one of "s1_segment", "s1_full", "s2"
        frame_start:  first frame index (s1_full windowing)
        frame_end:    last frame index  (-1 = all)

    Returns:
        (seq_dir, img_dir, out_tag) absolute paths + tag string
    """
    if sequence not in SEQUENCES:
        raise ValueError(f"Unknown sequence '{sequence}'. Choose from: {list(SEQUENCES)}")

    cfg     = SEQUENCES[sequence]
    seq_dir = os.path.join(project_root, "data", cfg["subdir"])
    img_dir = os.path.join(seq_dir, "images")
    tag_parts = [sequence, f"{frame_start}-{frame_end}"]
    out_tag = "_".join(tag_parts)
    print(f"Sequence: {sequence}  ({cfg['description']})")
    print(f"  Images: {img_dir}")
    print(f"  Output tag: {out_tag}")
    return seq_dir, img_dir, out_tag


# ── Image list & timestamps ───────────────────────────────────────────────────

_TS_PATTERN = re.compile(r'^(\d+)_')


def build_image_list(
    img_dir: str,
    skip: int = 1,
    frame_start: int = 0,
    frame_end: int = -1,
    extensions: set[str] | None = None,
) -> tuple[list[float], list[str]]:
    """
    Collect image paths from img_dir in timestamp order, applying
    SKIP stride and FRAME_START/END window.

    File naming convention: {timestamp_ms}_{image_id}.jpg

    Args:
        img_dir:     directory containing images
        skip:        frame stride (1 = no subsampling)
        frame_start: first frame index
        frame_end:   last frame index (-1 = all)
        extensions:  allowed file extensions (default: jpg/jpeg/png)

    Returns:
        timestamps:   list of timestamps in seconds (float)
        image_paths:  corresponding absolute image paths
    """
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png"}

    entries = []
    for fname in os.listdir(img_dir):
        if Path(fname).suffix.lower() not in extensions:
            continue
        m = _TS_PATTERN.match(fname)
        if m:
            entries.append((int(m.group(1)), os.path.join(img_dir, fname)))
    entries.sort(key=lambda x: x[0])

    if not entries:
        raise FileNotFoundError(f"No images found in {img_dir}")

    if frame_end == -1:
        frame_end = len(entries)
    entries = entries[frame_start:frame_end:skip]

    timestamps   = [ts / 1000.0 for ts, _ in entries]   # ms → seconds
    image_paths  = [p for _, p in entries]

    print(f"Image list: {len(image_paths)} frames (skip={skip}, [{frame_start}:{frame_end}])")
    return timestamps, image_paths


def write_rgb_txt(
    timestamps: list[float],
    image_paths: list[str],
    seq_dir: str,
    filename: str = "rgb.txt",
) -> str:
    """
    Write a TUM-compatible rgb.txt file.

    Format:
        # timestamp filename
        {ts:.6f} images/{fname}

    Args:
        timestamps:  seconds timestamps
        image_paths: absolute paths to images
        seq_dir:     sequence root directory
        filename:    output filename

    Returns:
        Absolute path to the written file
    """
    out_path = os.path.join(seq_dir, filename)
    with open(out_path, "w") as f:
        f.write("# TUM RGB-D format\n")
        f.write("# timestamp filename\n")
        f.write("# generated by models.droid_slam.api\n")
        for ts, p in zip(timestamps, image_paths):
            rel = os.path.relpath(p, seq_dir)
            f.write(f"{ts:.6f} {rel}\n")
    print(f"rgb.txt written: {out_path}  ({len(timestamps)} entries)")
    return out_path


# ── Intrinsics ────────────────────────────────────────────────────────────────

def load_intrinsics(
    yaml_path: str | None = None,
) -> tuple[float, float, float, float, int, int]:
    """
    Load camera intrinsics for DROID-SLAM.

    If yaml_path is given, reads fx/fy/cx/cy/width/height from an ORB-SLAM3
    settings YAML. Falls back to hard-coded Mapillary defaults.

    Args:
        yaml_path: path to ORB-SLAM3 .yaml settings file (optional)

    Returns:
        (fx, fy, cx, cy, width, height)
    """
    if yaml_path and os.path.exists(yaml_path):
        try:
            import yaml  # PyYAML
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
            cam = cfg.get("Camera", cfg)
            fx = float(cam.get("fx", DEFAULT_INTRINSICS["fx"]))
            fy = float(cam.get("fy", DEFAULT_INTRINSICS["fy"]))
            cx = float(cam.get("cx", DEFAULT_INTRINSICS["cx"]))
            cy = float(cam.get("cy", DEFAULT_INTRINSICS["cy"]))
            w  = int(cam.get("width",  DEFAULT_INTRINSICS["w"]))
            h  = int(cam.get("height", DEFAULT_INTRINSICS["h"]))
            print(f"Intrinsics from YAML: fx={fx} fy={fy} cx={cx} cy={cy} {w}×{h}")
            return fx, fy, cx, cy, w, h
        except Exception as e:
            print(f"[api] YAML load failed ({e}) — using defaults.")

    d = DEFAULT_INTRINSICS
    print(f"Intrinsics (default): fx={d['fx']} fy={d['fy']} cx={d['cx']} cy={d['cy']} {d['w']}×{d['h']}")
    return d["fx"], d["fy"], d["cx"], d["cy"], d["w"], d["h"]


# ── TUM trajectory I/O ────────────────────────────────────────────────────────

def load_tum_trajectory(path: str) -> np.ndarray:
    """
    Load a TUM-format trajectory file.

    Format: timestamp tx ty tz qx qy qz qw (8 columns)

    Args:
        path: path to the .txt trajectory file

    Returns:
        (N, 8) float64 array
    """
    traj = np.loadtxt(path)
    if traj.ndim == 1:
        traj = traj.reshape(1, -1)
    assert traj.shape[1] == 8, f"Expected 8 columns (ts tx ty tz qx qy qz qw), got {traj.shape[1]}"
    print(f"Loaded trajectory: {len(traj)} frames from {path}")
    return traj


def save_tum_trajectory(
    traj: np.ndarray,
    timestamps: list[float],
    out_path: str,
    out_tag: str = "",
) -> None:
    """
    Save trajectory in TUM format.

    Args:
        traj:       (N, 7) [tx ty tz qx qy qz qw] or (N, 8) with timestamp
        timestamps: per-frame timestamps in seconds
        out_path:   output directory (file named <out_tag>.txt)
        out_tag:    filename stem
    """
    from scipy.spatial.transform import Rotation

    os.makedirs(out_path, exist_ok=True)
    file_path = os.path.join(out_path, f"{out_tag}.txt") if out_tag else out_path

    n = min(len(traj), len(timestamps))
    with open(file_path, "w") as f:
        for i in range(n):
            ts = timestamps[i]
            row = traj[i]
            if len(row) == 16:
                # 4×4 matrix flattened
                pose = row.reshape(4, 4)
                t = pose[:3, 3]
                q = Rotation.from_matrix(pose[:3, :3]).as_quat()  # [qx qy qz qw]
            elif len(row) == 7:
                t = row[:3]
                q = row[3:]
            elif len(row) == 8:
                t = row[1:4]
                q = row[4:]
            else:
                continue
            f.write(f"{ts:.6f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                    f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")

    print(f"TUM trajectory saved: {file_path}  ({n} frames)")


# ── Outlier detection & interpolation ────────────────────────────────────────

def detect_outliers(
    traj: np.ndarray,
    threshold_factor: float = 10.0,
) -> np.ndarray:
    """
    Detect per-frame pose outliers by inter-frame displacement.

    Flags frames whose displacement exceeds threshold_factor × median displacement.

    Args:
        traj:             (N, 8) TUM trajectory
        threshold_factor: outlier threshold as multiple of median displacement

    Returns:
        Boolean mask (N,) — True where frame is an outlier
    """
    xyz = traj[:, 1:4]
    displacements = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    median_d  = np.median(displacements)
    threshold = median_d * threshold_factor
    outlier_mask = np.concatenate([[False], displacements > threshold])
    n_out = outlier_mask.sum()
    print(f"Outliers: {n_out} / {len(traj)} frames (threshold={threshold:.3f} m = {threshold_factor}× median)")
    return outlier_mask


def interpolate_outliers(
    traj: np.ndarray,
    remove_indices: list[int],
) -> np.ndarray:
    """
    Linearly interpolate over specified frame indices.

    Args:
        traj:           (N, 8) TUM trajectory
        remove_indices: list of frame indices to replace via interpolation

    Returns:
        (N, 8) trajectory with outlier frames interpolated
    """
    traj_out = traj.copy()
    for idx in sorted(remove_indices):
        if idx <= 0 or idx >= len(traj) - 1:
            print(f"  [api] Cannot interpolate boundary frame {idx} — skipping.")
            continue
        traj_out[idx, 1:] = (traj[idx - 1, 1:] + traj[idx + 1, 1:]) / 2.0
        print(f"  Interpolated frame {idx}")
    return traj_out
