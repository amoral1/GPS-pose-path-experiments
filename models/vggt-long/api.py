"""
VGGT-Long pipeline setup and inference helpers.

Handles:
  - Environment / dependency installation (Colab)
  - Model clone and weight download
  - TensorFlow CPU-only configuration (frees ~2 GB VRAM for inference)
  - Google Drive mount and sequence directory resolution
  - VGGTLongTracked inference runner
  - Checkpoint save / load (chunk_records between sessions)
  - Pose output loading from camera_poses.txt / intrinsic.txt

Usage (Colab):
    from models.vggt_long.api import (
        setup_environment, clone_and_download, configure_tf_cpu,
        mount_drive, build_image_list, run_inference,
        save_checkpoint, load_checkpoint, load_poses,
    )

    setup_environment()
    clone_and_download(vggt_long_dir, weights_dir)
    configure_tf_cpu()
    seq_dir, output_dir = mount_drive(drive_root, seq_name, seq_id)
    image_paths = build_image_list(seq_dir)
    chunk_records = run_inference(image_paths, vggt_long_dir, cfg, output_dir)
    save_checkpoint(chunk_records, output_dir)
    cam_poses, cam_K = load_poses(output_dir)
"""

from __future__ import annotations

import gc
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


# ── Environment setup ─────────────────────────────────────────────────────────

VGGT_LONG_PACKAGES = [
    "Pillow",
    "huggingface_hub",
    "einops",
    "safetensors",
    "numba==0.61.2",
    "onnxruntime",
    "pyparsing",
    "importlib_metadata",
    "pypose",
    "trimesh",
    "open3d",
    "pyproj",
]


def setup_environment(extra_packages: list[str] | None = None) -> None:
    """
    Install VGGT-Long Python dependencies.
    Safe to re-run — pip skips already-installed packages.

    Args:
        extra_packages: additional package specs to install alongside defaults
    """
    packages = VGGT_LONG_PACKAGES + (extra_packages or [])
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q"] + packages,
        check=True,
    )
    print(f"Installed {len(packages)} packages.")


# ── Clone & weights ───────────────────────────────────────────────────────────

VGGT_LONG_REPO = "https://github.com/DengKaiHui/VGGT-Long.git"

WEIGHT_FILES = {
    "PRTM/model.safetensors": (
        "DengKaiHui/VGGT-Long",
        "PRTM/model.safetensors",
    ),
    "VGGTLong/model.safetensors": (
        "DengKaiHui/VGGT-Long",
        "VGGTLong/model.safetensors",
    ),
}


def clone_and_download(
    vggt_long_dir: str,
    weights_dir: str | None = None,
) -> None:
    """
    Clone the VGGT-Long repo and download model weights via huggingface_hub.

    Args:
        vggt_long_dir: target directory for the repo clone
        weights_dir:   where to write weights (default: <vggt_long_dir>/weights)
    """
    vggt_long_dir = Path(vggt_long_dir)
    weights_dir = Path(weights_dir) if weights_dir else vggt_long_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    if not vggt_long_dir.is_dir():
        subprocess.run(
            ["git", "clone", "--depth", "1", VGGT_LONG_REPO, str(vggt_long_dir)],
            check=True,
        )
        print(f"Cloned VGGT-Long → {vggt_long_dir}")
    else:
        print(f"VGGT-Long already present: {vggt_long_dir}")

    try:
        from huggingface_hub import hf_hub_download

        for local_rel, (repo_id, filename) in WEIGHT_FILES.items():
            local_path = weights_dir / local_rel
            if local_path.exists():
                print(f"  weights present: {local_rel}")
                continue
            local_path.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(weights_dir),
            )
            print(f"  downloaded: {local_rel}")
    except ImportError:
        print("[api] huggingface_hub not installed — install with pip install huggingface_hub")


# ── TensorFlow CPU configuration ──────────────────────────────────────────────

def configure_tf_cpu() -> None:
    """
    Force TensorFlow to run on CPU only, freeing ~2 GB VRAM for VGGT-Long.
    Must be called BEFORE any TF imports in the session.
    """
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        print("TF configured: CPU-only (GPU reserved for VGGT-Long).")
    except ImportError:
        print("[api] tensorflow not installed — skipping CPU config.")


# ── Drive mount & sequence resolution ────────────────────────────────────────

def mount_drive(
    drive_root: str,
    seq_name: str,
    seq_id: str,
    force_remount: bool = True,
) -> tuple[str, str]:
    """
    Mount Google Drive and resolve sequence + output directories.

    Args:
        drive_root:    Drive path to Mapillary sequences root
        seq_name:      short name, e.g. "seq_B"
        seq_id:        full Mapillary sequence folder name,
                       e.g. "seq_B_rO5jKtQfFyvpEqGHYPLAb2"
        force_remount: re-authenticate if session expired

    Returns:
        (seq_dir, output_dir) absolute paths
    """
    try:
        from google.colab import drive as _drive
        _drive.mount("/content/drive", force_remount=force_remount)
    except ImportError:
        print("[api] Not running in Colab — skipping Drive mount.")

    seq_dir    = os.path.join(drive_root, seq_id)
    output_dir = os.path.join(drive_root, "vggt_long_outputs", seq_id)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Sequence : {seq_dir}")
    print(f"Outputs  : {output_dir}")
    return seq_dir, output_dir


# ── Image list ────────────────────────────────────────────────────────────────

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def build_image_list(image_dir: str) -> list[Path]:
    """
    Return sorted list of image paths from image_dir.

    Args:
        image_dir: directory containing image files

    Returns:
        Sorted list of Path objects for each valid image
    """
    paths = sorted(
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in VALID_EXTS
    )
    assert paths, f"No images found in {image_dir}"
    print(f"Found {len(paths)} images in {image_dir}")
    return paths


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    image_paths: list[Path],
    vggt_long_dir: str,
    cfg: Any,
    output_dir: str,
    chunk_size: int | None = None,
) -> list[dict]:
    """
    Run VGGTLongTracked inference over image_paths.

    chunk_records entries contain at minimum:
        world_points, images, depth, extrinsics (per chunk)

    Args:
        image_paths:   ordered list from build_image_list()
        vggt_long_dir: path to the cloned VGGT-Long repo
        cfg:           OmegaConf config loaded from base_config.yaml
        output_dir:    where chunk .npy files and pose outputs are saved
        chunk_size:    override auto-tuned chunk size (None = auto)

    Returns:
        list of chunk record dicts
    """
    import torch

    if str(vggt_long_dir) not in sys.path:
        sys.path.insert(0, str(vggt_long_dir))

    from vggt_long import VGGT_Long  # type: ignore

    if chunk_size is None:
        free_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        chunk_size = 60 if free_gb >= 16 else 40 if free_gb >= 11 else 30
        print(f"Auto chunk_size={chunk_size}  ({free_gb:.1f} GB VRAM detected)")

    class VGGTLongTracked(VGGT_Long):
        """Subclass that saves each chunk to .npy and accumulates chunk_records."""

        chunk_records: list[dict] = []

        def process_chunk(self, chunk_imgs, chunk_idx, *args, **kwargs):
            result = super().process_chunk(chunk_imgs, chunk_idx, *args, **kwargs)
            if result is not None:
                npy_path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.npy")
                np.save(npy_path, result, allow_pickle=True)
                loaded = np.load(npy_path, allow_pickle=True).item()
                VGGTLongTracked.chunk_records.append(loaded)
            return result

    VGGTLongTracked.chunk_records = []

    pipeline = VGGTLongTracked(cfg)
    pipeline.run(image_paths, chunk_size=chunk_size, output_dir=output_dir)

    gc.collect()
    if hasattr(__builtins__, "__dict__"):
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"Inference complete. {len(VGGTLongTracked.chunk_records)} chunks.")
    return VGGTLongTracked.chunk_records


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(chunk_records: list[dict], output_dir: str) -> str:
    """
    Pickle chunk_records to disk so the session can resume after a Colab disconnect.

    Args:
        chunk_records: list from run_inference()
        output_dir:    directory to write checkpoint

    Returns:
        Path to the saved checkpoint file
    """
    ckpt_dir = os.path.join(output_dir, "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, "chunk_records.pkl")
    with open(path, "wb") as f:
        pickle.dump(chunk_records, f)
    size_mb = os.path.getsize(path) / 1024 ** 2
    print(f"Checkpoint saved: {path}  ({size_mb:.0f} MB)")
    return path


def load_checkpoint(output_dir: str) -> list[dict]:
    """
    Restore chunk_records from a previously saved checkpoint.

    Args:
        output_dir: directory where checkpoint/ subfolder lives

    Returns:
        List of chunk record dicts
    """
    path = os.path.join(output_dir, "checkpoint", "chunk_records.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with open(path, "rb") as f:
        chunk_records = pickle.load(f)
    print(f"Checkpoint loaded: {len(chunk_records)} chunks from {path}")
    return chunk_records


# ── Pose loading ──────────────────────────────────────────────────────────────

def load_poses(output_dir: str) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Load camera poses and intrinsics from VGGT-Long text outputs.

    camera_poses.txt — one flattened 4×4 C2W matrix per line (16 values)
    intrinsic.txt    — fx fy cx cy per line

    Args:
        output_dir: directory containing camera_poses.txt / intrinsic.txt

    Returns:
        cam_poses : (N, 4, 4) cam-to-world float32
        cam_K     : (N, 4) [fx, fy, cx, cy] float32, or None if file absent
    """
    poses_path     = os.path.join(output_dir, "camera_poses.txt")
    intrinsic_path = os.path.join(output_dir, "intrinsic.txt")

    raw_poses = np.loadtxt(poses_path)          # (N, 16)
    cam_poses = raw_poses.reshape(-1, 4, 4).astype(np.float32)
    print(f"Loaded {len(cam_poses)} camera poses.")

    cam_K = None
    if os.path.exists(intrinsic_path):
        cam_K = np.loadtxt(intrinsic_path).astype(np.float32)  # (N, 4)
        print(f"Loaded intrinsics: {cam_K.shape}")

    return cam_poses, cam_K
