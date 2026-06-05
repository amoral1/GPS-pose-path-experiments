"""
COLMAP-format export for downstream 3D Gaussian Splatting training.

Produces:
  <out_dir>/sparse/0/cameras.txt   — 1 PINHOLE camera
  <out_dir>/sparse/0/images.txt    — QW QX QY QZ TX TY TZ per frame
  <out_dir>/sparse/0/points3D.txt  — intentionally empty
  <out_dir>/images/                — renamed JPEGs

Usage:
    from mapanything_pipeline.colmap_export import export_colmap

Then train with:
    python train.py -s <out_dir> --iterations 7000
    (github.com/graphdeco-inria/gaussian-splatting)
"""

import os
import shutil
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation


def export_colmap(
    out_dir: str,
    cam_poses: np.ndarray,
    cam_K: np.ndarray | None,
    image_ids: list[str],
    img_dir: str,
    subsample: int = 1,
) -> None:
    """
    Write a COLMAP sparse reconstruction from MapAnything / VGGT-Long outputs.

    Args:
        out_dir:   destination directory (created if absent)
        cam_poses: (F, 4, 4) cam-to-world matrices from extract_reconstruction()
        cam_K:     (F, 3, 3) or (F, 4) intrinsics, or None (falls back to 60-deg FoV)
        image_ids: ordered list of Mapillary image IDs used for inference
        img_dir:   directory containing <id>.jpg source images
        subsample: export every Nth pose (1 = all frames)

    Notes:
        - striding already happened at download time (stride=5 from API)
        - subsample=1 uses all inference frames (~174)
        - COLMAP world-to-cam convention: R, t in images.txt are world-to-cam
          MapAnything outputs cam-to-world, so we invert here
    """
    out_dir = Path(out_dir)
    sparse  = out_dir / "sparse" / "0"
    imgs_out = out_dir / "images"
    sparse.mkdir(parents=True, exist_ok=True)
    imgs_out.mkdir(parents=True, exist_ok=True)

    # ── Intrinsics ────────────────────────────────────────────────────────────
    if cam_K is not None:
        K = cam_K[0]
        if K.ndim == 2 and K.shape == (3, 3):
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        else:  # (4,) vector [fx, fy, cx, cy]
            fx, fy, cx, cy = float(K[0]), float(K[1]), float(K[2]), float(K[3])
        IMG_W, IMG_H = int(round(cx * 2)), int(round(cy * 2))
        print(f"Model intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    else:
        IMG_W, IMG_H = 256, 256
        fx = fy = (IMG_W / 2) / np.tan(np.radians(30.0))  # 60-deg FoV
        cx, cy  = IMG_W / 2.0, IMG_H / 2.0
        print("No model intrinsics — assuming 60-deg FoV at 256x256")

    # ── cameras.txt ───────────────────────────────────────────────────────────
    with open(sparse / "cameras.txt", "w") as f:
        f.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[fx fy cx cy]\n")
        f.write(f"1 PINHOLE {IMG_W} {IMG_H} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    # ── images.txt ────────────────────────────────────────────────────────────
    pose_id_pairs = list(zip(cam_poses, image_ids[: len(cam_poses)]))[::subsample]

    n_copied = 0
    with open(sparse / "images.txt", "w") as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
        f.write("# (empty POINTS2D line follows each image)\n")

        for img_idx, (pose_4x4, iid) in enumerate(pose_id_pairs, start=1):
            # MapAnything outputs cam2world; COLMAP expects world2cam
            R_c2w = pose_4x4[:3, :3]
            t_c2w = pose_4x4[:3, 3]
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w

            q    = Rotation.from_matrix(R_w2c).as_quat()  # [qx, qy, qz, qw]
            qw, qx, qy, qz = float(q[3]), float(q[0]), float(q[1]), float(q[2])

            img_name = f"{img_idx:05d}.jpg"
            f.write(
                f"{img_idx} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                f"{float(t_w2c[0]):.9f} {float(t_w2c[1]):.9f} "
                f"{float(t_w2c[2]):.9f} 1 {img_name}\n"
            )
            f.write("\n")

            src = Path(img_dir) / f"{iid}.jpg"
            if src.exists():
                shutil.copy2(src, imgs_out / img_name)
                n_copied += 1

    # ── points3D.txt — empty; 3DGS trainer initialises from random Gaussians ──
    with open(sparse / "points3D.txt", "w") as f:
        f.write("# intentionally empty — 3DGS trainer uses random initialisation\n")

    print(f"\nCOLMAP export: {out_dir}/")
    print(f"  cameras.txt : 1 PINHOLE camera")
    print(f"  images.txt  : {len(pose_id_pairs)} frames  (subsample={subsample})")
    print(f"  images/     : {n_copied} JPEGs copied")
    print()
    print("Train 3DGS:")
    print("  git clone https://github.com/graphdeco-inria/gaussian-splatting")
    print("  pip install -r requirements.txt")
    print(f"  python train.py -s {out_dir} --iterations 7000")
