"""
Trajectory evaluation metrics.

Compares raw device GPS against Mapillary SfM-refined positions (computed_geometry),
and optionally compares MapAnything / VGGT-Long vision-derived poses against GPS.

Usage:
    from mapanything_pipeline.metrics import compute_all, align_to_gps
"""

import numpy as np
import pandas as pd


# ── Haversine ─────────────────────────────────────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS-84 points."""
    R = 6_371_000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def haversine_vectorised(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Vectorised haversine — operates on arrays."""
    R = 6_371_000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ── GPS vs SfM metrics (Section 5) ────────────────────────────────────────────
# These compare raw device GPS against Mapillary computed_geometry.
# MapAnything / VGGT-Long are NOT involved here.

def compute_ate(df: pd.DataFrame) -> float:
    """
    Absolute Trajectory Error (ATE) in metres — RMSE of per-frame offsets.
    Compares raw GPS (geometry) vs Mapillary SfM-refined (computed_geometry).
    """
    mask = df[["lat_raw", "lon_raw", "lat_sfm", "lon_sfm"]].notna().all(axis=1)
    d = df[mask]
    errors = haversine_vectorised(
        d["lat_raw"].values, d["lon_raw"].values,
        d["lat_sfm"].values, d["lon_sfm"].values,
    )
    return float(np.sqrt(np.mean(errors ** 2)))


def compute_rpe(df: pd.DataFrame, delta: int = 1) -> dict:
    """
    Relative Pose Error (RPE) in metres.
    Measures how consistently raw GPS replicates SfM segment lengths.

    Args:
        delta: frame gap for segment comparison (default 1 = consecutive frames)
    """
    mask = df[["lat_raw", "lon_raw", "lat_sfm", "lon_sfm"]].notna().all(axis=1)
    d = df[mask].reset_index(drop=True)
    errors = []
    for i in range(len(d) - delta):
        d_raw = haversine(d.lat_raw[i], d.lon_raw[i],
                          d.lat_raw[i + delta], d.lon_raw[i + delta])
        d_sfm = haversine(d.lat_sfm[i], d.lon_sfm[i],
                          d.lat_sfm[i + delta], d.lon_sfm[i + delta])
        errors.append(abs(d_raw - d_sfm))
    errors = np.array(errors)
    return {
        "rpe_mean_m":   float(np.mean(errors)),
        "rpe_rmse_m":   float(np.sqrt(np.mean(errors ** 2))),
        "rpe_max_m":    float(np.max(errors)),
        "rpe_median_m": float(np.median(errors)),
    }


def compute_max_drift(df: pd.DataFrame) -> float:
    """Worst single-frame deviation between raw GPS and SfM positions."""
    mask = df[["lat_raw", "lon_raw", "lat_sfm", "lon_sfm"]].notna().all(axis=1)
    d = df[mask]
    errors = haversine_vectorised(
        d["lat_raw"].values, d["lon_raw"].values,
        d["lat_sfm"].values, d["lon_sfm"].values,
    )
    return float(np.max(errors))


def compute_dtw(df: pd.DataFrame) -> float:
    """
    Dynamic Time Warping distance between raw GPS and SfM trajectories.
    Pure numpy — no external dependencies.
    Returns normalised DTW (divided by path length) in degree space.
    """
    mask = df[["lat_raw", "lon_raw", "lat_sfm", "lon_sfm"]].notna().all(axis=1)
    d = df[mask]
    A = d[["lat_raw", "lon_raw"]].values
    B = d[["lat_sfm",  "lon_sfm"]].values
    n, m = len(A), len(B)

    cost = np.full((n, m), np.inf)
    cost[0, 0] = np.linalg.norm(A[0] - B[0])
    for i in range(1, n):
        cost[i, 0] = cost[i - 1, 0] + np.linalg.norm(A[i] - B[0])
    for j in range(1, m):
        cost[0, j] = cost[0, j - 1] + np.linalg.norm(A[0] - B[j])
    for i in range(1, n):
        for j in range(1, m):
            local = np.linalg.norm(A[i] - B[j])
            cost[i, j] = local + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    return float(cost[n - 1, m - 1]) / (n + m)


def per_frame_errors(df: pd.DataFrame) -> pd.Series:
    """Per-frame haversine error (raw GPS vs SfM), indexed by image ID."""
    mask = df[["lat_raw", "lon_raw", "lat_sfm", "lon_sfm"]].notna().all(axis=1)
    d = df[mask].copy()
    d["error_m"] = haversine_vectorised(
        d["lat_raw"].values, d["lon_raw"].values,
        d["lat_sfm"].values, d["lon_sfm"].values,
    )
    return d.set_index("id")["error_m"]


def compute_all(df: pd.DataFrame, rpe_delta: int = 1) -> dict:
    """Run all GPS-vs-SfM metrics and return a summary dict."""
    return {
        "ATE (RMSE) m":     round(compute_ate(df), 4),
        "RPE mean m":       round(compute_rpe(df, rpe_delta)["rpe_mean_m"], 4),
        "RPE RMSE m":       round(compute_rpe(df, rpe_delta)["rpe_rmse_m"], 4),
        "RPE max m":        round(compute_rpe(df, rpe_delta)["rpe_max_m"],  4),
        "Max drift m":      round(compute_max_drift(df), 4),
        "DTW (normalised)": round(compute_dtw(df), 6),
        "n_frames":         int(df.notna().all(axis=1).sum()),
    }


# ── Vision pose vs GPS (Section 8) ────────────────────────────────────────────

def extract_translation(pose_4x4: np.ndarray) -> np.ndarray:
    """
    Camera centre in world coordinates from a cam-to-world 4x4 matrix.
    MapAnything and VGGT-Long both output OpenCV-convention cam2world poses.
    """
    R = pose_4x4[:3, :3]
    t = pose_4x4[:3, 3]
    return -R.T @ t


def align_to_gps(
    translations: np.ndarray,
    df: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    """
    Align vision-derived XYZ trajectory to GPS scale via a single global
    scale factor (total path length ratio). Anchors at frame 0.

    Note:
        df must be the *sampled* DataFrame (filtered to inference IDs only),
        not the full 870-row sequence DataFrame. Using the full df causes a
        stride mismatch — GPS rows 0..n won't correspond to inference frames.

    Returns:
        (aligned_xyz, scale_factor)
    """
    lat_col = "lat_sfm" if df["lat_sfm"].notna().any() else "lat_raw"
    lon_col = "lon_sfm" if df["lon_sfm"].notna().any() else "lon_raw"

    gps_pts = df[[lat_col, lon_col]].dropna().values
    n = min(len(translations), len(gps_pts))

    gps_dists = np.array([
        haversine_vectorised(
            gps_pts[i, 0], gps_pts[i, 1],
            gps_pts[i + 1, 0], gps_pts[i + 1, 1],
        )
        for i in range(n - 1)
    ])
    gps_length = float(gps_dists.sum())

    xyz_diffs  = np.diff(translations[:n], axis=0)
    xyz_length = float(np.linalg.norm(xyz_diffs, axis=1).sum())

    scale = gps_length / xyz_length if xyz_length > 0 else 1.0
    return translations[:n] * scale, scale
