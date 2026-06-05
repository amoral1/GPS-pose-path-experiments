"""
Mapillary Graph API — sequence fetching and trajectory parsing.

Usage:
    from mapanything_pipeline.api import fetch_image_ids, fetch_trajectory, trajectory_to_df
"""

import time
import requests
import pandas as pd

FIELDS = (
    "id,geometry,computed_geometry,"
    "captured_at,compass_angle,computed_compass_angle,altitude"
)


def fetch_image_ids(seq_id: str, token: str) -> list[str]:
    """Return ordered list of image IDs for a Mapillary sequence."""
    url = (
        f"https://graph.mapillary.com/image_ids"
        f"?access_token={token}&sequence_id={seq_id}"
    )
    r = requests.get(url)
    r.raise_for_status()
    return [obj["id"] for obj in r.json()["data"]]


def fetch_trajectory(
    image_ids: list[str],
    token: str,
    fields: str = FIELDS,
    sleep: float = 0.05,
) -> list[dict]:
    """
    Pull per-image metadata for trajectory reconstruction.

    Args:
        image_ids: ordered list of Mapillary image IDs
        token:     Mapillary access token (MLY_TOKEN)
        fields:    comma-separated API fields to request
        sleep:     seconds to wait between requests (rate-limit headroom)

    Returns:
        list of raw API response dicts, one per image
    """
    data = []
    for iid in image_ids:
        url = (
            f"https://graph.mapillary.com/{iid}"
            f"?access_token={token}&fields={fields}"
        )
        r = requests.get(url)
        if r.status_code == 200:
            data.append(r.json())
        else:
            print(f"  [warn] {iid}: HTTP {r.status_code}")
        time.sleep(sleep)
    return data


def trajectory_to_df(raw: list[dict]) -> pd.DataFrame:
    """
    Parse raw API records into a tidy DataFrame.

    Columns:
        id           — Mapillary image ID
        ts           — capture timestamp (UTC, datetime)
        compass_raw  — device compass bearing
        compass_sfm  — Mapillary SfM-refined bearing
        altitude     — device altitude (metres)
        lat_raw/lon_raw — device GPS (geometry field)
        lat_sfm/lon_sfm — Mapillary SfM-refined position (computed_geometry)
    """
    rows = []
    for rec in raw:
        row = {"id": rec["id"]}
        row["ts"]          = rec.get("captured_at")
        row["compass_raw"] = rec.get("compass_angle")
        row["compass_sfm"] = rec.get("computed_compass_angle")
        row["altitude"]    = rec.get("altitude")

        g = rec.get("geometry", {}).get("coordinates", [None, None])
        row["lon_raw"], row["lat_raw"] = g[0], g[1]

        cg = rec.get("computed_geometry", {}).get("coordinates", [None, None])
        row["lon_sfm"], row["lat_sfm"] = cg[0], cg[1]

        rows.append(row)

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def download_images(
    image_ids: list[str],
    token: str,
    out_dir: str,
    size: int = 256,
    stride: int = 5,
    sleep: float = 0.05,
) -> list[str]:
    """
    Download JPEG thumbnails for a strided subset of image IDs.

    Args:
        image_ids: full ordered ID list for the sequence
        token:     Mapillary access token
        out_dir:   directory to save <id>.jpg files
        size:      thumbnail size (256 or 1024)
        stride:    download every Nth frame
        sleep:     seconds between requests

    Returns:
        list of local file paths for downloaded images
    """
    import os
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = image_ids[::stride]
    saved_paths = []

    for iid in ids:
        out = out_dir / f"{iid}.jpg"
        if out.exists():
            saved_paths.append(str(out))
            continue
        url = (
            f"https://graph.mapillary.com/{iid}"
            f"?access_token={token}&fields=thumb_{size}_url"
        )
        r = requests.get(url)
        if r.status_code != 200:
            print(f"  [warn] {iid}: HTTP {r.status_code}")
            continue
        img_url = r.json().get(f"thumb_{size}_url")
        if not img_url:
            continue
        img_r = requests.get(img_url)
        if img_r.status_code == 200:
            out.write_bytes(img_r.content)
            saved_paths.append(str(out))
        time.sleep(sleep)

    print(f"Downloaded {len(saved_paths)} images to {out_dir}")
    return saved_paths
