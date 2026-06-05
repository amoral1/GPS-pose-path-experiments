"""
Visualization utilities.

  bev_drift_image  — render GPS vs vision trajectory as a color-coded BEV image
  plotly_pointcloud — interactive 3D scatter of the reconstructed point cloud
  serve_splat      — cloudflared tunnel to serve a .splat file for browser viewing

Usage:
    from mapanything_pipeline.viz import bev_drift_image, plotly_pointcloud, serve_splat
"""

import re
import time
import threading
import subprocess
import http.server
import numpy as np


# ── BEV drift visualization ───────────────────────────────────────────────────

def bev_drift_image(
    gps_pts: np.ndarray,
    vision_pts: np.ndarray,
    offsets: np.ndarray,
    out_path: str,
    canvas_size: int = 800,
    drift_thresh_m: float = 5.0,
) -> None:
    """
    Render GPS vs vision-derived trajectory as a BEV image, color-coded by
    per-frame gps_offset_m. Suitable as input for YOLO drift detection.

    Args:
        gps_pts:      (N, 2) float — [lat, lon] from computed_geometry
        vision_pts:   (N, 2) float — [lat, lon] from MapAnything / VGGT-Long
        offsets:      (N,)   float — gps_offset_m per frame
        out_path:     path to write PNG
        canvas_size:  image width and height in pixels
        drift_thresh_m: offset (metres) at which color saturates to red

    Output colors:
        White line  — GPS reference track
        Green->Red  — vision track, color encodes offset magnitude
        Orange dot  — first frame where offset exceeds drift_thresh_m (cold-start onset)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("pip install opencv-python")

    from pathlib import Path

    all_pts = np.vstack([gps_pts, vision_pts])
    lat_min, lat_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    lon_min, lon_max = all_pts[:, 1].min(), all_pts[:, 1].max()

    W = H = canvas_size
    pad = 20

    def to_px(lat, lon):
        x = int((lon - lon_min) / (lon_max - lon_min + 1e-12) * (W - 2 * pad) + pad)
        y = int((lat_max - lat) / (lat_max - lat_min + 1e-12) * (H - 2 * pad) + pad)
        return x, y

    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # GPS reference — white
    for i in range(len(gps_pts) - 1):
        cv2.line(canvas, to_px(*gps_pts[i]), to_px(*gps_pts[i + 1]), (255, 255, 255), 1)

    # Vision track — green to red by offset
    for i in range(len(vision_pts) - 1):
        norm  = float(np.clip(offsets[i] / drift_thresh_m, 0, 1))
        color = (0, int(255 * (1 - norm)), int(255 * norm))  # BGR
        cv2.line(canvas, to_px(*vision_pts[i]), to_px(*vision_pts[i + 1]), color, 2)

    # Cold-start onset marker
    onset = next((i for i, o in enumerate(offsets) if o > drift_thresh_m), None)
    if onset is not None:
        cv2.circle(canvas, to_px(*vision_pts[onset]), 8, (0, 165, 255), -1)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, canvas)
    print(f"BEV drift image saved: {out_path}")
    if onset is not None:
        print(f"  Cold-start onset at frame {onset}  ({offsets[onset]:.1f} m offset)")


# ── Plotly 3D point cloud ─────────────────────────────────────────────────────

def plotly_pointcloud(
    pts: np.ndarray,
    cols: np.ndarray,
    max_pts: int = 30_000,
    sigma_clip: float = 3.0,
    title: str = "MapAnything Point Cloud",
):
    """
    Interactive Plotly 3D scatter of the reconstructed point cloud.

    Args:
        pts:       (N, 3) float — 3D points
        cols:      (N, 3) float — RGB in [0, 1]
        max_pts:   subsample to this many points for browser performance
        sigma_clip: remove points beyond this many standard deviations
        title:     figure title

    Returns:
        plotly.graph_objects.Figure
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("pip install plotly")

    # Outlier clip
    mu, sd = pts.mean(axis=0), pts.std(axis=0)
    mask = np.all(np.abs(pts - mu) < sigma_clip * sd, axis=1)
    pts, cols = pts[mask], cols[mask]

    # Subsample
    if len(pts) > max_pts:
        idx  = np.random.choice(len(pts), max_pts, replace=False)
        pts  = pts[idx]
        cols = cols[idx]

    hex_colors = [
        "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        for r, g, b in np.clip(cols, 0, 1)
    ]

    fig = go.Figure(
        go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=1.2, color=hex_colors, opacity=0.85),
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data", bgcolor="black"),
        paper_bgcolor="black",
        font_color="white",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ── cloudflared splat server ──────────────────────────────────────────────────

def serve_splat(splat_path: str, port: int = 8765, timeout: int = 30) -> str | None:
    """
    Serve a .splat file over a cloudflared tunnel for browser viewing.
    No account or auth token required.

    Requires cloudflared to be installed:
        wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
            -O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared

    Args:
        splat_path: path to the .splat file
        port:       local HTTP port to bind
        timeout:    seconds to wait for tunnel URL

    Returns:
        Full antimatter15.com viewer URL, or None if tunnel failed.
    """
    import os
    from pathlib import Path

    serve_dir = str(Path(splat_path).parent)
    filename  = Path(splat_path).name

    class CORSHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=serve_dir, **kwargs)

        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Range")
            self.send_header("Access-Control-Expose-Headers", "Content-Length, Content-Range")
            super().end_headers()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("", port), CORSHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    tunnel_url = None
    for _ in range(timeout):
        line = proc.stderr.readline().decode()
        match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if match:
            tunnel_url = match.group()
            break
        time.sleep(1)

    if tunnel_url:
        viewer_url = f"https://antimatter15.com/splat/?url={tunnel_url}/{filename}"
        print(f"Open in browser:\n{viewer_url}")
        return viewer_url

    print("Tunnel URL not found — re-run serve_splat()")
    return None
