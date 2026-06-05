"""
MLflow tracking hooks for the MapAnything pipeline.

Wraps the existing pipeline modules — no MLflow calls are embedded in
api.py / metrics.py / reconstruction.py / splat.py themselves.
This module is the single integration point.

Tracked per run:
  Parameters  — model name, sequence ID, stride, inference settings
  Metrics     — ATE, RPE, DTW, max drift, n_frames, n_gaussians, splat_mb
  Artifacts   — GeoJSON trajectory, PLY point cloud, BEV drift image, .splat

Usage (Colab / local):
    import mlflow
    from mapanything_pipeline.tracking import PipelineRun

    with PipelineRun(experiment="MapAnything-Tokyo", model="mapanything") as run:
        run.log_params(stride=5, inference_frames=174, chunk_size=60)
        metrics = compute_all(df)
        run.log_trajectory_metrics(metrics, prefix="seq_B")
        run.log_splat("seq_B_improved.splat")
        run.log_artifact_path("seq_B_vision_trajectory.geojson")

Dashboard:
    mlflow ui --port 5000
    # or on Colab:
    !mlflow ui --host 0.0.0.0 --port 5000 &
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


# ── Run context manager ────────────────────────────────────────────────────────

class PipelineRun:
    """
    Context manager for a single MapAnything / VGGT-Long pipeline run.

    Creates an MLflow run on entry, ends it on exit (even on error).
    All log_* methods are no-ops if MLflow is not installed.

    Args:
        experiment: MLflow experiment name (e.g. "MapAnything-Tokyo")
        model:      model identifier — "mapanything" | "vggt-long" | "vggt"
        tags:       arbitrary key-value tags attached to the run
    """

    def __init__(
        self,
        experiment: str = "MapAnything",
        model: str = "mapanything",
        tags: dict[str, str] | None = None,
    ):
        self.experiment = experiment
        self.model      = model
        self.tags       = tags or {}
        self._run       = None
        self._mlflow    = None

    def __enter__(self):
        try:
            import mlflow
            self._mlflow = mlflow
            mlflow.set_experiment(self.experiment)
            self._run = mlflow.start_run(tags={"model": self.model, **self.tags})
            mlflow.log_param("model", self.model)
            print(f"MLflow run started: {self._run.info.run_id}")
        except ImportError:
            print("[tracking] mlflow not installed — logging disabled. pip install mlflow")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._mlflow and self._run:
            self._mlflow.end_run()
            ui_url = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
            print(f"Run ended. View at: {ui_url}")
        return False  # don't suppress exceptions

    # ── Parameter logging ──────────────────────────────────────────────────────

    def log_params(self, **kwargs: Any) -> None:
        """
        Log scalar parameters for this run.

        Suggested keys:
            sequence_id, stride, inference_frames, chunk_size,
            memory_efficient_inference, minibatch_size, n_splits,
            confidence_percentile, max_gaussians, k_neighbors
        """
        if not self._mlflow:
            return
        self._mlflow.log_params(kwargs)

    # ── Trajectory metric logging (from metrics.compute_all) ──────────────────

    def log_trajectory_metrics(
        self,
        metrics: dict,
        prefix: str = "",
    ) -> None:
        """
        Log ATE / RPE / DTW / max_drift from metrics.compute_all().

        Args:
            metrics: output of compute_all(df) — keys like "ATE (RMSE) m"
            prefix:  prepended to metric names, e.g. "seq_B/" or "mapanything/"

        Logged metric names (with prefix "seq_B/"):
            seq_B/ate_m, seq_B/rpe_mean_m, seq_B/rpe_rmse_m,
            seq_B/max_drift_m, seq_B/dtw_normalised, seq_B/n_frames
        """
        if not self._mlflow:
            return

        key_map = {
            "ATE (RMSE) m":     "ate_m",
            "RPE mean m":       "rpe_mean_m",
            "RPE RMSE m":       "rpe_rmse_m",
            "RPE max m":        "rpe_max_m",
            "Max drift m":      "max_drift_m",
            "DTW (normalised)": "dtw_normalised",
            "n_frames":         "n_frames",
        }
        sep = "/" if prefix else ""
        logged = {}
        for raw_key, short_key in key_map.items():
            if raw_key in metrics:
                logged[f"{prefix}{sep}{short_key}"] = float(metrics[raw_key])

        self._mlflow.log_metrics(logged)
        print(f"Logged {len(logged)} trajectory metrics" + (f" [{prefix}]" if prefix else ""))

    # ── Vision pose vs GPS metrics (Section 8 comparison) ─────────────────────

    def log_vision_comparison(
        self,
        comparison: dict,
        prefix: str = "",
    ) -> None:
        """
        Log MapAnything / VGGT-Long vs GPS comparison metrics.

        Args:
            comparison: output of Section 8 loop — keys like
                        "scale", "rpe_mean_m", "gps_path_m", "vision_path_m"
            prefix:     e.g. "seq_B/mapanything/"
        """
        if not self._mlflow:
            return

        key_map = {
            "scale":         "scale_factor",
            "rpe_mean_m":    "vision_rpe_mean_m",
            "rpe_rmse_m":    "vision_rpe_rmse_m",
            "rpe_max_m":     "vision_rpe_max_m",
            "gps_path_m":    "gps_path_m",
            "vision_path_m": "vision_path_m",
        }
        sep = "/" if prefix else ""
        logged = {
            f"{prefix}{sep}{v}": float(comparison[k])
            for k, v in key_map.items()
            if k in comparison
        }
        self._mlflow.log_metrics(logged)

    # ── Reconstruction stats ───────────────────────────────────────────────────

    def log_reconstruction(
        self,
        pts: np.ndarray,
        confs: np.ndarray | None = None,
        prefix: str = "",
    ) -> None:
        """
        Log point cloud statistics from extract_reconstruction().

        Logged: n_points, conf_mean, conf_min, conf_max (if confs provided)
        """
        if not self._mlflow:
            return

        sep = "/" if prefix else ""
        m: dict[str, float] = {f"{prefix}{sep}n_points": float(len(pts))}
        if confs is not None:
            m[f"{prefix}{sep}conf_mean"] = float(confs.mean())
            m[f"{prefix}{sep}conf_min"]  = float(confs.min())
            m[f"{prefix}{sep}conf_max"]  = float(confs.max())

        self._mlflow.log_metrics(m)

    # ── Splat stats ────────────────────────────────────────────────────────────

    def log_splat(self, splat_path: str, prefix: str = "") -> None:
        """
        Log .splat file size and Gaussian count, and upload as artifact.

        Args:
            splat_path: path to the .splat file
            prefix:     metric name prefix
        """
        if not self._mlflow:
            return

        path = Path(splat_path)
        if not path.exists():
            print(f"[tracking] splat not found: {splat_path}")
            return

        size_bytes = path.stat().st_size
        n_gaussians = size_bytes // 32  # 32 bytes per Gaussian

        sep = "/" if prefix else ""
        self._mlflow.log_metrics({
            f"{prefix}{sep}splat_mb":       round(size_bytes / 1024 ** 2, 2),
            f"{prefix}{sep}n_gaussians":    float(n_gaussians),
        })
        self._mlflow.log_artifact(str(path))
        print(f"Logged splat: {n_gaussians:,} Gaussians  {size_bytes/1024**2:.0f} MB")

    # ── Generic artifact logging ───────────────────────────────────────────────

    def log_artifact_path(self, path: str, artifact_subdir: str = "") -> None:
        """
        Upload any file as an MLflow artifact.
        Useful for: GeoJSON trajectories, PLY files, BEV drift images, CSVs.
        """
        if not self._mlflow:
            return
        p = Path(path)
        if not p.exists():
            print(f"[tracking] artifact not found: {path}")
            return
        self._mlflow.log_artifact(str(p), artifact_path=artifact_subdir or None)
        print(f"Artifact logged: {p.name}")

    def log_figure(self, fig, name: str) -> None:
        """
        Log a Plotly or matplotlib figure directly.

        Args:
            fig:  plotly.graph_objects.Figure  OR  matplotlib.figure.Figure
            name: filename without extension (extension added automatically)
        """
        if not self._mlflow:
            return
        try:
            # Plotly
            import plotly
            if isinstance(fig, plotly.graph_objects.Figure):
                self._mlflow.log_figure(fig, f"{name}.html")
                return
        except ImportError:
            pass
        try:
            # Matplotlib
            import matplotlib.pyplot as plt
            if hasattr(fig, "savefig"):
                self._mlflow.log_figure(fig, f"{name}.png")
                return
        except ImportError:
            pass
        print(f"[tracking] unrecognised figure type: {type(fig)}")


# ── DagsHub setup helper ──────────────────────────────────────────────────────

def dagshub_init(repo_owner: str, repo_name: str, token: str | None = None) -> str:
    """
    Point MLflow at DagsHub's hosted tracking server for this repo.

    Preferred on Colab — call this once before any PipelineRun:

        from mapanything_pipeline.tracking import dagshub_init
        dagshub_init("your-gh-username", "mapanything-mapillary-pipeline")

    Args:
        repo_owner: your GitHub / DagsHub username
        repo_name:  repository name
        token:      DagsHub access token (Settings → Tokens).
                    If None, falls back to DAGSHUB_TOKEN env var.
                    Never hardcode — use Colab Secrets or os.environ.

    Returns:
        The MLflow tracking URI that was set.
    """
    token = token or os.environ.get("DAGSHUB_TOKEN", "")

    try:
        import dagshub
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
        print(f"DagsHub MLflow ready: {uri}")
        print(f"Dashboard: https://dagshub.com/{repo_owner}/{repo_name}/experiments")
        return uri
    except ImportError:
        pass

    # Fallback: set env vars manually if dagshub package not installed
    uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
    os.environ["MLFLOW_TRACKING_URI"]      = uri
    os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    import mlflow
    mlflow.set_tracking_uri(uri)
    print(f"DagsHub MLflow (manual): {uri}")
    return uri


# ── Convenience: model comparison across runs ─────────────────────────────────

def compare_models(
    experiment: str,
    metric: str = "seq_B/ate_m",
) -> "pd.DataFrame | None":
    """
    Pull all completed runs from an experiment and compare by metric.

    Args:
        experiment: MLflow experiment name
        metric:     metric key to sort by (ascending)

    Returns:
        pandas DataFrame of runs sorted by metric, or None if mlflow unavailable.

    Example:
        df = compare_models("MapAnything-Tokyo", metric="seq_B/ate_m")
        print(df[["params.model", "metrics.seq_B/ate_m", "metrics.seq_B/rpe_mean_m"]])
    """
    try:
        import mlflow
    except ImportError:
        print("[tracking] mlflow not installed")
        return None

    client = mlflow.tracking.MlflowClient()
    exp    = client.get_experiment_by_name(experiment)
    if exp is None:
        print(f"Experiment '{experiment}' not found")
        return None

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=[f"metrics.{metric} ASC"],
    )
    return runs
