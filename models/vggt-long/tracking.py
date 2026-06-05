"""
MLflow tracking hooks for VGGT-Long pipeline runs.

Wraps the shared PipelineRun context manager and adds VGGT-Long specific
parameter / metric sets.

Tracked per run:
  Parameters  — model, sequence_id, chunk_size, conf_threshold, sample_ratio,
                n_frames, n_chunks, tf_cpu_only
  Metrics     — ATE, RPE, DTW, max_drift, n_points, path_m
  Artifacts   — PLY, GeoJSON trajectory, depth preview PNG, splat

Usage:
    from models.vggt_long.tracking import VGGTLongRun

    with VGGTLongRun(experiment="MapAnything-Tokyo") as run:
        run.log_params(
            sequence_id="seq_B",
            chunk_size=60,
            conf_threshold=0.65,
            n_frames=174,
            n_chunks=3,
            tf_cpu_only=True,
        )
        run.log_trajectory_metrics(metrics, prefix="seq_B")
        run.log_reconstruction(pts, confs, prefix="seq_B")
        run.log_artifact_path(ply_path)
        run.log_artifact_path(geojson_path)
"""

from __future__ import annotations

import numpy as np

from mapanything_pipeline.tracking import PipelineRun


class VGGTLongRun(PipelineRun):
    """
    Context manager for a single VGGT-Long pipeline run.

    Inherits all PipelineRun methods and sets model="vggt-long" by default.

    Args:
        experiment: MLflow experiment name (e.g. "MapAnything-Tokyo")
        tags:       arbitrary key-value tags
    """

    def __init__(
        self,
        experiment: str = "MapAnything",
        tags: dict[str, str] | None = None,
    ):
        super().__init__(experiment=experiment, model="vggt-long", tags=tags)

    # ── VGGT-Long specific parameter set ─────────────────────────────────────

    def log_run_params(
        self,
        *,
        sequence_id: str = "",
        chunk_size: int = 60,
        conf_threshold: float = 0.65,
        sample_ratio: float = 1.0,
        n_frames: int = 0,
        n_chunks: int = 0,
        tf_cpu_only: bool = True,
        epsg_utm: int = 32654,
        **extra,
    ) -> None:
        """
        Log standard VGGT-Long run parameters.

        Args:
            sequence_id:    Mapillary sequence identifier
            chunk_size:     frames per inference chunk
            conf_threshold: confidence cutoff for point aggregation
            sample_ratio:   fraction of confident points retained
            n_frames:       total inference frames
            n_chunks:       number of chunks processed
            tf_cpu_only:    whether TF was pinned to CPU
            epsg_utm:       UTM EPSG for GPS alignment
            **extra:        additional parameters forwarded to log_params()
        """
        self.log_params(
            sequence_id=sequence_id,
            chunk_size=chunk_size,
            conf_threshold=conf_threshold,
            sample_ratio=sample_ratio,
            n_frames=n_frames,
            n_chunks=n_chunks,
            tf_cpu_only=tf_cpu_only,
            epsg_utm=epsg_utm,
            **extra,
        )

    # ── Path-length metric ────────────────────────────────────────────────────

    def log_path_length(self, translations: np.ndarray, prefix: str = "") -> None:
        """
        Log cumulative camera path length derived from translations.

        Args:
            translations: (N, 3) camera centres
            prefix:       metric name prefix
        """
        if not self._mlflow:
            return
        step_dists = np.linalg.norm(np.diff(translations, axis=0), axis=1)
        path_m = float(step_dists.sum())
        sep = "/" if prefix else ""
        self._mlflow.log_metric(f"{prefix}{sep}path_m", path_m)
        print(f"Logged path_m={path_m:.1f}" + (f" [{prefix}]" if prefix else ""))
