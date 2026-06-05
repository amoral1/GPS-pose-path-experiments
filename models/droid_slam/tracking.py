"""
MLflow tracking hooks for DROID-SLAM pipeline runs.

Tracked per run:
  Parameters  — model, sequence, skip, frame_start, frame_end,
                gps_sigma, horn_scale
  Metrics     — gps_arc_m, droid_arc_m, horn_scale, horn_rotation_deg,
                gtsam_mean_error_m
  Artifacts   — TUM trajectory .txt, XZ plot PNG, GPS fusion plot PNG

Usage:
    from models.droid_slam.tracking import DroidSlamRun

    with DroidSlamRun(experiment="MapAnything-Tokyo") as run:
        run.log_run_params(
            sequence="s1_segment",
            skip=1,
            frame_start=0,
            frame_end=-1,
            gps_sigma=3.0,
        )
        run.log_alignment(horn_scale, horn_R, gps_arc_m=gps_arc_m, droid_arc_m=droid_arc_m)
        run.log_artifact_path(tum_path)
        run.log_artifact_path(xz_plot_path)
"""

from __future__ import annotations

import numpy as np

from mapanything_pipeline.tracking import PipelineRun


class DroidSlamRun(PipelineRun):
    """
    Context manager for a single DROID-SLAM pipeline run.

    Inherits all PipelineRun methods and sets model="droid-slam" by default.

    Args:
        experiment: MLflow experiment name
        tags:       arbitrary key-value tags
    """

    def __init__(
        self,
        experiment: str = "MapAnything",
        tags: dict[str, str] | None = None,
    ):
        super().__init__(experiment=experiment, model="droid-slam", tags=tags)

    # ── DROID-SLAM specific parameters ────────────────────────────────────────

    def log_run_params(
        self,
        *,
        sequence: str = "",
        skip: int = 1,
        frame_start: int = 0,
        frame_end: int = -1,
        gps_sigma: float = 3.0,
        **extra,
    ) -> None:
        """
        Log standard DROID-SLAM run parameters.

        Args:
            sequence:    sequence name ("s1_segment", "s1_full", "s2")
            skip:        frame stride used for inference
            frame_start: first frame index
            frame_end:   last frame index (-1 = all)
            gps_sigma:   GPS noise prior for GTSAM fusion (metres)
            **extra:     additional parameters forwarded to log_params()
        """
        self.log_params(
            sequence=sequence,
            skip=skip,
            frame_start=frame_start,
            frame_end=frame_end,
            gps_sigma=gps_sigma,
            **extra,
        )

    # ── Horn / GPS alignment metrics ──────────────────────────────────────────

    def log_alignment(
        self,
        horn_scale: float,
        horn_R: np.ndarray,
        *,
        gps_arc_m: float | None = None,
        droid_arc_m: float | None = None,
        prefix: str = "",
    ) -> None:
        """
        Log Horn transform and arc-length metrics.

        Args:
            horn_scale:  similarity scale factor from horn_transform()
            horn_R:      (2, 2) rotation matrix from horn_transform()
            gps_arc_m:   GPS track arc length in metres
            droid_arc_m: DROID-SLAM track arc length in metres
            prefix:      metric name prefix
        """
        if not self._mlflow:
            return

        sep = "/" if prefix else ""

        import math
        horn_rot_deg = float(math.degrees(math.atan2(float(horn_R[1, 0]), float(horn_R[0, 0]))))

        metrics: dict[str, float] = {
            f"{prefix}{sep}horn_scale":       horn_scale,
            f"{prefix}{sep}horn_rotation_deg": horn_rot_deg,
        }
        if gps_arc_m is not None:
            metrics[f"{prefix}{sep}gps_arc_m"] = gps_arc_m
        if droid_arc_m is not None:
            metrics[f"{prefix}{sep}droid_arc_m"] = droid_arc_m

        self._mlflow.log_metrics(metrics)
        print(
            f"Logged alignment: scale={horn_scale:.4f}  "
            f"rot={horn_rot_deg:.1f}°" +
            (f" [{prefix}]" if prefix else "")
        )

    # ── GTSAM fusion quality ──────────────────────────────────────────────────

    def log_gtsam_fusion(
        self,
        fused_xy: np.ndarray,
        gps_utm: np.ndarray,
        prefix: str = "",
    ) -> None:
        """
        Log mean absolute error between GTSAM-fused and GPS positions.

        Args:
            fused_xy: (N, 2) GTSAM result
            gps_utm:  (M, 2) GPS reference
            prefix:   metric name prefix
        """
        if not self._mlflow:
            return

        n = min(len(fused_xy), len(gps_utm))
        mean_err = float(np.linalg.norm(fused_xy[:n] - gps_utm[:n], axis=1).mean())
        sep = "/" if prefix else ""
        self._mlflow.log_metric(f"{prefix}{sep}gtsam_mean_error_m", mean_err)
        print(f"GTSAM mean error: {mean_err:.2f} m" + (f" [{prefix}]" if prefix else ""))
