"""
MLflow tracking hooks for ORB-SLAM3 pipeline runs.

Tracked per run:
  Parameters  — model, sequence, skip, frame_start, frame_end,
                orb_n_features, orb_scale_factor, fx, fy, cx, cy
  Metrics     — n_keyframes, path_m, mean_step_m, max_step_m,
                ate_m, horn_scale, duration_s
  Artifacts   — KeyFrameTrajectory.txt, trajectory PNG, GPS overlay PNG

Usage:
    from models.orb_slam3.tracking import OrbSlam3Run

    with OrbSlam3Run(experiment="MapAnything-Tokyo") as run:
        run.log_run_params(
            sequence="s2",
            skip=4,
            frame_start=0,
            frame_end=400,
            orb_n_features=1000,
            fx=1090.7,
        )
        run.log_trajectory_stats(stats, prefix="s2")
        run.log_ate(ate_m, prefix="s2")
        run.log_artifact_path(kf_traj_path)
"""

from __future__ import annotations

from mapanything_pipeline.tracking import PipelineRun


class OrbSlam3Run(PipelineRun):
    """
    Context manager for a single ORB-SLAM3 pipeline run.

    Inherits all PipelineRun methods and sets model="orb-slam3" by default.

    Args:
        experiment: MLflow experiment name
        tags:       arbitrary key-value tags
    """

    def __init__(
        self,
        experiment: str = "MapAnything",
        tags: dict[str, str] | None = None,
    ):
        super().__init__(experiment=experiment, model="orb-slam3", tags=tags)

    # ── ORB-SLAM3 specific parameters ─────────────────────────────────────────

    def log_run_params(
        self,
        *,
        sequence: str = "",
        skip: int = 4,
        frame_start: int = 0,
        frame_end: int = -1,
        orb_n_features: int = 1000,
        orb_scale_factor: float = 1.2,
        orb_n_levels: int = 8,
        fx: float = 1090.7,
        fy: float = 1090.7,
        cx: float = 1024.0,
        cy: float = 768.0,
        **extra,
    ) -> None:
        """
        Log standard ORB-SLAM3 run parameters.

        Args:
            sequence:         sequence name ("s1_segment", "s1_full", "s2")
            skip:             frame stride
            frame_start:      first frame index
            frame_end:        last frame index (-1 = all)
            orb_n_features:   ORB feature count per frame
            orb_scale_factor: ORB pyramid scale factor
            orb_n_levels:     ORB pyramid levels
            fx, fy, cx, cy:   camera intrinsics
            **extra:          additional parameters forwarded to log_params()
        """
        self.log_params(
            sequence=sequence,
            skip=skip,
            frame_start=frame_start,
            frame_end=frame_end,
            orb_n_features=orb_n_features,
            orb_scale_factor=orb_scale_factor,
            orb_n_levels=orb_n_levels,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            **extra,
        )

    # ── Trajectory statistics ─────────────────────────────────────────────────

    def log_trajectory_stats(
        self,
        stats: dict,
        prefix: str = "",
    ) -> None:
        """
        Log trajectory statistics from metrics.compute_trajectory_stats().

        Logged: n_keyframes, path_m, mean_step_m, max_step_m, duration_s

        Args:
            stats:  output of compute_trajectory_stats()
            prefix: metric name prefix
        """
        if not self._mlflow:
            return

        sep    = "/" if prefix else ""
        logged = {
            f"{prefix}{sep}{k}": float(v)
            for k, v in stats.items()
            if k in ("n_keyframes", "path_m", "mean_step_m", "max_step_m", "duration_s")
        }
        self._mlflow.log_metrics(logged)
        print(f"Logged {len(logged)} trajectory stats" + (f" [{prefix}]" if prefix else ""))

    # ── ATE ───────────────────────────────────────────────────────────────────

    def log_ate(self, ate_m: float, prefix: str = "") -> None:
        """
        Log ATE (RMSE) in metres.

        Args:
            ate_m:  from metrics.compute_ate_tum()
            prefix: metric name prefix
        """
        if not self._mlflow:
            return
        sep = "/" if prefix else ""
        self._mlflow.log_metric(f"{prefix}{sep}ate_m", float(ate_m))
        print(f"ATE logged: {ate_m:.3f} m" + (f" [{prefix}]" if prefix else ""))
