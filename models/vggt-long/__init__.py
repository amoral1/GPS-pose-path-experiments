"""
VGGT-Long model package.

Submodules mirror the mapanything_pipeline/ layout:
    api            — environment setup, inference, checkpoint, pose loading
    metrics        — GPS reference, pose summary, tf.summary, shared metric re-exports
    reconstruction — aggregate_world_points, write_ply, depth_preview, tsdf_fusion
    viz            — plotly_pointcloud, export_geojson, plot_trajectory
    tracking       — VGGTLongRun (MLflow context manager)
"""
