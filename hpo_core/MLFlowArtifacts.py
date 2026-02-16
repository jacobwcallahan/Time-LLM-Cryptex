from hpo_core.WorkDir import WorkDir
import mlflow
import os
import pandas as pd
from pathlib import Path

class MLFlowArtifacts:
    """
    This class is responsible for logging the artifacts to the MLflow run.
    """
    MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]

    def __init__(self, client: mlflow.tracking.MlflowClient, work_dir: WorkDir):
        self.client = client
        mlflow.set_tracking_uri(self.MLFLOW_TRACKING_URI)
        self.work_dir = work_dir

    def log_all_metrics(self, run_id: str, metrics: dict):
        """
        Logs all metrics to the MLflow run.

        Args:
            metrics (dict): Dictionary of metrics DataFrames to log
        """

        for metric, data in metrics.items():
            self.work_dir.write_metrics(metric, data)
            self.client.log_artifact(run_id = run_id, local_path = self.work_dir.metrics_path(metric))


    def log_summary_table(self, run_id: str, summary_table_path: Path):
        """
        Logs the summary table to the MLflow run.
        """
        self.client.log_artifact(run_id = run_id, local_path = summary_table_path)