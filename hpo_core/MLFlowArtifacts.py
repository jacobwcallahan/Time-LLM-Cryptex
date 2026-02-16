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

    def __init__(self, run_id: str, client: mlflow.tracking.MlflowClient, work_dir: WorkDir):
        self.run_id = run_id
        self.work_dir = work_dir
        self.client = client

    def log_all_metrics(self, metrics: dict):
        """
        Logs all metrics to the MLflow run.

        Args:
            metrics (dict): Dictionary of metrics DataFrames to log
        """

        for metric, data in metrics.items():
            self.work_dir.write_metrics(metric, data)
            self.client.log_artifact(run_id = self.run_id, local_path = self.work_dir.metrics_path(metric))


    def log_summary_table(self, summary_table_path: Path):
        """
        Logs the summary table to the MLflow run.
        """
        self.client.log_artifact(run_id = self.run_id, local_path = summary_table_path)

    def log_inference_data(self, inference_data_path: Path):
        """
        Logs the inference data to the MLflow run.
        """
        self.client.log_artifact(run_id = self.run_id, local_path = inference_data_path)