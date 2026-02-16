from hpo_core.WorkDir import WorkDir
import mlflow

class MLFlowArtifacts:
    """
    This class is responsible for logging the artifacts to the MLflow run.
    """

    def __init__(self, client: mlflow.tracking.MlflowClient, work_dir: WorkDir):
        self.client = client
        self.work_dir = work_dir

    def log_all_metrics(self, metrics: dict):
        """
        Logs all metrics to the MLflow run.

        Args:
            metrics (dict): Dictionary of metrics DataFrames to log
        """

        for metric, data in metrics.items():
            self.work_dir.write_metrics(metric, data)
            self.client.log_artifact(self.work_dir.metrics_path(metric))


    def log_artifact(self, path: str):
        """
        Logs an artifact to the MLflow run.

        Args:
            path: Path to the artifact
        """
        self.client.log_artifact(path)