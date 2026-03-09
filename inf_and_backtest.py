"""
Standalone inference + backtest: run PipelineRunner inference and backtest without training first.

Requires an existing MLflow run (trained model). Use --run_id (MLflow run name) and --experiment_name.

Example:
    python inf_and_backtest.py --experiment_name my_exp --run_id trial_0_daily_full_dates_... --inf_start 2024-01-01 --inf_end 2024-12-31
"""

import argparse
import os
from pathlib import Path

# Set MLflow/MinIO env before PipelineRunner import (for model artifact loading)
MLFLOW_SERVER_IP = os.environ.get("MLFLOW_SERVER_IP", "192.168.1.103")
if "MLFLOW_TRACKING_URI" not in os.environ:
    os.environ["MLFLOW_TRACKING_URI"] = f"http://{MLFLOW_SERVER_IP}:5000"
if "AWS_ACCESS_KEY_ID" not in os.environ:
    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
if "AWS_SECRET_ACCESS_KEY" not in os.environ:
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
if "MLFLOW_S3_ENDPOINT_URL" not in os.environ:
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{MLFLOW_SERVER_IP}:9000"

import mlflow

from hpo_core.HpoArgs import HpoArgs
from hpo_core.WorkDir import WorkDir
from hpo_core.DataManager import DataManager
from hpo_core.PipelineRunner import PipelineRunner
from hpo_core.CalcMetrics import CalcMetrics


def _parse_cli():
    parser = argparse.ArgumentParser(description="Run inference and backtest without training")
    parser.add_argument("--experiment_name", required=True, help="MLflow experiment name")
    parser.add_argument("--run_id", required=True, help="MLflow run name / model ID (from a trained run)")
    parser.add_argument("--inf_start", type=str, default=None, help="Inference start date (YYYY-MM-DD). Default: first date in dataset")
    parser.add_argument("--inf_end", type=str, default=None, help="Inference end date (YYYY-MM-DD). Default: last date in dataset")
    parser.add_argument("--granularity", type=str, default="daily", help="Data granularity")
    parser.add_argument("--aggregate", type=int, default=1, help="Aggregation period")
    parser.add_argument("--returns", action="store_true", help="Use returns target")
    parser.add_argument("--data_path", type=str, default=None, help="Override dataset path")
    parser.add_argument("--work_dir", type=str, default="temp/standalone_inf", help="Work directory for this run")
    parser.add_argument("--no_backtest", action="store_true", help="Skip backtest (default: run backtest)")
    return parser.parse_args()


def inf_and_backtest(
    experiment_name: str,
    run_id: str,
    inf_start: str = None,
    inf_end: str = None,
    granularity: str = "daily",
    aggregate: int = 1,
    returns: bool = False,
    data_path: str = None,
    work_dir_path: str = "temp/standalone_inf",
    run_backtest: bool = True,
):
    """
    Run inference and optionally backtest without training.

    Args:
        experiment_name: MLflow experiment name
        run_id: MLflow run name / model ID (must exist from prior training)
        inf_start: Inference start date (YYYY-MM-DD). None = first date in dataset
        inf_end: Inference end date (YYYY-MM-DD). None = last date in dataset
        granularity: Data granularity (daily, hourly, etc.)
        aggregate: Aggregation period
        returns: Use returns target
        data_path: Override dataset path
        work_dir_path: Work directory for outputs
        run_backtest: Whether to run backtest after inference
    """
    args = HpoArgs(parse_cli=False)
    args.experiment_name = experiment_name
    args.inf_start = inf_start
    args.inf_end = inf_end
    args.granularity = granularity
    args.aggregate = aggregate
    args.returns = returns
    args.data_path = data_path

    work_dir = WorkDir(args, work_dir=Path(work_dir_path))
    work_dir.create_work_dir()

    data_manager = DataManager(work_dir)
    pipeline_runner = PipelineRunner(work_dir)

    pipeline_runner.run_inference(experiment_name=experiment_name, run_id=run_id)

    if run_backtest:
        pipeline_runner.run_backtest(pipeline=True, run_id=run_id, experiment_name=experiment_name)
        print(f"Backtest summary saved to {work_dir.summary_table_path()}")

    # Calc metrics and log to MLflow
    mlflow_run_id, mlflow_params = pipeline_runner.get_mlflow_run_info(
        run_id, experiment_name=experiment_name, tracking_uri=os.environ.get("MLFLOW_TRACKING_URI")
    )
    client = mlflow.tracking.MlflowClient()
    calc_metrics = CalcMetrics(args, data_manager, work_dir, params=mlflow_params)
    calc_metrics.calc_and_log_to_mlflow(
        client=client,
        run_id=mlflow_run_id,
        log_returns_inference=args.returns,
        summary_table_path=work_dir.summary_table_path() if run_backtest else None,
    )

    print(f"Inference results saved to {work_dir.get_inferenced_path()}")


if __name__ == "__main__":
    cli = _parse_cli()
    inf_and_backtest(
        experiment_name=cli.experiment_name,
        run_id=cli.run_id,
        inf_start=cli.inf_start,
        inf_end=cli.inf_end,
        granularity=cli.granularity,
        aggregate=cli.aggregate,
        returns=cli.returns,
        data_path=cli.data_path,
        work_dir_path=cli.work_dir,
        run_backtest=not cli.no_backtest,
    )
