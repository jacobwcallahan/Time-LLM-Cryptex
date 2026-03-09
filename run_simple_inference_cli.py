#!/usr/bin/env python3
"""
CLI for Simple Inference: run inference from day after training end to end of dataset.
Invoked by the frontend to stream console output. Usage:
  python run_simple_inference_cli.py --experiment_name X --run_id Y
"""
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from hpo_core.PipelineRunner import PipelineRunner
from hpo_core.DataManager import DataManager
from hpo_core.HpoArgs import HpoArgs
from hpo_core.WorkDir import WorkDir
import mlflow

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://192.168.1.103:5000")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--run_id", required=True)
    args = parser.parse_args()

    experiment_name = args.experiment_name
    run_id = args.run_id

    print(f"Simple Inference: experiment={experiment_name}, run_id={run_id}")
    print("-" * 60)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"Error: Experiment '{experiment_name}' not found")
        sys.exit(1)
    runs = client.search_runs(experiment_ids=[exp.experiment_id], filter_string=f"run_id = '{run_id}'", max_results=1)
    if not runs:
        runs = client.search_runs(experiment_ids=[exp.experiment_id], filter_string=f"tags.mlflow.runName = '{run_id}'", max_results=1)
    if not runs:
        print(f"Error: No run found with ID '{run_id}'")
        sys.exit(1)

    run = runs[0]
    params = dict(run.data.params)
    end_date_str = params.get("end date") or params.get("end_date")
    if not end_date_str:
        print("Error: Run has no 'end date' param. Cannot determine inference start.")
        sys.exit(1)
    granularity = params.get("granularity", "daily")
    aggregate = int(params.get("aggregate", 1))
    model_name = run.data.tags.get("mlflow.runName", run.info.run_id)

    end_dt = datetime.strptime(str(end_date_str).strip()[:10], "%Y-%m-%d")
    inf_start_dt = end_dt + timedelta(days=1)
    inf_start_str = inf_start_dt.strftime("%Y-%m-%d")

    print(f"Training end date: {end_date_str}")
    print(f"Inference start: {inf_start_str} (day after training end)")
    print(f"Inference end: last date in dataset")
    print(f"Granularity: {granularity}, Aggregate: {aggregate}")
    print("-" * 60)

    project_root = Path(__file__).parent
    dataset_path = project_root / "dataset" / "candles"
    work_dir_path = project_root / "temp" / "frontend_simple_inference"

    hpo_args = HpoArgs(
        parse_cli=False,
        model_name=model_name,
        experiment_name=experiment_name,
        granularity=granularity,
        aggregate=aggregate,
        inf_start=inf_start_str,
        inf_end=None,
        data_path=None,
    )
    work_dir = WorkDir(hpo_args, work_dir=work_dir_path, dataset_path=dataset_path)
    work_dir.create_work_dir()

    data_file = Path(work_dir.get_full_data_path())
    if not data_file.exists():
        print(f"Error: Dataset not found: {data_file}")
        sys.exit(1)

    data_manager = DataManager(work_dir)
    pipeline_runner = PipelineRunner(work_dir)
    pipeline_runner.run_inference(experiment_name, model_name)

    print("-" * 60)
    print("Simple inference completed. Results saved to MLflow.")


if __name__ == "__main__":
    main()
