"""Experiment Runs tab helpers - list runs, run simple inference."""

import os
import subprocess
import sys
import traceback
from pathlib import Path

import mlflow

from .common import MLFLOW_TRACKING_URI, logger


def _get_run_artifact_status(client, run_id):
    """Check if inference and backtesting were run based on MLflow artifacts.
    Returns (has_inference: bool, has_backtest: bool)."""
    has_inference, has_backtest = False, False
    try:
        artifacts = client.list_artifacts(run_id)
        for a in artifacts:
            path_lower = (a.path or "").lower()
            if "ohlcv_inference" in path_lower or (path_lower.endswith("inference.csv") and "ret_" not in path_lower):
                has_inference = True
            if "summary_table" in path_lower:
                has_backtest = True
            if has_inference and has_backtest:
                break
    except Exception as e:
        logger.debug(f"list_artifacts error for run {run_id}: {e}")
    return has_inference, has_backtest


def list_experiment_runs_with_status(experiment_name):
    """
    List all non-failed runs in an experiment with their inference/backtest status.
    Returns list of dicts: [{"run_id": str, "run_name": str, "has_inference": bool, "has_backtest": bool}, ...]
    or error message string.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return f"Error: Experiment '{experiment_name}' not found"
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string='attributes.status = "FINISHED"',
            order_by=["start_time DESC"],
        )
        result = []
        for r in runs:
            run_uuid = r.info.run_id
            run_name = r.data.tags.get("mlflow.runName", run_uuid)
            has_inf, has_bt = _get_run_artifact_status(client, run_uuid)
            result.append({
                "run_id": run_uuid,
                "run_name": run_name,
                "has_inference": has_inf,
                "has_backtest": has_bt,
            })
        return result
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"


def run_simple_inference(experiment_name, run_id):
    """
    Run inference on the selected model with streaming console output.
    Yields output progressively, then final status.
    """
    if not run_id:
        yield "Error: Select a run first."
        return
    if not experiment_name:
        yield "Error: Experiment name is required."
        return

    try:
        project_root = Path(__file__).resolve().parent.parent.parent
        script_path = project_root / "run_simple_inference_cli.py"
        if not script_path.exists():
            yield f"Error: Script not found: {script_path}"
            return

        cmd = [
            sys.executable,
            str(script_path),
            "--experiment_name", str(experiment_name),
            "--run_id", str(run_id),
        ]
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        process = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        output = ""
        for line in iter(process.stdout.readline, ""):
            output += line
            yield output

        process.stdout.close()
        return_code = process.wait()

        if return_code != 0:
            output += f"\n\n--- Process exited with code {return_code} ---"
            yield output
    except Exception as e:
        yield f"Error: {str(e)}\n\n{traceback.format_exc()}"
