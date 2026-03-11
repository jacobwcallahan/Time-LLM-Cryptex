"""Inference tab helpers - run inference, plot MLflow inference data."""

import traceback
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from hpo_core.PipelineRunner import PipelineRunner
from hpo_core.DataManager import DataManager
from hpo_core.HpoArgs import HpoArgs
from hpo_core.WorkDir import WorkDir

from .common import MLFLOW_TRACKING_URI, _to_date_str, logger


def _project_root():
    return Path(__file__).resolve().parent.parent.parent


def _fetch_metrics_from_mlflow(client, run_id, pred_horizon=1):
    """Fetch MAE, MSE, MDA from MLflow artifacts for a specific prediction horizon."""
    pred_horizon = int(pred_horizon) if pred_horizon else 1
    mae, mse, mda = None, None, None
    for artifact_name, suffix, out_var in [
        ("mae_metrics.csv", "mae", "mae"),
        ("mse_metrics.csv", "mse", "mse"),
        ("mda_metrics.csv", "mda", "mda"),
    ]:
        try:
            path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_name)
            df = pd.read_csv(path)
            if "prediction" in df.columns and "value" in df.columns:
                target_pred = f"inf_pred_{pred_horizon}_{suffix}"
                row = df[df["prediction"].astype(str).str.strip() == target_pred]
                if len(row) > 0:
                    val = float(row["value"].iloc[0])
                    if out_var == "mae":
                        mae = val
                    elif out_var == "mse":
                        mse = val
                    else:
                        mda = val
        except Exception as e:
            logger.debug(f"Could not fetch {artifact_name}: {e}")
    return mae, mse, mda


def run_inference_handler(model_name, experiment_name, custom_dataset_path, granularity, aggregate, start_date, end_date, save_path=None):
    """Run inference using PipelineRunner."""
    try:
        if not model_name:
            return "Error: Model Name (Run ID) is required"
        if not experiment_name:
            return "Error: Experiment Name is required"

        project_root = _project_root()
        data_path = None
        if custom_dataset_path and str(custom_dataset_path).strip():
            p = Path(custom_dataset_path.strip())
            data_path = str(p) if p.is_absolute() else str(project_root / p)

        args = HpoArgs(
            parse_cli=False,
            model_name=model_name,
            experiment_name=experiment_name,
            granularity=granularity,
            aggregate=int(aggregate or 1),
            inf_start=_to_date_str(start_date),
            inf_end=_to_date_str(end_date),
            data_path=data_path,
        )

        dataset_path = project_root / "dataset" / "candles"
        work_dir_path = project_root / "temp" / "frontend_inference"

        work_dir = WorkDir(args, work_dir=work_dir_path, dataset_path=dataset_path)
        work_dir.create_work_dir()

        data_file = Path(work_dir.get_full_data_path())
        if not data_file.exists():
            hint = (
                "Provide a Custom Data Path in the Inference tab (e.g. dataset/candles/your_data.csv)."
                if not data_path
                else f"Check that the path exists: {data_file}"
            )
            return (
                f"Error: Dataset not found: {data_file}\n\n"
                f"{hint}\n\n"
                f"For default {granularity} data, expected: {dataset_path}/candlesticks-{work_dir.granularity_map.get(granularity.lower(), 'D')}.csv"
            )
        data_manager = DataManager(work_dir)
        pipeline_runner = PipelineRunner(work_dir)
        pipeline_runner.run_inference(experiment_name, model_name)

        return f"Inference completed successfully!\nResults saved to: {work_dir.get_inferenced_path()}"
    except Exception as e:
        return f"Error running inference: {str(e)}\n\n{traceback.format_exc()}"


def check_and_plot_mlflow_inference(experiment_name, run_id, pred_horizon=1):
    """Check if MLflow has inference data for the given run and plot it as candlestick with prediction overlay."""
    logger.debug(f"Checking and plotting MLflow inference for run: {run_id} in experiment: {experiment_name}")
    if not run_id:
        return "Error: Please enter a Run ID", None, None, None, None

    try:
        pred_horizon = int(pred_horizon) if pred_horizon else 1
    except (ValueError, TypeError):
        pred_horizon = 1

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    except Exception as e:
        return f"Error: {str(e)}", None, None, None, None

    try:
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return f"Error: Experiment '{experiment_name}' not found", None, None, None, None
        runs = client.search_runs(experiment_ids=[exp.experiment_id], filter_string=f"run_id = '{run_id}'", max_results=1)
        if not runs:
            runs = client.search_runs(experiment_ids=[exp.experiment_id], filter_string=f"tags.mlflow.runName = '{run_id}'", max_results=1)
    except Exception as e:
        return f"Error searching runs: {str(e)}", None, None, None, None

    if not runs:
        return f"No run found with ID/name '{run_id}' in experiment '{experiment_name}'", None, None, None, None

    run_id = runs[0].info.run_id

    try:
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="ohlcv_inference.csv")
    except Exception as e:
        return f"Error downloading artifacts: {str(e)}", None, None, None, None

    df = pd.read_csv(path)
    mae, mse, mda = _fetch_metrics_from_mlflow(client, run_id, pred_horizon)

    date_col = None
    if 'date' in df.columns:
        date_col = 'date'
    elif 'timestamp' in df.columns:
        date_col = 'timestamp'

    if date_col:
        if df[date_col].dtype in ['int64', 'float64']:
            df[date_col] = pd.to_datetime(df[date_col], unit='s')
        else:
            df[date_col] = pd.to_datetime(df[date_col])

    pred_cols = [c for c in df.columns if 'predicted' in c.lower()]
    max_horizon = len(pred_cols) if pred_cols else 0
    has_ohlcv = all(col in df.columns for col in ['open', 'high', 'low', 'close'])

    if has_ohlcv and date_col:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df[date_col],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLCV',
            increasing_line_color='green',
            decreasing_line_color='red'
        ))
        pred_col = f'close_predicted_{pred_horizon}'
        if pred_col not in df.columns:
            pred_col = next((c for c in pred_cols if str(pred_horizon) in c), None)
        if pred_col and pred_col in df.columns:
            x_pred = df[date_col].iloc[pred_horizon:].values
            y_pred = df[pred_col].iloc[:-pred_horizon].values
            valid = np.isfinite(y_pred)
            if valid.any():
                fig.add_trace(go.Scatter(
                    x=x_pred[valid], y=y_pred[valid],
                    mode='lines',
                    name=f'Prediction ({pred_horizon} step{"s" if pred_horizon > 1 else ""} ahead)',
                    line=dict(color='blue', width=2)
                ))
        fig.update_layout(
            title=f"Inference Results for Run: {run_id}",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            width=1000,
            height=600,
        )
        status = f"Found inference data for run: {run_id}\nShape: {df.shape[0]} rows, {df.shape[1]} columns\nAvailable prediction horizons: 1-{max_horizon}" if max_horizon > 0 else f"Found inference data for run: {run_id}\nShape: {df.shape[0]} rows, {df.shape[1]} columns"
        return status, fig, mae, mse, mda

    elif date_col:
        fig = go.Figure()
        if 'close' in df.columns:
            fig.add_trace(go.Scatter(
                x=df[date_col],
                y=df['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='black', width=1)
            ))
        pred_col = f'close_predicted_{pred_horizon}'
        if pred_col not in df.columns:
            pred_col = next((c for c in pred_cols if str(pred_horizon) in c), None)
        if pred_col and pred_col in df.columns:
            x_pred = df[date_col].iloc[pred_horizon:].values
            y_pred = df[pred_col].iloc[:-pred_horizon].values
            valid = np.isfinite(y_pred)
            if valid.any():
                fig.add_trace(go.Scatter(
                    x=x_pred[valid], y=y_pred[valid],
                    mode='lines',
                    name=f'Prediction ({pred_horizon} step{"s" if pred_horizon > 1 else ""} ahead)',
                    line=dict(color='blue', width=2)
                ))
        fig.update_layout(
            title=f"Inference Results for Run: {run_id}",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white",
            width=1000,
            height=600,
        )
        status = f"Found inference data for run: {run_id}\nShape: {df.shape[0]} rows, {df.shape[1]} columns\nAvailable prediction horizons: 1-{max_horizon}" if max_horizon > 0 else f"Found inference data for run: {run_id}\nShape: {df.shape[0]} rows, {df.shape[1]} columns"
        return status, fig, mae, mse, mda

    else:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns[:4].tolist()
        fig = px.line(df, y=numeric_cols, title=f"Inference Results for Run: {run_id}")
        fig.update_layout(width=1000, height=600)
        return f"Found inference data for run: {run_id}\nShape: {df.shape[0]} rows, {df.shape[1]} columns", fig, mae, mse, mda
