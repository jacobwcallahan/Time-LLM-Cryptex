import gradio as gr
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hpo_core.PipelineRunner import PipelineRunner
from hpo_core.DataManager import DataManager
from hpo_core.HpoArgs import HpoArgs
from hpo_core.WorkDir import WorkDir
import logging
import traceback

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


# MLflow configuration
MLFLOW_TRACKING_URI = "http://192.168.1.103:5000"
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://192.168.1.103:9000"

def check_inf_after_train(end_date, inf_start):
    """Checks if the inference start is after the training end date.
    
    Must be used in the change event of the inference start."""
    print(f"End date: {end_date}, Inf start: {inf_start}")
    if end_date is None or inf_start is None:
        return gr.update()

    if inf_start <= end_date:
        return gr.update(
            value=datetime.fromtimestamp(end_date + 86400.0).strftime("%Y-%m-%d"),
            info="Inference start MUST be after training end date."
        )

    return gr.update(info=None)


def start_before_end(start_date, end_date):
    """Checks if the start date is before the end date.

    Must be used in the change event of the start date."""
    if start_date is None or end_date is None:
        return gr.update()

    def _to_ts(val):
        if isinstance(val, (int, float)):
            return val
        if hasattr(val, "timestamp"):
            return val.timestamp()
        return None

    start_ts = _to_ts(start_date)
    end_ts = _to_ts(end_date)
    if start_ts is None or end_ts is None:
        return gr.update()

    if start_ts >= end_ts:
        return gr.update(
            value=datetime.fromtimestamp(end_ts - 86400.0).strftime("%Y-%m-%d"),
            info="Start date MUST be before end date."
        )
    return gr.update(info=None)

def end_after_start(end_date, start_date):
    """Checks if the end date is after the start date.

    Must be used in the change event of the end date."""
    if end_date is None or start_date is None:
        return gr.update()

    def _to_ts(val):
        if isinstance(val, (int, float)):
            return val
        if hasattr(val, "timestamp"):
            return val.timestamp()
        return None

    start_ts = _to_ts(start_date)
    end_ts = _to_ts(end_date)
    if start_ts is None or end_ts is None:
        return gr.update()

    if end_ts <= start_ts:
        return gr.update(
            value=datetime.fromtimestamp(start_ts + 86400.0).strftime("%Y-%m-%d"),
            info="End date MUST be after start date."
        )
    return gr.update(info=None)

def _to_date_str(val):
    """Convert Gradio DateTime (datetime, timestamp, or str) to YYYY-MM-DD."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return datetime.fromtimestamp(val).strftime("%Y-%m-%d")
    if hasattr(val, "strftime"):
        return val.strftime("%Y-%m-%d")
    if isinstance(val, str) and len(val) >= 10:
        return val[:10]
    return str(val) if val else None


def run_simple_inference(experiment_name, run_id):
    """
    Run inference on the selected model with streaming console output.
    - Start date: day after the model's training end_date
    - End date: end of dataset
    Yields output progressively, then final status.
    """
    if not run_id:
        yield "Error: Select a run first."
        return
    if not experiment_name:
        yield "Error: Experiment name is required."
        return

    try:
        project_root = Path(__file__).parent.parent
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


def _clean_numeric_columns(df, skip_columns=None):
    """Remove non-numeric characters ($, commas, spaces, etc.) from price/numeric columns.
    Skips datetime/timestamp columns to avoid corrupting date values.
    skip_columns: optional set/list of column names to never clean (e.g. timestamp column)."""
    import re
    _datetime_cols = {"timestamp", "date", "datetime", "time", "dt", "created_at", "updated_at"}
    skip = set(skip_columns) if skip_columns else set()
    skip.update(_datetime_cols)
    for col in df.columns:
        if col in skip or col.lower().strip() in _datetime_cols:
            continue
        if df[col].dtype == object or df[col].dtype.name == "string":
            # Strip $, commas, spaces; keep digits, decimal point, minus
            def clean_val(x):
                if pd.isna(x):
                    return x
                s = str(x).strip()
                s = re.sub(r"[^\d.\-eE]", "", s)
                return s if s else None
            df[col] = df[col].apply(clean_val)
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_csv_prices(csv_file, timestamp_column=None):
    """Clean non-numeric characters from price columns in uploaded CSV. Returns (cleaned_path, preview, status).
    timestamp_column: column name to skip (datetime values must not be cleaned)."""
    if csv_file is None:
        return None, None, "Upload a CSV file first."
    try:
        path = csv_file if isinstance(csv_file, str) else (getattr(csv_file, "name", None) or str(csv_file))
        df = pd.read_csv(path)
        skip = [timestamp_column] if timestamp_column else None
        df = _clean_numeric_columns(df, skip_columns=skip)
        project_root = Path(__file__).parent.parent
        work_dir = project_root / "temp" / "frontend_custom_inference"
        work_dir.mkdir(parents=True, exist_ok=True)
        out_path = work_dir / "cleaned_upload.csv"
        df.to_csv(out_path, index=False)
        return str(out_path), df.head(5), "Cleaned: removed $, commas, and other non-numeric characters from price columns."
    except Exception as e:
        return None, None, f"Error cleaning: {e}"


def compute_metrics_and_plot_from_csv(csv_path, pred_horizon=1, target="close"):
    """
    Compute MAE, MSE, MDA from inference CSV and create a plot.
    Returns (status, fig, mae, mse, mda).
    """
    if not csv_path or not Path(csv_path).exists():
        return "No inference file found.", None, None, None, None
    try:
        pred_horizon = int(pred_horizon) if pred_horizon else 1
        df = pd.read_csv(csv_path)
        pred_cols = [c for c in df.columns if "predicted" in c.lower()]
        pred_len = len(pred_cols)
        if pred_len == 0:
            return "No prediction columns found in inference output.", None, None, None, None
        if pred_horizon > pred_len:
            pred_horizon = 1

        # Compute MAE, MSE, MDA
        pred_col = f"{target}_predicted_{pred_horizon}"
        if pred_col not in df.columns:
            pred_col = next((c for c in pred_cols if str(pred_horizon) in c), pred_cols[0])

        mae_vals, mse_vals, mda_vals = [], [], []
        for i in range(len(df) - pred_horizon):
            row = df.iloc[i]
            if pd.isna(row.get(pred_col, np.nan)):
                continue
            next_row = df.iloc[i + pred_horizon]
            if pd.notna(next_row.get(target, np.nan)):
                err = row[pred_col] - next_row[target]
                mae_vals.append(abs(err))
                mse_vals.append(err ** 2)
                current_actual = row.get(target, row[pred_col])
                pred_dir = row[pred_col] - current_actual
                actual_dir = next_row[target] - current_actual
                mda_vals.append((pred_dir * actual_dir) > 0)

        mae = float(np.mean(mae_vals)) if mae_vals else None
        mse = float(np.mean(mse_vals)) if mse_vals else None
        mda = float(np.mean(mda_vals)) if mda_vals else None

        # Normalize OHLCV column names for plotting (inference output may use different casing)
        for k, v in [("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close"), ("Volume", "volume")]:
            if k in df.columns and v not in df.columns:
                df = df.rename(columns={k: v})

        # Build plot
        date_col = "timestamp" if "timestamp" in df.columns else ("date" if "date" in df.columns else None)
        if date_col:
            df = df.copy()
            if df[date_col].dtype in ["int64", "float64"]:
                df[date_col] = pd.to_datetime(df[date_col], unit="s")
            else:
                df[date_col] = pd.to_datetime(df[date_col])

        fig = None
        if date_col:
            fig = go.Figure()
            has_ohlcv = all(c in df.columns for c in ["open", "high", "low", "close"])
            if has_ohlcv:
                fig.add_trace(go.Candlestick(
                    x=df[date_col], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                    name="OHLCV", increasing_line_color="green", decreasing_line_color="red",
                ))
            elif "close" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df[date_col], y=df["close"], mode="lines", name="Close",
                    line=dict(color="black", width=1),
                ))
            elif target in df.columns:
                fig.add_trace(go.Scatter(
                    x=df[date_col], y=df[target], mode="lines", name=target.capitalize(),
                    line=dict(color="black", width=1),
                ))
            # Prediction: pred[i] forecasts close at row i+pred_horizon — plot it at that date
            if pred_col in df.columns:
                x_pred = df[date_col].iloc[pred_horizon:].values
                y_pred = df[pred_col].iloc[:-pred_horizon].values
                valid = np.isfinite(y_pred)
                if valid.any():
                    fig.add_trace(go.Scatter(
                        x=x_pred[valid], y=y_pred[valid], mode="lines",
                        name=f"Prediction ({pred_horizon} step ahead)",
                        line=dict(color="blue", width=2),
                    ))
            fig.update_layout(
                title="Custom Inference Results",
                xaxis_title="Date", yaxis_title="Price",
                xaxis_rangeslider_visible=False, template="plotly_white", width=1000, height=600,
            )

        status = f"Inference: {len(df)} rows, {pred_len} prediction horizon(s)."
        return status, fig, mae, mse, mda
    except Exception as e:
        return f"Error computing metrics: {e}", None, None, None, None


def _timestamp_to_unix(series, format=None):
    """Convert a pandas Series to Unix seconds (int64). Handles numeric, datetime strings, etc.

    Args:
        series: pandas Series with timestamp values
        format: Optional strftime format string (e.g. '%Y-%m-%d', '%Y-%m-%d %H:%M:%S').
                Use 'unix' to force numeric Unix timestamp parsing.
                When provided for string data, uses this format for parsing.
    """
    # Explicit unix/numeric request
    if format and str(format).strip().lower() == "unix":
        vals = pd.to_numeric(series, errors="coerce")
        if vals.isna().all():
            raise ValueError(
                "Format is 'unix' but column could not be parsed as numbers. "
                f"Sample: {series.dropna().head(3).tolist()}"
            )
        valid = vals.dropna()
        if len(valid) > 0 and valid.abs().max() > 1e12:
            vals = vals / 1000  # milliseconds to seconds
        return vals.astype("int64")

    # Try numeric conversion first (handles int64, float64, and object dtype with numeric strings)
    try:
        vals = pd.to_numeric(series, errors="coerce")
        valid_mask = vals.notna()
        if valid_mask.any():
            valid = vals[valid_mask]
            max_abs = valid.abs().max()
            if max_abs >= 1e8:  # Likely Unix timestamp (seconds or ms)
                if max_abs > 1e12:
                    vals = vals / 1000  # milliseconds to seconds
                # Only use numeric path if most values parsed (avoid partial success)
                if valid_mask.all():
                    return vals.astype("int64")
    except (ValueError, TypeError):
        pass

    # Parse as datetime - try format first if provided
    fmt = str(format).strip() if format else None
    dt = None
    if fmt:
        dt = pd.to_datetime(series, format=fmt, errors="coerce")
    if dt is None or (dt.isna().all() if dt is not None else True):
        # Fallback: infer without format (handles ISO, common formats)
        dt = pd.to_datetime(series, errors="coerce")
    if dt is None or dt.isna().all():
        # Try format='mixed' for pandas 2.0+ (handles varying formats in same column)
        try:
            dt = pd.to_datetime(series, format="mixed", errors="coerce")
        except Exception:
            pass
    if dt is None or dt.isna().all():
        # Last resort: try with unit='s' for numeric strings
        try:
            numeric = pd.to_numeric(series, errors="coerce")
            if not numeric.isna().all():
                dt = pd.to_datetime(numeric, unit="s", errors="coerce")
        except (ValueError, TypeError):
            pass

    if dt is None or dt.isna().all():
        sample = series.dropna().head(3).tolist()
        raise ValueError(
            f"Could not parse timestamp column as dates or Unix values. "
            f"Sample values: {sample}. "
            f"Dtype: {series.dtype}. "
            f"Try specifying a format (e.g. %Y-%m-%d or %Y-%m-%d %H:%M:%S)."
        )
    return (dt.astype("int64") // 10**9).astype("int64")


def run_custom_inference(experiment_name, run_id, csv_file, timestamp_column, timestamp_format, target_column, ohlcv_columns=None, pred_horizon=1):
    """
    Run inference on an uploaded CSV file using the selected model.
    Saves results to custom_inference.csv in the project root.
    Does not log to MLflow.
    """
    err = lambda msg: (msg, None, None, None, None)
    try:
        if not run_id:
            return err("Error: Select a model (run) first.")
        if not experiment_name:
            return err("Error: Experiment name is required.")
        if csv_file is None:
            return err("Error: Upload a CSV file first.")
        if not timestamp_column:
            return err("Error: Select the timestamp column.")
        if not target_column:
            return err("Error: Select the target column.")

        project_root = Path(__file__).parent.parent
        work_dir_path = project_root / "temp" / "frontend_custom_inference"
        custom_output_path = project_root / "custom_inference.csv"

        # Save uploaded file to temp and prepare data
        import tempfile
        import shutil
        file_path = csv_file if isinstance(csv_file, str) else (getattr(csv_file, "name", None) or str(csv_file))
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            shutil.copy(file_path, tmp.name)
            uploaded_path = tmp.name

        try:
            df = pd.read_csv(uploaded_path)

            # Clean non-numeric characters from price columns ($, commas, etc.); skip datetime column
            df = _clean_numeric_columns(df, skip_columns=[timestamp_column])

            # Validate required columns
            if timestamp_column not in df.columns:
                return err(f"Error: Timestamp column '{timestamp_column}' not found in CSV. Columns: {list(df.columns)}")
            if target_column not in df.columns:
                return err(f"Error: Target column '{target_column}' not found in CSV. Columns: {list(df.columns)}")

            # Convert timestamp column to Unix seconds and rename to 'timestamp'
            try:
                df["timestamp"] = _timestamp_to_unix(df[timestamp_column], format=timestamp_format)
            except Exception as e:
                return err(f"Error: Could not convert timestamp column to Unix format: {e}")
            if timestamp_column != "timestamp":
                df = df.drop(columns=[timestamp_column])

            # Get model's expected target from MLflow
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = mlflow.tracking.MlflowClient()
            exp = client.get_experiment_by_name(experiment_name)
            if exp is None:
                return err(f"Error: Experiment '{experiment_name}' not found.")
            runs = client.search_runs([exp.experiment_id], f"run_id = '{run_id}'", max_results=1)
            if not runs:
                runs = client.search_runs([exp.experiment_id], f"tags.mlflow.runName = '{run_id}'", max_results=1)
            if not runs:
                return err(f"Error: Run '{run_id}' not found in experiment.")
            model_target = runs[0].data.params.get("target", "close")

            # Rename target column if needed to match model's expected target
            if target_column != model_target:
                df = df.rename(columns={target_column: model_target})

            # Apply OHLCV column mapping (user-selected or auto-guessed)
            if ohlcv_columns:
                for std_name, user_col in ohlcv_columns.items():
                    if user_col and str(user_col).strip() and user_col in df.columns and user_col != std_name:
                        df = df.rename(columns={user_col: std_name})

            # Fallback: normalize common variants (Open->open, etc.)
            ohlcv_map = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
            rename_map = {k: v for k, v in ohlcv_map.items() if k in df.columns and v not in df.columns}
            if rename_map:
                df = df.rename(columns=rename_map)

            # Ensure 'close' exists for inference (model typically expects it)
            if "close" not in df.columns:
                for alt in ["Close", "close_price", "price"]:
                    if alt in df.columns:
                        df = df.rename(columns={alt: "close"})
                        break
                if "close" not in df.columns:
                    return err(
                        "Error: Data must have a 'close' column. Select it in OHLCV mapping or ensure your CSV has it. "
                        f"Current columns: {list(df.columns)}"
                    )

            # Sort by date ascending (oldest first, most recent at bottom)
            df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)

            # Save prepared data
            work_dir_path.mkdir(parents=True, exist_ok=True)
            data_path = work_dir_path / "uploaded_data.csv"
            df.to_csv(data_path, index=False)

            # Create WorkDir and DataManager
            args = HpoArgs(
                parse_cli=False,
                model_name=run_id,
                experiment_name=experiment_name,
                granularity="daily",
                aggregate=1,
                inf_start=None,
                inf_end=None,
                data_path=str(data_path),
            )
            dataset_path = project_root / "dataset" / "candles"
            work_dir = WorkDir(args, work_dir=work_dir_path, dataset_path=dataset_path)
            work_dir.create_work_dir()

            data_manager = DataManager(work_dir)
            pipeline_runner = PipelineRunner(work_dir)
            pipeline_runner.run_inference(experiment_name, run_id, skip_mlflow_logging=True)

            # Copy output to custom_inference.csv and compute metrics/plot
            # Prefer ohlcv_inference.csv (has OHLCV + close_predicted_* for returns models)
            ohlcv_path = work_dir.get_ohlcv_inferenced_path()
            inferenced_path = ohlcv_path if ohlcv_path.exists() else work_dir.get_inferenced_path()
            if inferenced_path.exists():
                shutil.copy(inferenced_path, custom_output_path)
                status, fig, mae, mse, mda = compute_metrics_and_plot_from_csv(
                    str(custom_output_path), pred_horizon=pred_horizon, target=model_target
                )
                full_status = f"Inference completed successfully!\nResults saved to: {custom_output_path}\n\n{status}"
                return full_status, fig, mae, mse, mda
            return f"Inference completed but output not found at {inferenced_path}", None, None, None, None
        finally:
            Path(uploaded_path).unlink(missing_ok=True)
    except Exception as e:
        return (f"Error running custom inference: {str(e)}\n\nTraceback:\n{traceback.format_exc()}", None, None, None, None)


def run_inference_handler(model_name, experiment_name, custom_dataset_path, granularity, aggregate, start_date, end_date, save_path=None):
    """Run inference using PipelineRunner."""
    try:
        if not model_name:
            return "Error: Model Name (Run ID) is required"
        if not experiment_name:
            return "Error: Experiment Name is required"

        project_root = Path(__file__).parent.parent

        # Resolve custom data path relative to project root if provided
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

        # Use project-root paths so inference works regardless of cwd
        dataset_path = project_root / "dataset" / "candles"
        work_dir_path = project_root / "temp" / "frontend_inference"

        work_dir = WorkDir(args, work_dir=work_dir_path, dataset_path=dataset_path)
        work_dir.create_work_dir()

        # Validate data file exists before proceeding
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
        return f"Error running inference: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"


def _fetch_metrics_from_mlflow(client, run_id, pred_horizon=1):
    """
    Fetch MAE, MSE, MDA from MLflow artifacts for a specific prediction horizon.
    Returns (mae, mse, mda) for inf_pred_{pred_horizon}_*, or (None, None, None) if not found.
    """
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


def load_metrics_from_mlflow(experiment_name, run_id, pred_horizon=1):
    """
    Load MAE, MSE, MDA from MLflow run artifacts for the given prediction horizon.
    Returns (mae, mse, mda) or (None, None, None) on error.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return None, None, None
        runs = client.search_runs(experiment_ids=[exp.experiment_id], filter_string=f"run_id = '{run_id}'", max_results=1)
        if not runs:
            runs = client.search_runs(experiment_ids=[exp.experiment_id], filter_string=f"tags.mlflow.runName = '{run_id}'", max_results=1)
        if not runs:
            return None, None, None
        run_uuid = runs[0].info.run_id
        return _fetch_metrics_from_mlflow(client, run_uuid, pred_horizon)
    except Exception as e:
        logger.debug(f"load_metrics_from_mlflow error: {e}")
        return None, None, None


def _get_run_artifact_status(client, run_id):
    """
    Check if inference and backtesting were run based on MLflow artifacts.
    Returns (has_inference: bool, has_backtest: bool).
    """
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


def check_and_plot_mlflow_inference(experiment_name, run_id, pred_horizon=1):
    """Check if MLflow has inference data for the given run and plot it as candlestick with prediction overlay.

    Returns:
        (status, plot, mae, mse, mda) - metrics are computed when data is loaded.
    """
    logger.debug(f"Checking and plotting MLflow inference for run: {run_id} in experiment: {experiment_name}")
    if not run_id:
        return "Error: Please enter a Run ID", None, None, None, None
    
    try:
        pred_horizon = int(pred_horizon) if pred_horizon else 1
    except (ValueError, TypeError):
        pred_horizon = 1
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.debug(f"Getting MLflow client for experiment: {experiment_name}")
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    except Exception as e:
        return f"Error: {str(e)}", None, None, None, None

    try:
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return f"Error: Experiment '{experiment_name}' not found", None, None, None, None
        # Try run_id (UUID) first, then run name
        runs = client.search_runs(experiment_ids=[exp.experiment_id], filter_string=f"run_id = '{run_id}'", max_results=1)
        if not runs:
            runs = client.search_runs(experiment_ids=[exp.experiment_id], filter_string=f"tags.mlflow.runName = '{run_id}'", max_results=1)
    except Exception as e:
        return f"Error searching runs: {str(e)}", None, None, None, None

    if not runs:
        return f"No run found with ID/name '{run_id}' in experiment '{experiment_name}'", None, None, None, None

    # Use actual run UUID for artifact download
    run_id = runs[0].info.run_id

    logger.debug(f"Tracking URI: {mlflow.get_tracking_uri()}")

    try:
        path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="ohlcv_inference.csv",
        )
    except Exception as e:
        return f"Error downloading artifacts: {str(e)}", None, None, None, None

    df = pd.read_csv(path)
    mae, mse, mda = _fetch_metrics_from_mlflow(client, run_id, pred_horizon)
    
    # Determine date column
    date_col = None
    if 'date' in df.columns:
        date_col = 'date'
    elif 'timestamp' in df.columns:
        date_col = 'timestamp'
    
    if date_col:
        # Convert timestamp to datetime
        if df[date_col].dtype in ['int64', 'float64']:
            df[date_col] = pd.to_datetime(df[date_col], unit='s')
        else:
            df[date_col] = pd.to_datetime(df[date_col])
    
    # Find available prediction columns
    pred_cols = [c for c in df.columns if 'predicted' in c.lower()]
    max_horizon = len(pred_cols) if pred_cols else 0
    
    # Check if we have OHLCV data
    has_ohlcv = all(col in df.columns for col in ['open', 'high', 'low', 'close'])
    
    if has_ohlcv and date_col:
        # Create candlestick chart
        fig = go.Figure()
        
        # Add candlestick trace
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
        
        # Add prediction line: pred[i] forecasts close at row i+pred_horizon — plot at that date
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
        # Fallback to line plot if no OHLCV data
        fig = go.Figure()
        
        # Plot close price if available
        if 'close' in df.columns:
            fig.add_trace(go.Scatter(
                x=df[date_col],
                y=df['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='black', width=1)
            ))
        
        # Add prediction line (aligned: pred[i] at date i+pred_horizon)
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
        # No date column, create a simple index plot
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns[:4].tolist()
        fig = px.line(df, y=numeric_cols, title=f"Inference Results for Run: {run_id}")
        fig.update_layout(width=1000, height=600)
        return f"Found inference data for run: {run_id}\nShape: {df.shape[0]} rows, {df.shape[1]} columns", fig, mae, mse, mda


def _backtest_plot_from_results(data_df, trade_log_df, strategy_name):
    """Create plotly figure: candlestick + buy/sell markers from backtest results."""
    if data_df is None or data_df.empty:
        return None
    dates = data_df.index.tolist() if hasattr(data_df.index, "tolist") else list(range(len(data_df)))
    n = len(dates)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=dates,
        open=data_df["open"].values,
        high=data_df["high"].values,
        low=data_df["low"].values,
        close=data_df["close"].values,
        name="OHLC",
        increasing_line_color="green",
        decreasing_line_color="red",
    ))

    if trade_log_df is not None and not trade_log_df.empty and "bar_open" in trade_log_df.columns:
        buy_dates, buy_prices = [], []
        for _, row in trade_log_df.iterrows():
            b = int(row["bar_open"])
            if 0 <= b < n:
                buy_dates.append(dates[b])
                buy_prices.append(float(row.get("entry_price", row.get("close", 0))))
        if buy_dates:
            fig.add_trace(go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode="markers",
                name="Buy",
                marker=dict(symbol="triangle-up", size=12, color="lime", line=dict(width=1, color="darkgreen")),
            ))
        if "bar_close" in trade_log_df.columns:
            sell_dates, sell_prices = [], []
            for _, row in trade_log_df.iterrows():
                bc = row.get("bar_close")
                if pd.notna(bc):
                    b = int(bc)
                    if 0 <= b < n:
                        sell_dates.append(dates[b])
                        sell_prices.append(float(row.get("exit_price", row.get("close", 0))))
            if sell_dates:
                fig.add_trace(go.Scatter(
                    x=sell_dates,
                    y=sell_prices,
                    mode="markers",
                    name="Sell",
                    marker=dict(symbol="triangle-down", size=12, color="red", line=dict(width=1, color="darkred")),
                ))

    fig.update_layout(
        title=f"Backtest: {strategy_name} (Buys & Sells)",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
    )
    return fig


def fetch_summary_table_from_mlflow(experiment_name, run_id):
    """
    Fetch backtest summary_table from MLflow run artifacts.
    Returns (summary_df, error_msg). On success, error_msg is None. summary_df is pandas DataFrame or None.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return None, f"Experiment '{experiment_name}' not found"
        runs = client.search_runs(experiment_ids=[exp.experiment_id], filter_string=f"run_id = '{run_id}'", max_results=1)
        if not runs:
            runs = client.search_runs(experiment_ids=[exp.experiment_id], filter_string=f"tags.mlflow.runName = '{run_id}'", max_results=1)
        if not runs:
            return None, "Run not found"
        run_uuid = runs[0].info.run_id

        for artifact_path in ["summary_table.csv", "summary_table"]:
            try:
                downloaded = mlflow.artifacts.download_artifacts(run_id=run_uuid, artifact_path=artifact_path)
                path = Path(downloaded)
                if path.is_dir():
                    csv_path = path / "summary_table.csv"
                else:
                    csv_path = path
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    return df, None
            except Exception:
                continue
        return None, None  # No summary_table found, no error
    except Exception as e:
        return None, str(e)


def run_backtest(experiment_name, run_id, strategy, initial_capital, start_date, end_date, threshold, log_to_mlflow=False):
    """Run backtest on inference data from MLflow. Returns (plot, total_return, sharpe, max_dd, win_rate, num_trades, profit_factor)."""
    import shutil
    import tempfile

    null_result = (None, None, None, None, None, None, None, None, None)

    try:
        if not run_id:
            return (None, "Error: Run ID is required", None, None, None, None, None, None, None)
        if not experiment_name:
            return (None, "Error: Experiment Name is required", None, None, None, None, None, None, None)

        from backtesting.backtest import BacktestRunner, STRATEGIES
        from backtesting.utils import load_and_prepare_data

        if strategy and strategy not in STRATEGIES:
            return (None, f"Error: Strategy '{strategy}' not found", None, None, None, None, None, None, None)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return (None, f"Error: Experiment '{experiment_name}' not found", None, None, None, None, None, None, None)
        runs = client.search_runs(experiment_ids=[exp.experiment_id], filter_string=f"run_id = '{run_id}'", max_results=1)
        if not runs:
            runs = client.search_runs(experiment_ids=[exp.experiment_id], filter_string=f"tags.mlflow.runName = '{run_id}'", max_results=1)
        if not runs:
            return (None, f"Error: No run found with ID/name '{run_id}'", None, None, None, None, None, None, None)

        run_uuid = runs[0].info.run_id
        try:
            downloaded = mlflow.artifacts.download_artifacts(run_id=run_uuid, artifact_path="ohlcv_inference.csv")
            artifact_path = Path(downloaded)
            if artifact_path.is_dir():
                artifact_path = artifact_path / "ohlcv_inference.csv"
            if not artifact_path.exists():
                return (None, "Error: ohlcv_inference.csv not found in run artifacts", None, None, None, None, None, None, None)
        except Exception as e:
            return (None, f"Error: Could not download ohlcv_inference.csv: {e}", None, None, None, None, None, None, None)

        runner = BacktestRunner(
            str(artifact_path),
            cash=float(initial_capital or 10000),
            commission=0.001,
            pipeline=True,
        )

        if strategy:
            runner.run_strategy(strategy)
        else:
            runner.run_all_strategies()

        summary_df = runner.create_summary_table()
        if summary_df is None or summary_df.empty:
            return (None, "Backtest completed but no results to display.", None, None, None, None, None, None, None)

        strat_name = str(summary_df.iloc[0]["Strategy"])
        row = summary_df.iloc[0]
        total_return = float(row.get("Total Return (%)", 0))
        sharpe = float(row.get("Sharpe Ratio", 0))
        max_dd = float(row.get("Max Drawdown (%)", 0))
        win_rate = float(row.get("Win Rate (%)", 0))
        num_trades = int(row.get("Total Trades", 0))

        trade_log_df = runner.results[strat_name]["analyzers"].get("trade_log", pd.DataFrame())
        if not isinstance(trade_log_df, pd.DataFrame):
            trade_log_df = pd.DataFrame()
        if not trade_log_df.empty and "pnl" in trade_log_df.columns:
            gross_profit = trade_log_df[trade_log_df["pnl"] > 0]["pnl"].sum()
            gross_loss = abs(trade_log_df[trade_log_df["pnl"] < 0]["pnl"].sum())
            profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
        else:
            profit_factor = 0.0

        fig = _backtest_plot_from_results(runner.data, trade_log_df, strat_name)
        summary_str = summary_df.to_string(index=False, float_format="%.2f")

        if log_to_mlflow:
            try:
                import tempfile
                tmpdir = Path(tempfile.mkdtemp())
                summary_path = tmpdir / "summary_table.csv"
                summary_df.to_csv(summary_path, index=False)
                client.log_artifact(run_id=run_uuid, local_path=str(summary_path), artifact_path="summary_table")
                summary_path.unlink(missing_ok=True)
                tmpdir.rmdir()
            except Exception as log_err:
                logger.debug(f"Failed to log summary_table to MLflow: {log_err}")

        return (fig, summary_str, total_return, sharpe, max_dd, win_rate, num_trades, profit_factor, summary_df)
    except Exception as e:
        return (None, f"Error: {str(e)}\n{traceback.format_exc()}", None, None, None, None, None, None, None)
