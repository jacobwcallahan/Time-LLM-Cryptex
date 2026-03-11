"""Custom Inference tab helpers - CSV cleaning, inference, metrics, plotting."""

import re
import shutil
import tempfile
import traceback
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from hpo_core.PipelineRunner import PipelineRunner
from hpo_core.DataManager import DataManager
from hpo_core.HpoArgs import HpoArgs
from hpo_core.WorkDir import WorkDir

from .common import MLFLOW_TRACKING_URI


def _project_root():
    return Path(__file__).resolve().parent.parent.parent


def _clean_numeric_columns(df, skip_columns=None):
    """Remove non-numeric characters ($, commas, spaces, etc.) from price/numeric columns.
    Skips datetime/timestamp columns to avoid corrupting date values."""
    _datetime_cols = {"timestamp", "date", "datetime", "time", "dt", "created_at", "updated_at"}
    skip = set(skip_columns) if skip_columns else set()
    skip.update(_datetime_cols)
    for col in df.columns:
        if col in skip or col.lower().strip() in _datetime_cols:
            continue
        if df[col].dtype == object or df[col].dtype.name == "string":
            def clean_val(x):
                if pd.isna(x):
                    return x
                s = str(x).strip()
                s = re.sub(r"[^\d.\-eE]", "", s)
                return s if s else None
            df[col] = df[col].apply(clean_val)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_csv_prices(csv_file, timestamp_column=None):
    """Clean non-numeric characters from price columns in uploaded CSV. Returns (cleaned_path, preview, status)."""
    if csv_file is None:
        return None, None, "Upload a CSV file first."
    try:
        path = csv_file if isinstance(csv_file, str) else (getattr(csv_file, "name", None) or str(csv_file))
        df = pd.read_csv(path)
        skip = [timestamp_column] if timestamp_column else None
        df = _clean_numeric_columns(df, skip_columns=skip)
        project_root = _project_root()
        work_dir = project_root / "temp" / "frontend_custom_inference"
        work_dir.mkdir(parents=True, exist_ok=True)
        out_path = work_dir / "cleaned_upload.csv"
        df.to_csv(out_path, index=False)
        return str(out_path), df.head(5), "Cleaned: removed $, commas, and other non-numeric characters from price columns."
    except Exception as e:
        return None, None, f"Error cleaning: {e}"


def compute_metrics_and_plot_from_csv(csv_path, pred_horizon=1, target="close"):
    """Compute MAE, MSE, MDA from inference CSV and create a plot. Returns (status, fig, mae, mse, mda)."""
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

        for k, v in [("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close"), ("Volume", "volume")]:
            if k in df.columns and v not in df.columns:
                df = df.rename(columns={k: v})

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
    """Convert a pandas Series to Unix seconds (int64)."""
    if format and str(format).strip().lower() == "unix":
        vals = pd.to_numeric(series, errors="coerce")
        if vals.isna().all():
            raise ValueError(
                "Format is 'unix' but column could not be parsed as numbers. "
                f"Sample: {series.dropna().head(3).tolist()}"
            )
        valid = vals.dropna()
        if len(valid) > 0 and valid.abs().max() > 1e12:
            vals = vals / 1000
        return vals.astype("int64")

    try:
        vals = pd.to_numeric(series, errors="coerce")
        valid_mask = vals.notna()
        if valid_mask.any():
            valid = vals[valid_mask]
            max_abs = valid.abs().max()
            if max_abs >= 1e8:
                if max_abs > 1e12:
                    vals = vals / 1000
                if valid_mask.all():
                    return vals.astype("int64")
    except (ValueError, TypeError):
        pass

    fmt = str(format).strip() if format else None
    dt = None
    if fmt:
        dt = pd.to_datetime(series, format=fmt, errors="coerce")
    if dt is None or (dt.isna().all() if dt is not None else True):
        dt = pd.to_datetime(series, errors="coerce")
    if dt is None or dt.isna().all():
        try:
            dt = pd.to_datetime(series, format="mixed", errors="coerce")
        except Exception:
            pass
    if dt is None or dt.isna().all():
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
            f"Sample values: {sample}. Dtype: {series.dtype}. "
            f"Try specifying a format (e.g. %Y-%m-%d or %Y-%m-%d %H:%M:%S)."
        )
    return (dt.astype("int64") // 10**9).astype("int64")


def run_custom_inference(experiment_name, run_id, csv_file, timestamp_column, timestamp_format, target_column, ohlcv_columns=None, pred_horizon=1):
    """Run inference on an uploaded CSV file using the selected model. Saves to custom_inference.csv."""
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

        project_root = _project_root()
        work_dir_path = project_root / "temp" / "frontend_custom_inference"
        custom_output_path = project_root / "custom_inference.csv"

        file_path = csv_file if isinstance(csv_file, str) else (getattr(csv_file, "name", None) or str(csv_file))
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            shutil.copy(file_path, tmp.name)
            uploaded_path = tmp.name

        try:
            df = pd.read_csv(uploaded_path)
            df = _clean_numeric_columns(df, skip_columns=[timestamp_column])

            if timestamp_column not in df.columns:
                return err(f"Error: Timestamp column '{timestamp_column}' not found in CSV. Columns: {list(df.columns)}")
            if target_column not in df.columns:
                return err(f"Error: Target column '{target_column}' not found in CSV. Columns: {list(df.columns)}")

            try:
                df["timestamp"] = _timestamp_to_unix(df[timestamp_column], format=timestamp_format)
            except Exception as e:
                return err(f"Error: Could not convert timestamp column to Unix format: {e}")
            if timestamp_column != "timestamp":
                df = df.drop(columns=[timestamp_column])

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = mlflow.tracking.MlflowClient()
            exp = client.get_experiment_by_name(experiment_name)
            if exp is None:
                return err(f"Error: Experiment '{experiment_name}' not found.")
            runs = client.search_runs([exp.experiment_id], f"run_id = '{run_id}'", max_results=1)
            if not runs:
                runs = client.search_runs([exp.experiment_id], f"tags.mlflow.runName = '{run_id}'", max_results=1)
            if not runs:
                return err("Error: Run '{run_id}' not found in experiment.")
            model_target = runs[0].data.params.get("target", "close")

            if target_column != model_target:
                df = df.rename(columns={target_column: model_target})

            if ohlcv_columns:
                for std_name, user_col in ohlcv_columns.items():
                    if user_col and str(user_col).strip() and user_col in df.columns and user_col != std_name:
                        df = df.rename(columns={user_col: std_name})

            ohlcv_map = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
            rename_map = {k: v for k, v in ohlcv_map.items() if k in df.columns and v not in df.columns}
            if rename_map:
                df = df.rename(columns=rename_map)

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

            df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)

            work_dir_path.mkdir(parents=True, exist_ok=True)
            data_path = work_dir_path / "uploaded_data.csv"
            df.to_csv(data_path, index=False)

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
