import gradio as gr
from pathlib import Path
from datetime import timedelta
from datetime import datetime
import mlflow
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import shutil
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
        
        # Add prediction line if available
        pred_col = f'close_predicted_{pred_horizon}'
        if pred_col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[date_col],
                y=df[pred_col],
                mode='lines',
                name=f'Prediction ({pred_horizon} step{"s" if pred_horizon > 1 else ""} ahead)',
                line=dict(color='blue', width=2)
            ))
        else:
            # Try alternative prediction column naming
            alt_pred_cols = [c for c in pred_cols if str(pred_horizon) in c]
            if alt_pred_cols:
                fig.add_trace(go.Scatter(
                    x=df[date_col],
                    y=df[alt_pred_cols[0]],
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
        
        # Add prediction line
        pred_col = f'close_predicted_{pred_horizon}'
        if pred_col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[date_col],
                y=df[pred_col],
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


def run_backtest(experiment_name, run_id, strategy, initial_capital, start_date, end_date, threshold):
    """Run backtest on inference data from MLflow. Returns (plot, total_return, sharpe, max_dd, win_rate, num_trades, profit_factor)."""
    import shutil
    import tempfile

    null_result = (None, None, None, None, None, None, None)

    try:
        if not run_id:
            return (None, "Error: Run ID is required", None, None, None, None, None)
        if not experiment_name:
            return (None, "Error: Experiment Name is required", None, None, None, None, None)

        from backtesting.backtest import BacktestRunner, STRATEGIES
        from backtesting.utils import load_and_prepare_data

        if strategy and strategy not in STRATEGIES:
            return (None, f"Error: Strategy '{strategy}' not found", None, None, None, None, None)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return (None, f"Error: Experiment '{experiment_name}' not found", None, None, None, None, None)
        runs = client.search_runs(experiment_ids=[exp.experiment_id], filter_string=f"run_id = '{run_id}'", max_results=1)
        if not runs:
            runs = client.search_runs(experiment_ids=[exp.experiment_id], filter_string=f"tags.mlflow.runName = '{run_id}'", max_results=1)
        if not runs:
            return (None, f"Error: No run found with ID/name '{run_id}'", None, None, None, None, None)

        run_uuid = runs[0].info.run_id
        try:
            downloaded = mlflow.artifacts.download_artifacts(run_id=run_uuid, artifact_path="ohlcv_inference.csv")
            artifact_path = Path(downloaded)
            if artifact_path.is_dir():
                artifact_path = artifact_path / "ohlcv_inference.csv"
            if not artifact_path.exists():
                return (None, "Error: ohlcv_inference.csv not found in run artifacts", None, None, None, None, None)
        except Exception as e:
            return (None, f"Error: Could not download ohlcv_inference.csv: {e}", None, None, None, None, None)

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
            return (None, "Backtest completed but no results to display.", None, None, None, None, None)

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

        return (fig, summary_str, total_return, sharpe, max_dd, win_rate, num_trades, profit_factor)
    except Exception as e:
        return (None, f"Error: {str(e)}\n{traceback.format_exc()}", None, None, None, None, None)
