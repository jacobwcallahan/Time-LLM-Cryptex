import gradio as gr
from datetime import timedelta
from datetime import datetime
import mlflow
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tempfile
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

    if start_date >= end_date:
        return gr.update(
            value=datetime.fromtimestamp(end_date - 86400.0).strftime("%Y-%m-%d"),
            info="Start date MUST be before end date."
        )
    return gr.update(info=None)

def end_after_start(end_date, start_date):
    """Checks if the end date is after the start date.
    
    Must be used in the change event of the end date."""
    if end_date is None or start_date is None:
        return gr.update()

    if end_date <= start_date:
        return gr.update(
            value=datetime.fromtimestamp(start_date + 86400.0).strftime("%Y-%m-%d"),
            info="End date MUST be after start date."
        )

def run_inference_handler(model_name, experiment_name, custom_dataset_path, granularity, aggregate, start_date, end_date, save_path = None):
    """Run inference using run_inf_and_backtest with backtest=False"""
    try:
        if not model_name:
            return "Error: Model Name (Run ID) is required"
        if not experiment_name:
            return "Error: Experiment Name is required"
        
        args = HpoArgs(
            model_name=model_name,
            experiment_name=experiment_name,
            granularity=granularity,
            aggregate=int(aggregate),
            start_date=start_date,
            end_date=end_date,
            custom_dataset_path=custom_dataset_path,
        )
        work_dir = WorkDir(args)
        # Convert dates to string format if provided
        start_str = start_date if start_date else None
        end_str = end_date if end_date else None

        data_manager = DataManager(args, work_dir)
        pipeline_runner = PipelineRunner(work_dir)
        pipeline_runner.run_inference(data_manager, experiment_name, model_name)

        return f"Inference completed successfully!\nResults saved to: {save_path}"
    except Exception as e:
        return f"Error running inference: {str(e)}"


def check_and_plot_mlflow_inference(experiment_name, run_id, pred_horizon=1):
    """Check if MLflow has inference data for the given run and plot it as candlestick with prediction overlay.
    
    Args:
        run_id: MLflow run ID
        pred_horizon: Prediction horizon to plot (1 = 1 step ahead, 2 = 2 steps ahead, etc.)
    """
    logger.debug(f"Checking and plotting MLflow inference for run: {run_id} in experiment: {experiment_name}")
    if not run_id:
        return "Error: Please enter a Run ID", None
    
    try:
        pred_horizon = int(pred_horizon) if pred_horizon else 1
    except (ValueError, TypeError):
        pred_horizon = 1
    
    try:
        MLFLOW_TRACKING_URI = "http://192.168.1.106:5005"
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.debug(f"Getting MLflow client for experiment: {experiment_name}")
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    except Exception as e:
        return f"Error: {str(e)}", None

    runs = client.search_runs(experiment_ids=[experiment_name], filter_string=f"run_id = '{run_id}'", max_results=1)

    logger.debug(f"Runs: {runs}")
        
    return None, None

    logger.debug(f"Tracking URI: {mlflow.get_tracking_uri()}")


    path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="ohlcv_inference.csv"   # relative inside artifact root
    )

    df = pd.read_csv(path)
    
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
            template="plotly_white"
        )
        
        status = f"Found inference data for run: {run_id}\nShape: {df.shape[0]} rows, {df.shape[1]} columns\nAvailable prediction horizons: 1-{max_horizon}" if max_horizon > 0 else f"Found inference data for run: {run_id}\nShape: {df.shape[0]} rows, {df.shape[1]} columns"
        return status, fig
    
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
            template="plotly_white"
        )
        
        status = f"Found inference data for run: {run_id}\nShape: {df.shape[0]} rows, {df.shape[1]} columns\nAvailable prediction horizons: 1-{max_horizon}" if max_horizon > 0 else f"Found inference data for run: {run_id}\nShape: {df.shape[0]} rows, {df.shape[1]} columns"
        return status, fig
    
    else:
        # No date column, create a simple index plot
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns[:4].tolist()
        fig = px.line(df, y=numeric_cols, title=f"Inference Results for Run: {run_id}")
        return f"Found inference data for run: {run_id}\nShape: {df.shape[0]} rows, {df.shape[1]} columns", fig


def run_backtest(inference_path, strategy, initial_capital, start_date, end_date, threshold):
    """Run backtest on inference results."""
    return "Backtest not implemented yet"
    # TODO: Implement actual backtest logic
    # return bt.main({
    #     'data': inference_path,
    #     'strategy': strategy,
    #     'initial_capital': initial_capital,
    #     'start_date': start_date,
    #     'end_date': end_date,
    #     'threshold': threshold
    # })
