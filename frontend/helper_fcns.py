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
from run_inf_and_backtest import run_inference_and_backtest as run_inf

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

def run_inference_handler(model_name, experiment_name, custom_dataset_path, granularity, aggregate, start_date, end_date, save_path):
    """Run inference using run_inf_and_backtest with backtest=False"""
    try:
        if not model_name:
            return "Error: Model Name (Run ID) is required"
        if not experiment_name:
            return "Error: Experiment Name is required"
        
        # Convert dates to string format if provided
        start_str = start_date if start_date else None
        end_str = end_date if end_date else None
        
        run_inf(
            model_name=model_name,
            experiment_name=experiment_name,
            granularity=granularity,
            aggregate=int(aggregate),
            start_date=start_str,
            end_date=end_str,
            custom_dataset_path=custom_dataset_path,
            save_path=save_path,
            backtest=False
        )
        return f"Inference completed successfully!\nResults saved to: {save_path}"
    except Exception as e:
        return f"Error running inference: {str(e)}"


def check_and_plot_mlflow_inference(run_id, pred_horizon=1):
    """Check if MLflow has inference data for the given run and plot it as candlestick with prediction overlay.
    
    Args:
        run_id: MLflow run ID
        pred_horizon: Prediction horizon to plot (1 = 1 step ahead, 2 = 2 steps ahead, etc.)
    """
    if not run_id:
        return "Error: Please enter a Run ID", None
    
    try:
        pred_horizon = int(pred_horizon) if pred_horizon else 1
    except (ValueError, TypeError):
        pred_horizon = 1
    
    try:
        client = mlflow.tracking.MlflowClient()
        
        # List artifacts for the run
        artifacts = client.list_artifacts(run_id, path="inference")
        
        if not artifacts:
            return f"No inference data found for run: {run_id}", None
        
        # Find the inference CSV file (prefer OHLCV file)
        inference_file = None
        for artifact in artifacts:
            if 'inference' in artifact.path.lower() and artifact.path.endswith('.csv') or artifact.path.contains('ohlcv'):
                inference_file = artifact.path
                break
        
        # Fallback to any CSV if no OHLCV file found
        if not inference_file:
            for artifact in artifacts:
                if artifact.path.endswith('.csv'):
                    inference_file = artifact.path
                    break
        
        if not inference_file:
            return f"No CSV inference file found for run: {run_id}", None
        
        # Download the artifact to a temp directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = client.download_artifacts(run_id, inference_file, tmp_dir)
            df = pd.read_csv(local_path)
        
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
        
    except mlflow.exceptions.MlflowException as e:
        return f"MLflow Error: {str(e)}", None
    except Exception as e:
        return f"Error: {str(e)}", None


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
