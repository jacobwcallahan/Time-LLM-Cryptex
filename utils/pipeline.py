"""
Utility functions for the pipeline of Training to Inference to Backtesting.
As well as functions for the Metrics database.

Functions:
    run_inference: Runs the inference pipeline for the model.
    perform_backtest: Performs the backtest on the inference data.
    get_mda_vals: Gets the MDA values for the inference data.
    get_mse_vals: Gets the MSE values for the inference data.
    get_mae_vals: Gets the MAE values for the inference data.
    create_metrics_json: Creates the metrics JSON for the MLflow run.
    metrics_to_db: Saves the metrics to the database.
    aggregate_data: Aggregates the data to the specified granularity.
    convert_to_returns: Converts the OHLCV data to returns.
    convert_back_to_candlesticks: Converts the returns data back to OHLCV candlesticks.
"""

import subprocess
from datetime import datetime
import pandas as pd
import numpy as np
import mlflow
import os
import sqlite3
import json
from pathlib import Path
import warnings
import mlflow

def run_inference(model_id, 
        mlflow_client,
        experiment_name,
        dataset_path = Path("/mnt/nfs/datasets/"), 
        custom_dataset_path = None,
        granularity = 'daily', 
        aggregate = 1, 
        start_date = None, 
        end_date = None, 
        save_path = None):

    """
    Runs the inference pipeline for the model.

    This function converts the data to returns if the target is returns and converts the data back to candlesticks if the target is OHLCV.
    It also saves the inference data to the save_path.

    returns the path to the OHLCV inference data
    Args:
        model_id: MLflow model id
        mlflow_client: mlflow client
        experiment_name: name of the experiment
        dataset_path: path to the dataset (default: /mnt/nfs/datasets/)
        granularity: granularity of the data (default: daily)
        aggregate: aggregate the data to the specified granularity (default: 1)
        start_date: start date of the data (format: YYYY-MM-DD)
        end_date: end date of the data (format: YYYY-MM-DD)
        save_path: path to save the inference data (default: temp folder)
    
    Returns:
        path to the OHLCV inference data
    """

    os.makedirs("temp", exist_ok=True)

    # Sets the save path for the inference data
    if save_path is None:
        inf_save_path = Path("temp")   # Folder name for the inference data
    else:
        inf_save_path = Path(save_path)


    org_inf_path = Path("temp") / "org_inf_data.csv"  # Path to the orginal data to be inferenced

    dataset_path = Path(dataset_path)

    # Sets the dataset path based on the granularity
    if not custom_dataset_path:
        if granularity.lower() in ['daily', 'd']:
            dataset_path = dataset_path / "candlesticks-D.csv"
        elif granularity.lower() in ['hourly', 'h']:
            dataset_path = dataset_path / "candlesticks-h.csv"
        elif granularity.lower() in ['weekly', 'w']:
            dataset_path = dataset_path / "candlesticks-W.csv"
        elif granularity.lower() in ['minute', 'min']:
            dataset_path = dataset_path / "candlesticks-Min.csv"
    else:
        dataset_path = custom_dataset_path

    inf_data = pd.read_csv(dataset_path)
    
    # Filters the data based on the start and end dates
    if not start_date and not end_date:
        warnings.warn("No start or end date provided. Using the entire dataset.")
    
    if start_date:
        start_date = datetime.strptime(start_date, '%Y-%m-%d').timestamp()
        inf_data = inf_data[inf_data['timestamp'] >= start_date]

    if end_date:
        end_date = datetime.strptime(end_date, '%Y-%m-%d').timestamp()
        inf_data = inf_data[inf_data['timestamp'] <= end_date]
        
    if aggregate > 1:
        inf_data = aggregate_data(inf_data, aggregate)

    inf_data.to_csv(org_inf_path, index=False)
    inf_data.to_csv(Path(inf_save_path) / "inference.csv", index=False)

    experiment = mlflow_client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_name} not found.")
    
    runs = mlflow_client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{model_id}'"
    )
    
    run = runs[0]

    if run.data.params['target'] == 'returns':
        convert_to_returns(Path(inf_save_path) / "inference.csv")
    

    llm_model = run.data.params['llm_model']
    
    print("\n==============================================")
    print(f"\nRunning inference for {model_id} with {llm_model}")
    print("\n==============================================\n")

    inf_path = Path(inf_save_path) / "inference.csv"
    try:
        cmd = f"python run_inference.py --model_id {model_id} --llm_model {llm_model} --data_path {inf_path} --save_path {inf_save_path} --experiment_name {experiment_name}"
        subprocess.run(cmd, shell=True)
    except Exception as e:
        print(f"Error running inference: {e}")
        raise ValueError(f"Error running inference: \n{e}")

    inf_data = pd.read_csv(inf_path)
    ohlcv_path = inf_path
    if run.data.params['target'] == 'returns':
        inf_data_ret = Path("temp") / "inference_ret.csv"
        inf_data.to_csv(inf_data_ret, index=False)
        ohlcv_path = convert_back_to_candlesticks(original_data_path = org_inf_path, # Original Candlestick Data Path
                                        inferenced_data_path = inf_data_ret, 
                                        num_predictions = int(run.data.params['pred_len']),
                                        custom_save_path = Path("temp") / "inference_ohlcv.csv")

        print(f"OHLCV data saved to: {ohlcv_path}")

        mlflow.log_artifact(ohlcv_path, run_id = run.info.run_id)
    
    return ohlcv_path
    

def perform_backtest(inf_output_path, optimize=False, save_path = "temp"):
    """
    Perform backtest on the inference data.
    This function saves the summary table to the temp folder.

    args:
        inf_output_path: path to the inferenced data in candlestick format
    """
    print("\n==============================================")
    print(f"\nPerforming backtest for {inf_output_path}")
    print("\n==============================================\n")


    if optimize:
        cmd = f"python backtesting/backtest.py --data {inf_output_path} --walk_forward 12 --optimize BollingerAI --pipeline"
    else:
        cmd = f"python backtesting/backtest.py --data {inf_output_path} --pipeline"

    subprocess.run(cmd, shell=True)


def get_mda_vals(inf_path):
    """
    Perform analysis on the inference data.
    It can only be used on the OHLCV data.

    args:
        client: mlflow client
        new_data_path: path to the new data
    """
    target = 'close'

    data = pd.read_csv(inf_path)
    pred_len = data.columns.str.contains('predicted').sum()

    mda_vals = {}   
    
    for pred in range(1, pred_len+1):
        pred_col = f'{target}_predicted_{pred}'
        
        if pred_col not in data.columns:
            print(f"Column {pred_col} not found in data.")
            continue
        
        # Find rows where this prediction column has values (not NaN)
        valid_pred_mask = data[pred_col].notna()
        valid_pred_indices = data.index[valid_pred_mask].values
        
        if len(valid_pred_indices) == 0:
            print(f"No valid predictions found in {pred_col}")
            continue
        
        # For each valid prediction, we need to check if we have the future actual value
        # The prediction at index i predicts the value at index i+pred
        valid_comparisons = []
        
        for idx in valid_pred_indices:
            future_idx = idx + pred
            
            # Check if future actual value exists in the dataframe
            if future_idx < len(data) and pd.notna(data[target].iloc[future_idx]):
                current_actual = data[target].iloc[idx]
                predicted_value = data[pred_col].iloc[idx]
                future_actual = data[target].iloc[future_idx]
                
                # Calculate directions
                predicted_direction = predicted_value - current_actual
                actual_direction = future_actual - current_actual
                
                # Check if signs match
                correct = (predicted_direction * actual_direction) > 0
                valid_comparisons.append(correct)
        
        if len(valid_comparisons) == 0:
            print(f"No valid comparisons for prediction horizon {pred}")
            continue
        
        # Calculate directional accuracy
        mda = np.mean(valid_comparisons)
        mda_vals[f'inf_pred_{pred}_mda'] = mda

    if len(mda_vals) == 0:
        print(inf_path)
        raise ValueError("No valid MDA values found.")
    
    return mda_vals

def get_mse_vals(inf_path, pred_len, target = 'close'):
    """
    Get the MSE values for the inference data.

    args:
        inf_path: path to the inference data
        pred_len: number of predictions to get MSE values for
        target: target column
    """
    data = pd.read_csv(inf_path)
    pred_len = data.columns.str.contains('predicted').sum()
    mse_vals = {}
    print(data.columns)
    try:
        errors = {f'pred_{pred}': [] for pred in range(1, pred_len+1)}
        for i in range(len(data) - pred_len):
            row = data.iloc[i]
            if pd.isna(row[f'{target}_predicted_1']):
                continue
            for pred in range(1, pred_len+1):
                next_row = data.iloc[i+pred]
                if pd.notna(row[f'{target}_predicted_{pred}']):
                    error = row[f'{target}_predicted_{pred}'] - next_row[target]
                    sq_error = error ** 2
                    errors[f'pred_{pred}'].append(sq_error)

        for pred in range(1, pred_len+1):
            mse_vals[f'inf_pred_{pred}_mse'] = np.mean(errors[f'pred_{pred}'])
    except Exception as e:
        print(f"Error getting MSE values: {e}")
        raise ValueError(f"Error getting MSE values: {e}")

    return mse_vals

def get_mae_vals(inf_path, pred_len, target = 'close'):
    """
    Get the MAE values for the inference data.
    """
    data = pd.read_csv(inf_path)
    pred_len = data.columns.str.contains('predicted').sum()
    mae_vals = {}
    print(data.columns)
    try:
        errors = {f'pred_{pred}': [] for pred in range(1, pred_len+1)}
        for i in range(len(data) - pred_len):
            row = data.iloc[i]
            if pd.isna(row[f'{target}_predicted_1']):
                continue
            for pred in range(1, pred_len+1):
                next_row = data.iloc[i+pred]
                if pd.notna(row[f'{target}_predicted_{pred}']):
                    error = row[f'{target}_predicted_{pred}'] - next_row[target]
                    abs_error = abs(error)
                    errors[f'pred_{pred}'].append(abs_error)

        for pred in range(1, pred_len+1):
            mae_vals[f'inf_pred_{pred}_mae'] = np.mean(errors[f'pred_{pred}'])
    except Exception as e:
        print(f"Error getting MAE values: {e}")
        raise ValueError(f"Error getting MAE values: {e}")

    return mae_vals

def create_metrics_json(mlflow_run_id, llm_model, experiment_name, summary_table, mda_vals, trial_dict):
    """
    Create the metrics dataframe.

    args:
        summary_table: summary table
        mda_vals: mda values
    """
    metrics_dict = {}

    # Creates two metrics. The next candle prediction and the last candle prediction.

    metrics_dict["mlflow_run_id"] = mlflow_run_id
    metrics_dict["llm_model"] = llm_model
    metrics_dict["experiment_name"] = experiment_name
    metrics_dict["summary_table"] = summary_table.to_dict()
    metrics_dict["trial_parameters"] = trial_dict
    metrics_dict["inf_analysis"] = mda_vals

    metrics_json = json.dumps(metrics_dict)

    return metrics_json

def metrics_to_db(metrics_db_path, model_id, metrics_json):
    """
    Save metrics to the database as a JSON string.
    This is done to avoid the need to create a new table for each model.
    The metrics are stored in a JSON string so that they can be easily queried and analyzed.

    Args:
        metrics_db_path: path to the metrics database
        model_id: unique model identifier (primary key)
        metrics_json: dict of metrics (will be stored as JSON)
    """

    # Connects to the database
    print(f"Connecting to the metrics database at {metrics_db_path}\n\n")
    db = sqlite3.connect(metrics_db_path)
    cursor = db.cursor()

    try:
        # Creates the table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                model_id TEXT PRIMARY KEY,
                metrics JSON
            )
        """)
    except Exception as e:
        print(f"Failed to create metrics table in {metrics_db_path}: \n{e}")
        raise ValueError(f"Failed to create metrics table in {metrics_db_path}: \n{e}")

    try:
        # Creates the table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                model_id TEXT PRIMARY KEY,
                metrics JSON
            )
        """)
    except Exception as e:
        print(f"Failed to insert metrics into {metrics_db_path}: \n{e}")
        raise ValueError(f"Failed to insert metrics into {metrics_db_path}")

    # Inserts the metrics into the database
    cursor.execute("""
        INSERT OR REPLACE INTO metrics (model_id, metrics)
        VALUES (?, ?)
    """, (model_id, json.dumps(metrics_json)))

    print(f"Metrics inserted into {metrics_db_path} for model {model_id}\n")
    db.commit()
    db.close()

def aggregate_data(data, aggregate):
    """
    Aggregate OHLCV data from the original granularity to the specified granularity.
    Saves to save_path

    args:
        data_path: path to the data
        aggregate: aggregate period (e.g., '5' for 5 minutes from 1 minute, '60' for 1 hour from 1 minute, '1440' for 1 day from 1 minute)
    returns:
        Path to the aggregated data file (save_path)
    """
    if aggregate <= 1:
        warnings.warn("Aggregate period is 1 or less. Returning original data.")
        return data

    if 'timestamp' not in data:
        raise ValueError("Missing 'timestamp' column")

    unix = data['timestamp'].dtype in ('int64', 'float64')
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s', utc=True) if unix else pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)

    # Convert numeric aggregate to pandas frequency string (e.g., 5 -> '5T' for 5 minutes)
    freq = f'{aggregate}T' if isinstance(aggregate, (int, float)) else aggregate

    agg = {k: v for k, v in {
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }.items() if k in data}
    agg.update({c: 'mean' for c in data if c not in agg})

    out = data.resample(freq).agg(agg).dropna().reset_index()
    if unix:
        out['timestamp'] = out['timestamp'].astype('int64') // 10**9

    return out

def convert_to_returns(data_path, keep_high_low=False, keep_volume=True, log_returns=False):
    """
    Convert data to returns.

    args:
        data_path: path to the data
        log_returns: bool, if True, the data is converted to log returns
        keep_high_low: bool, if True, the high and low prices are kept
        keep_volume: bool, if True, the volume column is kept
    returns:
        Path to the returns data file (data_path)
    """
    # Checks if the data path is a file or a directory and saves the output path accordingly

    try:
        data = pd.read_csv(data_path)
    except:
        raise ValueError(f"Data path {data_path} is not a valid file.")
    try:
        data = pd.DataFrame({"close": data["close"], "volume": data["volume"], "timestamp": data["timestamp"]})
    except:
        print("-------------ERROR-------------------")
        print(data.head())
        print("-------------ERROR-------------------")
        raise ValueError(f"Data path {data_path} is not a valid file.")
    if log_returns:
        data["returns"] = np.log(data["close"] / data["close"].shift(1))
    else:
        data["returns"] = data["close"] / data["close"].shift(1) - 1
    
    data = data.dropna().reset_index(drop=True)

    final_data = pd.DataFrame()
    final_data['returns'] = data['returns']

    if keep_high_low:
        final_data["high"] = data["high"]
        final_data["low"] = data["low"]

    if keep_volume:
        final_data["volume"] = data["volume"]

    final_data["timestamp"] = data["timestamp"]

    final_data.to_csv(data_path, index=False)

    return data_path

def convert_back_to_candlesticks(original_data_path, inferenced_data_path, num_predictions: int, custom_save_path = None):
    """
    Convert inferenced returns data back to candlesticks. This is used to backtest the model.
    Writes the inferenced data to the inferenced data path.

    args:
        original_data_path: path to the original candlestick data 
        inferenced_data_path: path to save the inferenced data
        num_predictions: number of predictions to convert back to candlesticks
    """
    if not os.path.exists(Path(inferenced_data_path)):
        raise ValueError(f"Inference data path {inferenced_data_path} does not exist.")

    if not os.path.exists(original_data_path):
        raise ValueError(f"Original data path {original_data_path} does not exist.")

    result = pd.read_csv(original_data_path)
    predicted_returns = pd.read_csv(inferenced_data_path)

    # Get the last known close price before predictions start
    try:
        last_close = result.loc[result.index[predicted_returns['returns_predicted_1'].first_valid_index()-1], 'close']
    except Exception as e:
        print(f"Error getting last close price: {e}\n")
        raise ValueError(f"Error getting last close price: {e}")

    for i in range(1, num_predictions+1):  
        col = f'returns_predicted_{i}'
        if col in predicted_returns.columns:
            # Calculate cumulative returns 
            pred_close = last_close * (1 + predicted_returns[col])
            # Rename column
            result[f'close_predicted_{i}'] = pred_close

    # Convert unix timestamp to UTC datetime
    result["timestamp"] = pd.to_datetime(result["timestamp"], unit='s', utc=True)

    if custom_save_path is None:
        result.to_csv(inferenced_data_path, index=False)
    else:
        result.to_csv(custom_save_path, index=False)
    print(f"Predicted candlesticks saved to {custom_save_path if custom_save_path is not None else inferenced_data_path}")

    return custom_save_path if custom_save_path is not None else inferenced_data_path
    

