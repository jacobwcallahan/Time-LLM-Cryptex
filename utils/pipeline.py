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
from hpo_core.DataManager import DataManager
    

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
    

