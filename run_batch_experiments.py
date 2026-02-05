import subprocess
import sys
import mlflow
import uuid
import time
import os
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

# --- Centralized Configuration ---
MLFLOW_SERVER_IP = "192.168.1.103"
# MLflow
os.environ["MLFLOW_TRACKING_URI"] = f"http://{MLFLOW_SERVER_IP}:5000" # Assumes the server is running. Can set to "" to save locally

# MinIO
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{MLFLOW_SERVER_IP}:9000"

# --- Global Parameters ---
GLOBAL_PARAMS = {
    'llm_model': 'LLAMA3.1',
    'llm_layers': 6,
    'd_model': 32,
    'n_heads': 8,
    'd_ff': 64,
    'dropout': 0.15,
    'patch_len': 4,  # Default for seq_len=7, will override for seq_len=14
    'stride': 2,  # Default for seq_len=7, will override for seq_len=14
    'num_tokens': 1000,
    'data': 'CRYPTEX',
    'root_path': './dataset',
    'features': 'MS',
    'target': 'close',
    'enc_in': 5,  # Default for OHLCV, will override per experiment
    'train_epochs': 10,
    'batch_size': 32,
    'eval_batch_size': 8,
    'learning_rate': 1e-05,
    'loss': 'MSE',
    'metric': 'MDA',
    'lradj': 'WARM_COS',
    'pct_start': 0.2,
    'warmup_ratio': 0.1,
    'min_lr': 1e-6,
    'plateau_patience': 2,
    'plateau_delta': 0.0,
    'peak_decay': 0.8,
    'restart_T0_epochs': 1.0,
    'patience': 10,
    'percent': 100,
    'use_amp': False,
    'seq_len': 7,  # Default, will override for 2-week experiments
    'pred_len': 1,
    'num_workers': 10,
    'seed': 2021,
    'enable_mlflow': True,
}

# --- Experiments List ---
# Legacy setups (for reference only):
"""
EXPERIMENTS = [
    {
        'name': 'daily_1week_to_1day_25pct',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 7,
        'pred_len': 1,
        'batch_size': 16,
        'percent': 25,
        'loss': 'MSE',
        'metric': 'MDA',
        'lradj': 'constant',
        'patch_len': 4,
        'stride': 2,
    },
    {
        'name': 'daily_2weeks_to_1day_25pct',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 14,
        'pred_len': 1,
        'batch_size': 16,
        'percent': 25,
        'loss': 'MSE',
        'metric': 'MDA',
        'lradj': 'constant',
        'patch_len': 7,
        'stride': 3,
    },
    {
        'name': 'daily_1week_to_1day_50pct',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 7,
        'pred_len': 1,
        'batch_size': 16,
        'percent': 50,
        'loss': 'MSE',
        'metric': 'MDA',
        'lradj': 'constant',
        'patch_len': 4,
        'stride': 2,
    },
    {
        'name': 'daily_2weeks_to_1day_50pct',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 14,
        'pred_len': 1,
        'batch_size': 16,
        'percent': 50,
        'loss': 'MSE',
        'metric': 'MDA',
        'lradj': 'constant',
        'patch_len': 7,
        'stride': 3,
    },
    {
        'name': 'daily_1week_to_1day_75pct',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 7,
        'pred_len': 1,
        'batch_size': 16,
        'percent': 75,
        'loss': 'MSE',
        'metric': 'MDA',
        'lradj': 'constant',
        'patch_len': 4,
        'stride': 2,
    },
    {
        'name': 'daily_2weeks_to_1day_75pct',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 14,
        'pred_len': 1,
        'batch_size': 16,
        'percent': 75,
        'loss': 'MSE',
        'metric': 'MDA',
        'lradj': 'constant',
        'patch_len': 7,
        'stride': 3,
    },
    {
        'name': 'daily_1week_to_1day_100pct',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 7,
        'pred_len': 1,
        'batch_size': 16,
        'percent': 100,
        'loss': 'MSE',
        'metric': 'MDA',
        'lradj': 'constant',
        'patch_len': 4,
        'stride': 2,
    },
    {
        'name': 'daily_2weeks_to_1day_100pct',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 14,
        'pred_len': 1,
        'batch_size': 16,
        'percent': 100,
        'loss': 'MSE',
        'metric': 'MDA',
        'lradj': 'constant',
        'patch_len': 7,
        'stride': 3,
    },
]
"""

EXPERIMENTS = [
    # OLD EXPERIMENTS - COMMENTED OUT
    # # OHLCV Baseline - 1 week to 1 day (seq_len=7)
    # {
    #     'name': 'ohlcv_1week_to_1day_100pct',
    #     'data_path': 'cryptex/daily/candlesticks-D.csv',
    #     'seq_len': 7,
    #     'pred_len': 1,
    #     'enc_in': 5,
    #     'patch_len': 4,
    #     'stride': 2,
    #     'percent': 100,
    # },
    # {
    #     'name': 'ohlcv_1week_to_1day_75pct',
    #     'data_path': 'cryptex/daily/candlesticks-D.csv',
    #     'seq_len': 7,
    #     'pred_len': 1,
    #     'enc_in': 5,
    #     'patch_len': 4,
    #     'stride': 2,
    #     'percent': 75,
    # },
    # {
    #     'name': 'ohlcv_1week_to_1day_50pct',
    #     'data_path': 'cryptex/daily/candlesticks-D.csv',
    #     'seq_len': 7,
    #     'pred_len': 1,
    #     'enc_in': 5,
    #     'patch_len': 4,
    #     'stride': 2,
    #     'percent': 50,
    # },
    # {
    #     'name': 'ohlcv_1week_to_1day_25pct',
    #     'data_path': 'cryptex/daily/candlesticks-D.csv',
    #     'seq_len': 7,
    #     'pred_len': 1,
    #     'enc_in': 5,
    #     'patch_len': 4,
    #     'stride': 2,
    #     'percent': 25,
    # },
    # 
    # # Feature-Engineered Sets - 1 week to 1 day (seq_len=7, all 100%)
    # {
    #     'name': 'momentum_1week_to_1day',
    #     'data_path': 'cryptex/daily/candlesticks-D_features_momentum.csv',
    #     'seq_len': 7,
    #     'pred_len': 1,
    #     'enc_in': 6,
    #     'patch_len': 4,
    #     'stride': 2,
    #     'percent': 100,
    # },
    # {
    #     'name': 'volatility_1week_to_1day',
    #     'data_path': 'cryptex/daily/candlesticks-D_features_volatility.csv',
    #     'seq_len': 7,
    #     'pred_len': 1,
    #     'enc_in': 5,
    #     'patch_len': 4,
    #     'stride': 2,
    #     'percent': 100,
    # },
    # {
    #     'name': 'onchain_price_1week_to_1day',
    #     'data_path': 'cryptex/daily/candlesticks-D_features_onchain_price.csv',
    #     'seq_len': 7,
    #     'pred_len': 1,
    #     'enc_in': 8,
    #     'patch_len': 4,
    #     'stride': 2,
    #     'percent': 100,
    # },
    # {
    #     'name': 'volume_price_1week_to_1day',
    #     'data_path': 'cryptex/daily/candlesticks-D_features_volume_price.csv',
    #     'seq_len': 7,
    #     'pred_len': 1,
    #     'enc_in': 11,
    #     'patch_len': 4,
    #     'stride': 2,
    #     'percent': 100,
    # },
    # {
    #     'name': 'technical_1week_to_1day',
    #     'data_path': 'cryptex/daily/candlesticks-D_features_technical.csv',
    #     'seq_len': 7,
    #     'pred_len': 1,
    #     'enc_in': 6,
    #     'patch_len': 4,
    #     'stride': 2,
    #     'percent': 100,
    # },
    # {
    #     'name': 'hybrid_1week_to_1day',
    #     'data_path': 'cryptex/daily/candlesticks-D_features_hybrid.csv',
    #     'seq_len': 7,
    #     'pred_len': 1,
    #     'enc_in': 10,
    #     'patch_len': 4,
    #     'stride': 2,
    #     'percent': 100,
    # },
    # {
    #     'name': 'returns_1week_to_1day',
    #     'data_path': 'cryptex/daily/candlesticks-D_features_returns.csv',
    #     'seq_len': 7,
    #     'pred_len': 1,
    #     'enc_in': 9,
    #     'patch_len': 4,
    #     'stride': 2,
    #     'percent': 100,
    # },
    # {
    #     'name': 'temporal_1week_to_1day',
    #     'data_path': 'cryptex/daily/candlesticks-D_features_temporal.csv',
    #     'seq_len': 7,
    #     'pred_len': 1,
    #     'enc_in': 8,
    #     'patch_len': 4,
    #     'stride': 2,
    #     'percent': 100,
    # },
    # 
    # # OHLCV 2-week Baseline - 2 weeks to 1 day (seq_len=14)
    # {
    #     'name': 'ohlcv_2weeks_to_1day_100pct',
    #     'data_path': 'cryptex/daily/candlesticks-D.csv',
    #     'seq_len': 14,
    #     'pred_len': 1,
    #     'enc_in': 5,
    #     'patch_len': 7,
    #     'stride': 3,
    #     'percent': 100,
    # },
    # {
    #     'name': 'ohlcv_2weeks_to_1day_75pct',
    #     'data_path': 'cryptex/daily/candlesticks-D.csv',
    #     'seq_len': 14,
    #     'pred_len': 1,
    #     'enc_in': 5,
    #     'patch_len': 7,
    #     'stride': 3,
    #     'percent': 75,
    # },
    # {
    #     'name': 'ohlcv_2weeks_to_1day_50pct',
    #     'data_path': 'cryptex/daily/candlesticks-D.csv',
    #     'seq_len': 14,
    #     'pred_len': 1,
    #     'enc_in': 5,
    #     'patch_len': 7,
    #     'stride': 3,
    #     'percent': 50,
    # },
    # {
    #     'name': 'ohlcv_2weeks_to_1day_25pct',
    #     'data_path': 'cryptex/daily/candlesticks-D.csv',
    #     'seq_len': 14,
    #     'pred_len': 1,
    #     'enc_in': 5,
    #     'patch_len': 7,
    #     'stride': 3,
    #     'percent': 25,
    # },
    
    # OHLCV LR scheduler sweep - 30d-to-1d and 120d-to-30d (all schedulers except type3)
    # OHLCV 30d-to-1d x 8 schedulers
    {
        'name': 'ohlcv_trading_30days_to_1day_constant',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 30,
        'pred_len': 1,
        'enc_in': 5,
        'patch_len': 10,
        'stride': 5,
        'percent': 100,
        'loss': 'TRADING',
        'lradj': 'constant',
    },
    {
        'name': 'ohlcv_trading_30days_to_1day_type1',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 30,
        'pred_len': 1,
        'enc_in': 5,
        'patch_len': 10,
        'stride': 5,
        'percent': 100,
        'loss': 'TRADING',
        'lradj': 'type1',
    },
    {
        'name': 'ohlcv_trading_30days_to_1day_type2',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 30,
        'pred_len': 1,
        'enc_in': 5,
        'patch_len': 10,
        'stride': 5,
        'percent': 100,
        'loss': 'TRADING',
        'lradj': 'type2',
    },
    {
        'name': 'ohlcv_trading_30days_to_1day_PEMS',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 30,
        'pred_len': 1,
        'enc_in': 5,
        'patch_len': 10,
        'stride': 5,
        'percent': 100,
        'loss': 'TRADING',
        'lradj': 'PEMS',
    },
    {
        'name': 'ohlcv_trading_30days_to_1day_COS',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 30,
        'pred_len': 1,
        'enc_in': 5,
        'patch_len': 10,
        'stride': 5,
        'percent': 100,
        'loss': 'TRADING',
        'lradj': 'COS',
    },
    {
        'name': 'ohlcv_trading_30days_to_1day_TST',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 30,
        'pred_len': 1,
        'enc_in': 5,
        'patch_len': 10,
        'stride': 5,
        'percent': 100,
        'loss': 'TRADING',
        'lradj': 'TST',
    },
    {
        'name': 'ohlcv_trading_30days_to_1day_WARM_COS',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 30,
        'pred_len': 1,
        'enc_in': 5,
        'patch_len': 10,
        'stride': 5,
        'percent': 100,
        'loss': 'TRADING',
        'lradj': 'WARM_COS',
    },
    {
        'name': 'ohlcv_trading_30days_to_1day_PLATEAU_RESTART',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 30,
        'pred_len': 1,
        'enc_in': 5,
        'patch_len': 10,
        'stride': 5,
        'percent': 100,
        'loss': 'TRADING',
        'lradj': 'PLATEAU_RESTART',
    },
    # OHLCV 120d-to-30d x 8 schedulers
    {
        'name': 'ohlcv_trading_120days_to_30days_constant',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 120,
        'pred_len': 30,
        'enc_in': 5,
        'patch_len': 30,
        'stride': 15,
        'percent': 100,
        'loss': 'TRADING',
        'lradj': 'constant',
    },
    {
        'name': 'ohlcv_trading_120days_to_30days_type1',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 120,
        'pred_len': 30,
        'enc_in': 5,
        'patch_len': 30,
        'stride': 15,
        'percent': 100,
        'loss': 'TRADING',
        'lradj': 'type1',
    },
    {
        'name': 'ohlcv_trading_120days_to_30days_type2',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 120,
        'pred_len': 30,
        'enc_in': 5,
        'patch_len': 30,
        'stride': 15,
        'percent': 100,
        'loss': 'TRADING',
        'lradj': 'type2',
    },
    {
        'name': 'ohlcv_trading_120days_to_30days_PEMS',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 120,
        'pred_len': 30,
        'enc_in': 5,
        'patch_len': 30,
        'stride': 15,
        'percent': 100,
        'loss': 'TRADING',
        'lradj': 'PEMS',
    },
    {
        'name': 'ohlcv_trading_120days_to_30days_COS',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 120,
        'pred_len': 30,
        'enc_in': 5,
        'patch_len': 30,
        'stride': 15,
        'percent': 100,
        'loss': 'TRADING',
        'lradj': 'COS',
    },
    {
        'name': 'ohlcv_trading_120days_to_30days_TST',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 120,
        'pred_len': 30,
        'enc_in': 5,
        'patch_len': 30,
        'stride': 15,
        'percent': 100,
        'loss': 'TRADING',
        'lradj': 'TST',
    },
    {
        'name': 'ohlcv_trading_120days_to_30days_WARM_COS',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 120,
        'pred_len': 30,
        'enc_in': 5,
        'patch_len': 30,
        'stride': 15,
        'percent': 100,
        'loss': 'TRADING',
        'lradj': 'WARM_COS',
    },
    {
        'name': 'ohlcv_trading_120days_to_30days_PLATEAU_RESTART',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 120,
        'pred_len': 30,
        'enc_in': 5,
        'patch_len': 30,
        'stride': 15,
        'percent': 100,
        'loss': 'TRADING',
        'lradj': 'PLATEAU_RESTART',
    },
]

def _extract_feature_set_name(data_path: str):
    if "_features_" not in data_path:
        return None
    parts = data_path.split("_features_", 1)
    if len(parts) != 2:
        return None
    return parts[1].replace(".csv", "")


def _get_posttrain_inference_input_path(training_data_path: str, holdout_base_path: str) -> str:
    """
    Map a training data_path to the correct post-train inference input CSV.
    Returned path is relative to the dataset root_path.
    """
    feature_set = _extract_feature_set_name(training_data_path)
    if feature_set is None:
        return holdout_base_path
    holdout_dir = os.path.dirname(holdout_base_path)
    return os.path.join(holdout_dir, f"inference_test_btc_D_posttrain_features_{feature_set}.csv")


def _write_backtest_ready_inference_csv(
    inference_csv_path: str,
    holdout_ohlcv_csv_path: str,
    output_csv_path: str,
) -> None:
    """
    Ensure output CSV has OHLCV + close_predicted_* columns for backtesting.
    """
    df_inf = pd.read_csv(inference_csv_path)
    if "timestamp" not in df_inf.columns:
        raise ValueError(f"inference.csv missing timestamp column: {inference_csv_path}")

    required_ohlcv = ["open", "high", "low", "close", "volume"]
    has_ohlcv = all(c in df_inf.columns for c in required_ohlcv)
    pred_cols = [c for c in df_inf.columns if c.startswith("close_predicted_")]
    if not pred_cols:
        raise ValueError(f"inference.csv has no close_predicted_* columns: {inference_csv_path}")

    if has_ohlcv:
        out_cols = ["timestamp"] + required_ohlcv + pred_cols
        df_inf[out_cols].to_csv(output_csv_path, index=False)
        return

    df_holdout = pd.read_csv(holdout_ohlcv_csv_path)
    if "timestamp" not in df_holdout.columns:
        raise ValueError(f"Holdout OHLCV missing timestamp column: {holdout_ohlcv_csv_path}")
    for col in required_ohlcv:
        if col not in df_holdout.columns:
            raise ValueError(f"Holdout OHLCV missing required column '{col}': {holdout_ohlcv_csv_path}")

    df_inf = df_inf[["timestamp"] + pred_cols].copy()
    df_inf["timestamp"] = pd.to_datetime(df_inf["timestamp"], errors="raise")

    df_holdout = df_holdout[["timestamp"] + required_ohlcv].copy()
    df_holdout["timestamp"] = pd.to_datetime(df_holdout["timestamp"], unit="s", errors="raise")

    df_out = df_holdout.merge(df_inf, on="timestamp", how="left")
    df_out.to_csv(output_csv_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='1', help='If not GPU 1, changes OPTUNA_STORAGE_PATH.')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Directory for checkpoints.')
    parser.add_argument('--start-from', type=int, default=0, 
                       help='Start from experiment index (0-based). Default: 0 (run all experiments)')
    parser.add_argument('--skip_inference_after_train', action='store_true',
                       help='If set, do not run inference after each successful training run.')
    parser.add_argument('--inference_output_dir', type=str, default='backtesting/inferences',
                       help='Directory where backtest-ready inference CSVs will be written.')
    parser.add_argument('--inference_holdout_path', type=str, default='cryptex/daily/inference_test_btc_D_posttrain.csv',
                       help='Post-train OHLCV holdout CSV relative to root_path.')
    return parser.parse_args()

def _validate_data_split(params, args):
    """
    Validate data split to ensure no overlap between train/val/test sets.
    Also checks data quality and file existence.
    
    Returns:
        bool: True if all checks pass, False otherwise
    """
    print("\n--- Data Split Validation ---")
    
    # Check data file exists
    data_file_path = os.path.join(params['root_path'], params['data_path'])
    if not os.path.exists(data_file_path):
        print(f"ERROR: Data file does not exist: {data_file_path}")
        return False
    print(f"Data file exists: {data_file_path}")
    
    # Load data to check structure
    try:
        df_raw = pd.read_csv(data_file_path)
        print(f"Data loaded: {len(df_raw)} rows, {len(df_raw.columns)} columns")
    except Exception as e:
        print(f"ERROR: Failed to load data file: {e}")
        return False
    
    # Check for required columns
    if 'timestamp' not in df_raw.columns:
        print("WARNING: 'timestamp' column not found (optional)")
    
    target = params.get('target', 'close')
    if target not in df_raw.columns:
        print(f"ERROR: Target column '{target}' not found in data")
        return False
    
    # Calculate split boundaries using same logic as Dataset_Crypto
    seq_len = params['seq_len']
    pred_len = params['pred_len']
    percent = params.get('percent', 100)
    
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    total_len = len(df_raw)
    num_train = int(total_len * train_ratio)
    num_vali = int(total_len * val_ratio)
    num_test = total_len - num_train - num_vali
    
    # Calculate borders for each split (same as Dataset_Crypto)
    train_border1 = 0
    train_border2 = num_train
    if percent < 100:
        train_border2 = (train_border2 - seq_len) * percent // 100 + seq_len
    
    val_border1 = num_train - seq_len
    val_border2 = num_train + num_vali
    
    test_border1 = total_len - num_test - seq_len
    test_border2 = total_len
    
    # Target prediction ranges (no overlap should occur here)
    train_target_start = seq_len
    train_target_end = train_border2
    
    val_target_start = num_train
    val_target_end = num_train + num_vali
    
    test_target_start = total_len - num_test
    test_target_end = total_len
    
    print(f"\nSplit boundaries:")
    print(f"  Train data: [{train_border1}, {train_border2}] ({train_border2 - train_border1} samples)")
    print(f"  Train targets: [{train_target_start}, {train_target_end}] ({train_target_end - train_target_start} samples)")
    print(f"  Val data: [{val_border1}, {val_border2}] ({val_border2 - val_border1} samples)")
    print(f"  Val targets: [{val_target_start}, {val_target_end}] ({val_target_end - val_target_start} samples)")
    print(f"  Test data: [{test_border1}, {test_border2}] ({test_border2 - test_border1} samples)")
    print(f"  Test targets: [{test_target_start}, {test_target_end}] ({test_target_end - test_target_start} samples)")
    
    # Verify no overlap in target ranges
    overlap_issues = []
    if train_target_end > val_target_start:
        overlap_issues.append(f"Train targets overlap with val targets: [{train_target_start}, {train_target_end}] vs [{val_target_start}, {val_target_end}]")
    if val_target_end > test_target_start:
        overlap_issues.append(f"Val targets overlap with test targets: [{val_target_start}, {val_target_end}] vs [{test_target_start}, {test_target_end}]")
    if train_target_end > test_target_start:
        overlap_issues.append(f"Train targets overlap with test targets: [{train_target_start}, {train_target_end}] vs [{test_target_start}, {test_target_end}]")
    
    if overlap_issues:
        print("\nERROR: Target prediction ranges overlap!")
        for issue in overlap_issues:
            print(f"  - {issue}")
        return False
    
    print("\nTarget ranges verified: No overlap detected")
    
    # Check feature count matches enc_in
    features = params.get('features', 'MS')
    if features == 'M' or features == 'MS':
        expected_features = len(df_raw.columns) - 1  # Exclude timestamp
    elif features == 'S':
        expected_features = 1
    else:
        expected_features = len(df_raw.columns) - 1
    
    enc_in = params.get('enc_in')
    if enc_in is not None and expected_features != enc_in:
        print(f"WARNING: Feature count mismatch. Expected {enc_in} (enc_in), but data has {expected_features} features")
        print(f"  This may be intentional if enc_in is manually set")
    else:
        print(f"Feature count: {expected_features} (matches enc_in={enc_in})")
    
    # Check for NaN/Inf values
    feature_cols = [col for col in df_raw.columns if col != 'timestamp']
    df_features = df_raw[feature_cols]
    nan_count = df_features.isna().sum().sum()
    inf_count = np.isinf(df_features.select_dtypes(include=[np.number])).sum().sum()
    
    if nan_count > 0:
        print(f"WARNING: Found {nan_count} NaN values in data")
    else:
        print("Data quality: No NaN values found")
    
    if inf_count > 0:
        print(f"WARNING: Found {inf_count} Inf values in data")
    else:
        print("Data quality: No Inf values found")
    
    # Check inference holdout file if inference will run
    if not args.skip_inference_after_train:
        inference_input_path = _get_posttrain_inference_input_path(
            training_data_path=params["data_path"],
            holdout_base_path=args.inference_holdout_path,
        )
        inference_input_full = os.path.join(params["root_path"], inference_input_path)
        holdout_full = os.path.join(params["root_path"], args.inference_holdout_path)
        
        if not os.path.exists(inference_input_full):
            print(f"WARNING: Inference input file does not exist: {inference_input_full}")
            print(f"  Inference will fail if run. Generate with generate_inference_feature_datasets.py")
        else:
            print(f"Inference input file exists: {inference_input_full}")
        
        if not os.path.exists(holdout_full):
            print(f"WARNING: Holdout OHLCV file does not exist: {holdout_full}")
            print(f"  Inference will fail if run. Generate with generate_inference_feature_datasets.py")
        else:
            print(f"Holdout OHLCV file exists: {holdout_full}")
    
    # Print date range if timestamps available
    if 'timestamp' in df_raw.columns:
        try:
            timestamps = pd.to_datetime(df_raw['timestamp'], unit='s', errors='coerce')
            if timestamps.notna().any():
                start_date = timestamps.min()
                end_date = timestamps.max()
                print(f"\nDate range: {start_date} to {end_date}")
                
                # Show split date ranges
                train_end_idx = min(train_border2, len(timestamps))
                val_end_idx = min(val_border2, len(timestamps))
                if train_end_idx > 0:
                    print(f"  Train ends: {timestamps.iloc[train_end_idx - 1]}")
                if val_end_idx > 0:
                    print(f"  Val ends: {timestamps.iloc[val_end_idx - 1]}")
                print(f"  Test ends: {timestamps.iloc[-1]}")
        except Exception as e:
            print(f"Could not parse timestamps: {e}")
    
    print("--- Validation Complete ---\n")
    return True


def _find_mlflow_run(client, experiment_name, model_id):
    """Finds an MLflow run based on its name within a given experiment."""
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Error: MLflow experiment '{experiment_name}' not found.")
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{model_id}'"
    )
    
    if runs:
        return runs[0]
    else:
        print(f"Error: MLflow run '{model_id}' not found in experiment '{experiment_name}'.")
        return None

def run_experiment(experiment_config, global_params, args):
    """Run a single experiment with given configuration."""
    
    # Merge global params with experiment-specific overrides
    params = global_params.copy()
    params.update(experiment_config)
    
    # Generate unique model_id
    experiment_name = params['name']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_id = f"{experiment_name}_{timestamp}"
    
    # Run data split validation before proceeding
    if not _validate_data_split(params, args):
        print(f"ERROR: Data validation failed for experiment {experiment_name}")
        print("Skipping this experiment due to validation errors.")
        return False, model_id, "Data validation failed"
    
    # Build the command
    cmd = [
        'accelerate', 'launch', '--multi_gpu', '--mixed_precision', 'bf16', '--num_processes', '4', '--main_process_port', '29500',
        'run_main.py',
        # All parameters
        '--model_id', model_id,
        '--data', params['data'],
        '--root_path', params['root_path'],
        '--data_path', params['data_path'],
        '--features', params['features'],
        '--target', params['target'],
        '--seq_len', str(params['seq_len']),
        '--pred_len', str(params['pred_len']),
        '--enc_in', str(params['enc_in']),
        '--d_model', str(params['d_model']),
        '--n_heads', str(params['n_heads']),
        '--d_ff', str(params['d_ff']),
        '--dropout', str(params['dropout']),
        '--patch_len', str(params['patch_len']),
        '--stride', str(params['stride']),
        '--llm_model', params['llm_model'],
        '--llm_layers', str(params['llm_layers']),
        '--num_tokens', str(params['num_tokens']),
        '--num_workers', str(params['num_workers']),
        '--train_epochs', str(params['train_epochs']),
        '--batch_size', str(params['batch_size']),
        '--eval_batch_size', str(params['eval_batch_size']),
        '--patience', str(params['patience']),
        '--learning_rate', str(params['learning_rate']),
        '--loss', params['loss'],
        '--metric', params['metric'],
        '--lradj', params['lradj'],
        '--pct_start', str(params['pct_start']),
        '--warmup_ratio', str(params.get('warmup_ratio', 0.1)),
        '--min_lr', str(params.get('min_lr', 1e-6)),
        '--plateau_patience', str(params.get('plateau_patience', 2)),
        '--plateau_delta', str(params.get('plateau_delta', 0.0)),
        '--peak_decay', str(params.get('peak_decay', 0.8)),
        '--restart_T0_epochs', str(params.get('restart_T0_epochs', 1.0)),
        '--percent', str(params['percent']),
        '--seed', str(params['seed']),
        '--checkpoints', args.checkpoints,
    ]
    
    if params.get('use_amp', False):
        cmd.append('--use_amp')
    
    if params.get('enable_mlflow', True):
        cmd.append('--enable_mlflow')
    
    print(f"\n--- Starting Experiment: {experiment_name} ---")
    print(f"Model ID: {model_id}")
    
    # Pre-execution validation
    data_file_path = os.path.join(params['root_path'], params['data_path'])
    print(f"Data path: {params['data_path']}")
    print(f"Full data path: {data_file_path}")
    print(f"Percent of data: {params['percent']}%")
    print(f"Seq len: {params['seq_len']}, Pred len: {params['pred_len']}")
    print(f"Training epochs: {params['train_epochs']}")
    print(f"Batch size: {params['batch_size']}")

    if not os.path.exists(data_file_path):
        print(f"WARNING: Data path does not exist: {data_file_path}")
        parent_path = os.path.dirname(data_file_path)
        if os.path.exists(parent_path):
            print(f"Available in {parent_path}:")
            for item in os.listdir(parent_path):
                print(f"  - {item}")
    
    # Show full command in readable format
    print(f"\n--- Full Command ---")
    for i in range(0, len(cmd), 4):
        print(' '.join(cmd[i:i+4]))
    print("---\n")
    
    # Run the experiment
    client = mlflow.tracking.MlflowClient()
    
    try:
        # Launch the subprocess (stream output to see real-time progress)
        result = subprocess.run(cmd, check=True, text=True)
        print(f"\n--- Experiment {experiment_name} Completed Successfully ---")
        
        # Give MLflow a moment to log everything
        time.sleep(2)
        
        # Find the MLflow run and get metrics
        run = _find_mlflow_run(client, params['llm_model'], model_id)
        if run:
            latest_metrics = run.data.metrics
            metric_key = f'vali_{params["metric"].lower()}_metric'
            print(f"Final validation metric: {latest_metrics.get(metric_key, 'N/A')}")

            if not args.skip_inference_after_train:
                os.makedirs(args.inference_output_dir, exist_ok=True)

                inference_input_path = _get_posttrain_inference_input_path(
                    training_data_path=params["data_path"],
                    holdout_base_path=args.inference_holdout_path,
                )
                inference_input_full = os.path.join(params["root_path"], inference_input_path)
                holdout_full = os.path.join(params["root_path"], args.inference_holdout_path)

                if not os.path.exists(inference_input_full):
                    raise FileNotFoundError(
                        f"Missing post-train inference input CSV: {inference_input_full}. "
                        "Generate it first with generate_inference_feature_datasets.py."
                    )
                if not os.path.exists(holdout_full):
                    raise FileNotFoundError(
                        f"Missing post-train OHLCV holdout CSV: {holdout_full}. "
                        "Generate it first with generate_inference_feature_datasets.py."
                    )

                print("\n--- Running inference (post-train holdout) ---")
                print(f"Inference input: {inference_input_path}")

                inf_cmd = [
                    sys.executable,
                    "run_inference.py",
                    "--model_id", model_id,
                    "--llm_model", params["llm_model"],
                    "--data_path", inference_input_path,
                ]
                subprocess.run(inf_cmd, check=True, text=True)

                run_id = run.info.run_id
                downloaded = client.download_artifacts(run_id, "inference.csv")
                output_csv = os.path.join(args.inference_output_dir, f"{experiment_name}.csv")
                _write_backtest_ready_inference_csv(
                    inference_csv_path=downloaded,
                    holdout_ohlcv_csv_path=holdout_full,
                    output_csv_path=output_csv,
                )
                print(f"Saved backtest-ready inference CSV: {output_csv}")

                # Save the backtest-ready inference output with the model run in MLflow
                client.log_artifact(run_id, output_csv, artifact_path="backtesting_inferences")
        
        return True, model_id, None
        
    except subprocess.CalledProcessError as e:
        print(f"\n--- Experiment {experiment_name} Failed ---")
        print(f"Exit code: {e.returncode}")
        # Note: stdout/stderr not available when capture_output=False
        print("See output above for error details")
        
        time.sleep(2)
        # Log error to MLflow
        run = _find_mlflow_run(client, params['llm_model'], model_id)
        if run:
            failed_run_id = run.info.run_id
            # Since we're not capturing output, log a simple error message
            error_message = f"Experiment {experiment_name} failed with exit code {e.returncode}. See console output for details."
            client.log_text(failed_run_id, error_message, f"failed_experiment_{experiment_name}_error.log")
            print(f"--> Error log saved as an artifact to failed MLflow run ID: {failed_run_id}")
            client.set_terminated(failed_run_id, "FAILED")
        
        return False, model_id, str(e)

def main():
    """Main function to run all experiments."""
    args = parse_args()
    
    # Validate start-from index
    if args.start_from < 0:
        print("Error: --start-from must be >= 0")
        return
    if args.start_from >= len(EXPERIMENTS):
        print(f"Error: --start-from ({args.start_from}) >= number of experiments ({len(EXPERIMENTS)})")
        return
    
    # Get experiments to run
    experiments_to_run = EXPERIMENTS[args.start_from:]
    
    print("=== Batch Experiments Runner ===")
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"Starting from experiment {args.start_from + 1} (skipping first {args.start_from} experiments)")
    print(f"Experiments to run: {len(experiments_to_run)}")
    print(f"Global LLM model: {GLOBAL_PARAMS['llm_model']}")
    print(f"MLflow server: {MLFLOW_SERVER_IP}:5000")
    print("=" * 50)
    
    successful_experiments = []
    failed_experiments = []
    
    for i, experiment_config in enumerate(experiments_to_run):
        actual_index = args.start_from + i
        print(f"\n[{actual_index + 1}/{len(EXPERIMENTS)}] Running experiment: {experiment_config['name']}")
        
        success, model_id, error = run_experiment(experiment_config, GLOBAL_PARAMS, args)
        
        if success:
            successful_experiments.append({
                'name': experiment_config['name'],
                'model_id': model_id
            })
        else:
            failed_experiments.append({
                'name': experiment_config['name'],
                'model_id': model_id,
                'error': error
            })
    
    # Print summary
    print("\n" + "=" * 50)
    print("=== BATCH EXPERIMENTS SUMMARY ===")
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"Experiments run: {len(experiments_to_run)}")
    print(f"Successful: {len(successful_experiments)}")
    print(f"Failed: {len(failed_experiments)}")
    
    if successful_experiments:
        print("\nSuccessful experiments:")
        for exp in successful_experiments:
            print(f"  OK {exp['name']} -> {exp['model_id']}")
    
    if failed_experiments:
        print("\nFailed experiments:")
        for exp in failed_experiments:
            print(f"  FAIL {exp['name']} -> {exp['model_id']}")
            print(f"    Error: {exp['error']}")
    
    print("=" * 50)

if __name__ == "__main__":
    main()



