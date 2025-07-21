import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features

def prepare_data_for_prediction(df, args, scaler=None):
    """Prepare data for prediction: scale, ensure target is last column, create time features."""
    # Ensure timestamp column exists and is properly formatted
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    elif 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'])
    else:
        raise ValueError("No timestamp or date column found in data")

    # Ensure target is last column (except timestamp)
    feature_cols = [col for col in df.columns if col not in ['timestamp', args.target]]
    ordered_cols = ['timestamp'] + feature_cols + [args.target]
    df = df[ordered_cols]

    # Select numeric columns (excluding timestamp)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'timestamp' in numeric_cols:
        numeric_cols.remove('timestamp')

    # Create scaler if not provided
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(df[numeric_cols])
    # Scale the data
    scaled_data = scaler.transform(df[numeric_cols])



    return scaled_data, scaler, numeric_cols 