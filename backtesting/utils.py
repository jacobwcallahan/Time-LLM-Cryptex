import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(csv_path):
    """
    Load CSV data from AI model inference and prepare for backtesting
    """
    df = pd.read_csv(csv_path)
    
    # Convert timestamp to datetime if needed
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
    
    # Ensure we have OHLCV columns (backtesting.py requirement)
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col.lower() not in df.columns and col.upper() not in df.columns:
            raise ValueError(f"Required column '{col}' not found in data")
    
    # Standardize column names (backtesting.py expects title case)
    column_mapping = {
        'open': 'Open', 'high': 'High', 'low': 'Low', 
        'close': 'Close', 'volume': 'Volume'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)
    
    # Remove any rows with NaN values in OHLCV
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    return df


def analyze_predictions(data):
    """
    Analyze AI model prediction accuracy
    """
    pred_cols = [col for col in data.columns if col.startswith('close_predicted_')]
    
    analysis = {}
    
    for pred_col in pred_cols:
        horizon = pred_col.split('_')[-1]
        
        # Calculate prediction vs actual correlation
        # Shift actual prices by horizon days for comparison
        actual_future = data['Close'].shift(-int(horizon))
        predicted = data[pred_col]
        
        # Remove NaN values
        mask = ~(pd.isna(actual_future) | pd.isna(predicted))
        correlation = np.corrcoef(actual_future[mask], predicted[mask])[0, 1]
        
        # Calculate direction accuracy
        actual_direction = (actual_future > data['Close']).astype(int)
        pred_direction = (predicted > data['Close']).astype(int)
        direction_accuracy = (actual_direction == pred_direction)[mask].mean()
        
        # Calculate mean absolute error
        mae = np.mean(np.abs(actual_future[mask] - predicted[mask]))
        
        analysis[f'horizon_{horizon}'] = {
            'correlation': correlation,
            'direction_accuracy': direction_accuracy,
            'mae': mae
        }
    
    return analysis
