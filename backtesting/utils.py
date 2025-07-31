import pandas as pd
import backtrader as bt

def load_and_prepare_data(data_path):
    """Load and prepare data for backtrader"""
    df = pd.read_csv(data_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    # Ensure we have the required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Find existing prediction columns
    prediction_cols = [col for col in df.columns if col.startswith('close_predicted_')]

    # Shift prediction columns by 1 since in any given row, close_pred 1 is for the same day close price
    for col in prediction_cols:
        df[col] = df[col].shift(1)
    df.dropna(inplace=True)

    # Create CustomPandasData class dynamically
    if prediction_cols:
        new_lines = tuple(prediction_cols)
        new_params = tuple((col, col) for col in prediction_cols)
        
        class CustomPandasData(bt.feeds.PandasData):
            lines = new_lines
            params = new_params

        data_feed_class = CustomPandasData
    else:
        # No prediction columns found, use standard PandasData
        data_feed_class = bt.feeds.PandasData
    
    return df, data_feed_class