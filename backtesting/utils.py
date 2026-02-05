import pandas as pd
import backtrader as bt

def _parse_timestamp_series(ts: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(ts):
        vals = pd.to_numeric(ts, errors='coerce')
        maxv = vals.dropna().max()
        unit = 's'
        if pd.notna(maxv):
            if maxv > 1e14:
                unit = 'ns'
            elif maxv > 1e11:
                unit = 'ms'
        return pd.to_datetime(vals, unit=unit, errors='coerce')
    return pd.to_datetime(ts, errors='coerce')


def load_and_prepare_data(data_path, train_data_path=None):
    """Load and prepare data for backtrader"""
    df = pd.read_csv(data_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = _parse_timestamp_series(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    # Ensure we have the required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Find existing prediction columns
    prediction_cols = [col for col in df.columns if col.startswith('close_predicted_')]

    max_horizon = 0
    horizon_map = {}
    for col in prediction_cols:
        try:
            horizon = int(col.rsplit('_', 1)[1])
        except (IndexError, ValueError):
            horizon = 0
        horizon_map[col] = horizon
        if horizon > max_horizon:
            max_horizon = horizon

    if max_horizon <= 0:
        max_horizon = 1

    prediction_cols = sorted(prediction_cols, key=lambda c: horizon_map.get(c, 0))

    for col in prediction_cols:
        df[col] = df[col].shift(max_horizon)

    df = df.dropna().copy()
    df.attrs['max_prediction_horizon'] = max_horizon

    if train_data_path:
        train_df = pd.read_csv(train_data_path, usecols=['timestamp'])
        train_end = _parse_timestamp_series(train_df['timestamp']).max()
        if pd.notna(train_end):
            df = df[df.index > train_end].copy()
            df.attrs['train_cutoff_timestamp'] = train_end

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


def mean_directional_accuracy(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Calculate Mean Directional Accuracy (MDA) between actual and predicted series.

    Both series should be aligned (matching indices). NaNs are dropped.
    Returns a float between 0 and 1.
    """
    df = pd.concat([actual, predicted], axis=1, join="inner").dropna()
    if df.empty:
        return float("nan")

    actual_diff = df.iloc[:, 0].diff().fillna(0)
    predicted_diff = df.iloc[:, 1].diff().fillna(0)

    directional_match = (actual_diff * predicted_diff) >= 0
    return directional_match.mean()