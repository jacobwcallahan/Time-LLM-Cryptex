"""
Feature engineering module for generating candidate features from OHLCV and on-chain data.
Creates comprehensive feature pool for selection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def compute_returns(df: pd.DataFrame, periods: List[int] = [1, 2, 7, 14, 21, 30], 
                    price_col: str = 'close', log_returns: bool = True) -> pd.DataFrame:
    """
    Compute log returns or simple returns over multiple periods.
    
    Args:
        df: DataFrame with price column
        periods: List of periods to compute returns over
        price_col: Column name for price
        log_returns: If True, compute log returns; else simple returns
    
    Returns:
        DataFrame with return columns added
    """
    result = df.copy()
    for period in periods:
        if log_returns:
            result[f'returns_{period}d'] = np.log(df[price_col] / df[price_col].shift(period))
        else:
            result[f'returns_{period}d'] = (df[price_col] / df[price_col].shift(period)) - 1
    return result


def compute_volatility_features(df: pd.DataFrame, returns_col: str = 'returns_1d',
                               windows: List[int] = [7, 14, 21, 30]) -> pd.DataFrame:
    """
    Compute volatility measures: rolling std, realized volatility, ATR, Bollinger Bands.
    
    Args:
        df: DataFrame with returns column
        returns_col: Column name for returns
        windows: List of rolling window sizes
    
    Returns:
        DataFrame with volatility features added
    """
    result = df.copy()
    
    # Rolling standard deviation of returns
    for window in windows:
        result[f'volatility_{window}d'] = df[returns_col].rolling(window=window, min_periods=1).std()
    
    # Realized volatility (square root of rolling variance)
    for window in windows[:2]:  # Just 7 and 14 day
        result[f'realized_vol_{window}d'] = np.sqrt(df[returns_col].rolling(window=window, min_periods=1).var())
    
    # Average True Range (ATR)
    for window in [14, 21]:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        result[f'atr_{window}'] = true_range.rolling(window=window, min_periods=1).mean()
    
    # Bollinger Bands
    for window in [14, 20]:
        ma = df['close'].rolling(window=window, min_periods=1).mean()
        std = df['close'].rolling(window=window, min_periods=1).std()
        result[f'bb_upper_{window}'] = ma + (2 * std)
        result[f'bb_lower_{window}'] = ma - (2 * std)
        result[f'bb_pctb_{window}'] = (df['close'] - result[f'bb_lower_{window}']) / (result[f'bb_upper_{window}'] - result[f'bb_lower_{window}'] + 1e-8)
    
    # High-Low range normalized
    result['hl_range_pct'] = (df['high'] - df['low']) / (df['close'] + 1e-8)
    
    return result


def compute_technical_indicators(df: pd.DataFrame, price_col: str = 'close',
                                volume_col: str = 'volume') -> pd.DataFrame:
    """
    Compute technical indicators: RSI, MACD, Moving Averages, Stochastic.
    
    Args:
        df: DataFrame with price and volume
        price_col: Column name for price
        volume_col: Column name for volume
    
    Returns:
        DataFrame with technical indicators added
    """
    result = df.copy()
    
    # RSI (Relative Strength Index)
    for period in [14, 21]:
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        result[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df[price_col].ewm(span=12, adjust=False).mean()
    ema_26 = df[price_col].ewm(span=26, adjust=False).mean()
    result['macd'] = ema_12 - ema_26
    result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
    result['macd_histogram'] = result['macd'] - result['macd_signal']
    
    # Simple Moving Averages
    for period in [7, 14, 21, 50, 200]:
        result[f'sma_{period}'] = df[price_col].rolling(window=period, min_periods=1).mean()
        result[f'price_sma_{period}_ratio'] = df[price_col] / (result[f'sma_{period}'] + 1e-8)
    
    # Exponential Moving Averages
    for period in [12, 26]:
        result[f'ema_{period}'] = df[price_col].ewm(span=period, adjust=False).mean()
    
    # Stochastic Oscillator
    for period in [14]:
        low_min = df['low'].rolling(window=period, min_periods=1).min()
        high_max = df['high'].rolling(window=period, min_periods=1).max()
        result[f'stoch_k_{period}'] = 100 * (df[price_col] - low_min) / (high_max - low_min + 1e-8)
        result[f'stoch_d_{period}'] = result[f'stoch_k_{period}'].rolling(window=3, min_periods=1).mean()
    
    # ADX (Average Directional Index) - simplified version
    high_diff = df['high'].diff()
    low_diff = -df['low'].diff()
    pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    atr_14 = compute_volatility_features(df, 'returns_1d', [14])['atr_14']
    pos_di = 100 * pos_dm.rolling(window=14, min_periods=1).mean() / (atr_14 + 1e-8)
    neg_di = 100 * neg_dm.rolling(window=14, min_periods=1).mean() / (atr_14 + 1e-8)
    dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di + 1e-8)
    result['adx_14'] = dx.rolling(window=14, min_periods=1).mean()
    
    return result


def compute_volume_features(df: pd.DataFrame, price_col: str = 'close',
                           volume_col: str = 'volume') -> pd.DataFrame:
    """
    Compute volume-based features: OBV, volume ratios, volume-price correlations.
    
    Args:
        df: DataFrame with price and volume
        price_col: Column name for price
        volume_col: Column name for volume
    
    Returns:
        DataFrame with volume features added
    """
    result = df.copy()
    
    # On-Balance Volume (OBV)
    price_change = df[price_col].diff()
    result['obv'] = (np.sign(price_change) * df[volume_col]).fillna(0).cumsum()
    
    # Volume moving averages
    for window in [7, 14, 30]:
        result[f'volume_ma_{window}'] = df[volume_col].rolling(window=window, min_periods=1).mean()
        result[f'volume_ratio_{window}'] = df[volume_col] / (result[f'volume_ma_{window}'] + 1e-8)
    
    # Volume momentum
    result['volume_momentum'] = df[volume_col].pct_change(periods=7)
    
    # Volume-price trend
    result['volume_price_trend'] = df[volume_col] * df[price_col].pct_change()
    
    # Volume-weighted price (VWAP approximation)
    for window in [14, 30]:
        result[f'vwap_{window}'] = (df[price_col] * df[volume_col]).rolling(window=window, min_periods=1).sum() / (df[volume_col].rolling(window=window, min_periods=1).sum() + 1e-8)
    
    # Accumulation/Distribution Line
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-8)
    result['ad_line'] = (clv * df[volume_col]).fillna(0).cumsum()
    
    # Volume on returns (separate positive/negative)
    returns = df[price_col].pct_change()
    result['volume_on_positive_returns'] = df[volume_col].where(returns > 0, 0).rolling(window=14, min_periods=1).mean()
    result['volume_on_negative_returns'] = df[volume_col].where(returns < 0, 0).rolling(window=14, min_periods=1).mean()
    
    # Rolling correlation between price and volume
    for window in [14, 30]:
        result[f'price_volume_corr_{window}'] = df[price_col].rolling(window=window).corr(df[volume_col])
    
    return result


def compute_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute candlestick pattern features: body/wick ratios, price position.
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        DataFrame with candlestick features added
    """
    result = df.copy()
    
    # Body and wick sizes
    body = np.abs(df['close'] - df['open'])
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']
    total_range = df['high'] - df['low']
    
    result['body_size'] = body
    result['upper_wick_size'] = upper_wick
    result['lower_wick_size'] = lower_wick
    
    # Ratios
    result['body_wick_ratio'] = body / (upper_wick + lower_wick + 1e-8)
    result['body_range_ratio'] = body / (total_range + 1e-8)
    
    # Price position in range
    result['price_position'] = (df['close'] - df['low']) / (total_range + 1e-8)
    
    return result


def compute_temporal_features(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Compute time-based features: day of week, hour, weekend indicator.
    
    Args:
        df: DataFrame with timestamp column
        timestamp_col: Column name for timestamp
    
    Returns:
        DataFrame with temporal features added
    """
    result = df.copy()
    
    # Convert timestamp to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        if df[timestamp_col].dtype in ['int64', 'float64']:
            result['datetime'] = pd.to_datetime(df[timestamp_col], unit='s')
        else:
            result['datetime'] = pd.to_datetime(df[timestamp_col])
    else:
        result['datetime'] = df[timestamp_col]
    
    # Extract temporal features
    result['day_of_week'] = result['datetime'].dt.dayofweek
    result['hour'] = result['datetime'].dt.hour
    result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)
    result['month'] = result['datetime'].dt.month
    result['day_of_month'] = result['datetime'].dt.day
    
    # Cyclical encoding (sin/cos)
    result['day_of_week_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
    result['day_of_week_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
    result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
    result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
    
    # Drop temporary datetime column
    result = result.drop(columns=['datetime'], errors='ignore')
    
    return result


def compute_onchain_features(df: pd.DataFrame, transactions_df: pd.DataFrame,
                            addresses_df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Compute on-chain features: transaction metrics, address metrics, growth rates, ratios.
    
    Args:
        df: Main DataFrame with timestamp
        transactions_df: DataFrame with transaction data (date, transactions_per_day)
        addresses_df: DataFrame with address data (date, n_unique_addresses)
        timestamp_col: Column name for timestamp in all DataFrames
    
    Returns:
        DataFrame with on-chain features merged and computed
    """
    result = df.copy()
    
    # Prepare timestamps for merging
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        if df[timestamp_col].dtype in ['int64', 'float64']:
            df_dt = pd.to_datetime(df[timestamp_col], unit='s')
        else:
            df_dt = pd.to_datetime(df[timestamp_col])
    else:
        df_dt = df[timestamp_col]
    
    # Merge transactions
    if 'date' in transactions_df.columns:
        trans_dt = pd.to_datetime(transactions_df['date'])
    elif 'timestamp' in transactions_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(transactions_df['timestamp']):
            trans_dt = pd.to_datetime(transactions_df['timestamp'], unit='s')
        else:
            trans_dt = transactions_df['timestamp']
    else:
        return result  # Can't merge without date/timestamp
    
    # Create date column for merging
    result['_merge_date'] = df_dt.dt.date
    
    if 'date' in transactions_df.columns:
        transactions_df['_merge_date'] = trans_dt.dt.date
        merge_col = '_merge_date'
    else:
        transactions_df['_merge_date'] = trans_dt.dt.date
        merge_col = '_merge_date'
    
    # Merge on date
    if 'transactions_per_day' in transactions_df.columns:
        result = result.merge(
            transactions_df[[merge_col, 'transactions_per_day']],
            left_on='_merge_date', right_on=merge_col, how='left'
        )
        
        # Transaction features
        result['transactions'] = result['transactions_per_day']
        result['transactions_ma_7'] = result['transactions'].rolling(window=7, min_periods=1).mean()
        result['transactions_ma_30'] = result['transactions'].rolling(window=30, min_periods=1).mean()
        result['transactions_growth'] = (result['transactions'] / (result['transactions_ma_7'] + 1e-8)) - 1
        result['transactions_momentum'] = result['transactions'].pct_change(periods=7)
        result['transactions_ratio'] = result['transactions'] / (result['transactions_ma_30'] + 1e-8)
    
    # Merge addresses
    if 'date' in addresses_df.columns:
        addr_dt = pd.to_datetime(addresses_df['date'])
    elif 'timestamp' in addresses_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(addresses_df['timestamp']):
            addr_dt = pd.to_datetime(addresses_df['timestamp'], unit='s')
        else:
            addr_dt = addresses_df['timestamp']
    else:
        result = result.drop(columns=['_merge_date'], errors='ignore')
        return result
    
    if 'date' in addresses_df.columns:
        addresses_df['_merge_date'] = addr_dt.dt.date
        merge_col_addr = '_merge_date'
    else:
        addresses_df['_merge_date'] = addr_dt.dt.date
        merge_col_addr = '_merge_date'
    
    if 'n_unique_addresses' in addresses_df.columns:
        result = result.merge(
            addresses_df[[merge_col_addr, 'n_unique_addresses']],
            left_on='_merge_date', right_on=merge_col_addr, how='left', suffixes=('', '_addr')
        )
        
        # Address features
        result['addresses'] = result['n_unique_addresses']
        result['addresses_ma_7'] = result['addresses'].rolling(window=7, min_periods=1).mean()
        result['addresses_ma_30'] = result['addresses'].rolling(window=30, min_periods=1).mean()
        result['addresses_growth'] = (result['addresses'] / (result['addresses_ma_7'] + 1e-8)) - 1
        result['addresses_momentum'] = result['addresses'].pct_change(periods=7)
        result['addresses_ratio'] = result['addresses'] / (result['addresses_ma_30'] + 1e-8)
        
        # Transaction-to-address ratio
        if 'transactions' in result.columns:
            result['tx_address_ratio'] = result['transactions'] / (result['addresses'] + 1e-8)
    
    # Clean up merge columns
    result = result.drop(columns=['_merge_date', merge_col, merge_col_addr], errors='ignore')
    
    # Z-scores for on-chain metrics (normalized)
    for col in ['transactions', 'addresses']:
        if col in result.columns:
            ma = result[col].rolling(window=30, min_periods=1).mean()
            std = result[col].rolling(window=30, min_periods=1).std()
            result[f'{col}_zscore'] = (result[col] - ma) / (std + 1e-8)
    
    return result


def engineer_all_features(df: pd.DataFrame, transactions_df: Optional[pd.DataFrame] = None,
                         addresses_df: Optional[pd.DataFrame] = None,
                         timestamp_col: str = 'timestamp', 
                         price_col: str = 'close') -> pd.DataFrame:
    """
    Apply all feature engineering functions to create comprehensive feature pool.
    
    Args:
        df: Main DataFrame with OHLCV data
        transactions_df: Optional DataFrame with transaction data
        addresses_df: Optional DataFrame with address data
        timestamp_col: Column name for timestamp
        price_col: Column name for price
    
    Returns:
        DataFrame with all engineered features
    """
    result = df.copy()
    
    # Compute returns (must be first for other features)
    result = compute_returns(result, periods=[1, 2, 7, 14, 21, 30], price_col=price_col)
    
    # Volatility features
    result = compute_volatility_features(result, returns_col='returns_1d')
    
    # Technical indicators
    result = compute_technical_indicators(result, price_col=price_col)
    
    # Volume features
    result = compute_volume_features(result, price_col=price_col)
    
    # Candlestick features
    result = compute_candlestick_features(result)
    
    # Temporal features
    result = compute_temporal_features(result, timestamp_col=timestamp_col)
    
    # On-chain features (if provided)
    if transactions_df is not None or addresses_df is not None:
        result = compute_onchain_features(result, 
                                        transactions_df if transactions_df is not None else pd.DataFrame(),
                                        addresses_df if addresses_df is not None else pd.DataFrame(),
                                        timestamp_col=timestamp_col)
    
    return result



