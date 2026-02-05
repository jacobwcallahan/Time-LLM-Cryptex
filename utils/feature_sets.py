"""
Feature set configurations defining candidate features for each feature set type.
Each feature set targets different aspects of crypto price prediction with
statistically diverse features.
"""

from typing import List, Dict, Set


# Define feature categories for easy reference
FEATURE_CATEGORIES = {
    'ohlcv_basic': ['open', 'high', 'low', 'close', 'volume'],
    'returns': [f'returns_{p}d' for p in [1, 2, 7, 14, 21, 30]],
    'volatility': [
        f'volatility_{w}d' for w in [7, 14, 21, 30]
    ] + [
        f'realized_vol_{w}d' for w in [7, 14]
    ] + [
        f'atr_{w}' for w in [14, 21]
    ] + [
        f'bb_upper_{w}' for w in [14, 20]
    ] + [
        f'bb_lower_{w}' for w in [14, 20]
    ] + [
        f'bb_pctb_{w}' for w in [14, 20]
    ] + ['hl_range_pct'],
    'technical': [
        f'rsi_{p}' for p in [14, 21]
    ] + ['macd', 'macd_signal', 'macd_histogram'] + [
        f'sma_{p}' for p in [7, 14, 21, 50, 200]
    ] + [
        f'price_sma_{p}_ratio' for p in [7, 14, 21, 50, 200]
    ] + [
        f'ema_{p}' for p in [12, 26]
    ] + [
        f'stoch_k_{p}' for p in [14]
    ] + [
        f'stoch_d_{p}' for p in [14]
    ] + ['adx_14'],
    'volume': [
        'obv'
    ] + [
        f'volume_ma_{w}' for w in [7, 14, 30]
    ] + [
        f'volume_ratio_{w}' for w in [7, 14, 30]
    ] + [
        'volume_momentum', 'volume_price_trend'
    ] + [
        f'vwap_{w}' for w in [14, 30]
    ] + [
        'ad_line', 'volume_on_positive_returns', 'volume_on_negative_returns'
    ] + [
        f'price_volume_corr_{w}' for w in [14, 30]
    ],
    'candlestick': [
        'body_size', 'upper_wick_size', 'lower_wick_size',
        'body_wick_ratio', 'body_range_ratio', 'price_position'
    ],
    'onchain': [
        'transactions', 'transactions_ma_7', 'transactions_ma_30',
        'transactions_growth', 'transactions_momentum', 'transactions_ratio',
        'addresses', 'addresses_ma_7', 'addresses_ma_30',
        'addresses_growth', 'addresses_momentum', 'addresses_ratio',
        'tx_address_ratio', 'transactions_zscore', 'addresses_zscore'
    ],
    'temporal': [
        'day_of_week', 'hour', 'is_weekend', 'month', 'day_of_month',
        'day_of_week_sin', 'day_of_week_cos', 'hour_sin', 'hour_cos'
    ]
}


FEATURE_SET_CONFIGS = {
    'momentum': {
        'name': 'momentum',
        'description': 'Focus on price momentum and trend-following signals',
        'max_features': 10,
        'candidate_categories': ['ohlcv_basic', 'returns', 'technical', 'volume'],
        'candidate_features': [
            'close', 'volume',
            'returns_7d', 'returns_30d',
            'sma_14', 'sma_50',
            'price_sma_14_ratio', 'price_sma_50_ratio',
            'macd_signal',
            'rsi_14',
            'volume_ratio_14', 'volume_momentum'
        ],
        'required_features': ['close'],  # Target must be included
        'vif_threshold': 10.0,
        'corr_threshold': 0.8,
        'similarity_threshold': 0.8
    },
    
    'volatility': {
        'name': 'volatility',
        'description': 'Focus on volatility and risk metrics',
        'max_features': 10,
        'candidate_categories': ['ohlcv_basic', 'returns', 'volatility'],
        'candidate_features': [
            'close', 'high', 'low', 'volume',
            'returns_1d',
            'volatility_14d', 'volatility_30d',
            'atr_14', 'atr_21',
            'bb_pctb_14', 'bb_pctb_20',
            'realized_vol_14d',
            'hl_range_pct'
        ],
        'required_features': ['close'],
        'vif_threshold': 10.0,
        'corr_threshold': 0.8,
        'similarity_threshold': 0.8
    },
    
    'onchain_price': {
        'name': 'onchain_price',
        'description': 'Combine price data with on-chain fundamentals',
        'max_features': 10,
        'candidate_categories': ['ohlcv_basic', 'returns', 'onchain'],
        'candidate_features': [
            'close', 'volume',
            'returns_1d', 'returns_7d',
            'transactions', 'transactions_growth', 'transactions_momentum',
            'addresses', 'addresses_growth', 'addresses_momentum',
            'tx_address_ratio',
            'volatility_14d'
        ],
        'required_features': ['close'],
        'vif_threshold': 10.0,
        'corr_threshold': 0.8,
        'similarity_threshold': 0.8
    },
    
    'volume_price': {
        'name': 'volume_price',
        'description': 'Focus on volume-price dynamics and relationships',
        'max_features': 10,
        'candidate_categories': ['ohlcv_basic', 'returns', 'volume'],
        'candidate_features': [
            'close', 'volume',
            'returns_1d', 'returns_7d',
            'obv',
            'volume_ratio_14', 'volume_ratio_30',
            'volume_momentum',
            'volume_price_trend',
            'price_volume_corr_14', 'price_volume_corr_30',
            'vwap_14'
        ],
        'required_features': ['close'],
        'vif_threshold': 10.0,
        'corr_threshold': 0.8,
        'similarity_threshold': 0.8
    },
    
    'technical': {
        'name': 'technical',
        'description': 'Comprehensive technical analysis indicators',
        'max_features': 12,
        'candidate_categories': ['ohlcv_basic', 'returns', 'technical', 'volume'],
        'candidate_features': [
            'close', 'high', 'low', 'volume',
            'returns_7d',
            'rsi_14',
            'macd', 'macd_signal',
            'sma_14', 'sma_50',
            'price_sma_14_ratio', 'price_sma_50_ratio',
            'ema_12', 'ema_26',
            'bb_pctb_14',
            'stoch_k_14',
            'volume_ratio_14'
        ],
        'required_features': ['close'],
        'vif_threshold': 10.0,
        'corr_threshold': 0.8,
        'similarity_threshold': 0.8
    },
    
    'hybrid': {
        'name': 'hybrid',
        'description': 'Combine volatility measures with on-chain fundamentals',
        'max_features': 12,
        'candidate_categories': ['ohlcv_basic', 'returns', 'volatility', 'onchain'],
        'candidate_features': [
            'close', 'volume',
            'returns_1d', 'returns_7d',
            'volatility_14d', 'volatility_30d',
            'atr_14',
            'transactions', 'transactions_growth',
            'addresses', 'addresses_growth',
            'volatility_14d'  # Note: duplicate in candidate list, will be deduped
        ],
        'required_features': ['close'],
        'vif_threshold': 10.0,
        'corr_threshold': 0.8,
        'similarity_threshold': 0.8
    },
    
    'returns': {
        'name': 'returns',
        'description': 'Deep returns analysis with momentum and mean reversion',
        'max_features': 10,
        'candidate_categories': ['ohlcv_basic', 'returns', 'technical', 'volume'],
        'candidate_features': [
            'close', 'volume',
            'returns_1d', 'returns_7d', 'returns_14d', 'returns_30d',
            'volatility_7d', 'volatility_14d', 'volatility_30d',
            'price_sma_14_ratio', 'price_sma_30_ratio',
            'volume_on_positive_returns', 'volume_on_negative_returns',
            'rsi_14'
        ],
        'required_features': ['close'],
        'vif_threshold': 10.0,
        'corr_threshold': 0.8,
        'similarity_threshold': 0.8
    },
    
    'minimal': {
        'name': 'minimal',
        'description': 'Minimal high-value features with maximum information gain',
        'max_features': 8,
        'candidate_categories': ['ohlcv_basic', 'returns', 'volatility', 'onchain', 'volume', 'technical'],
        'candidate_features': [
            'close', 'volume',
            'returns_7d', 'returns_14d',
            'volatility_14d',
            'transactions_growth', 'addresses_growth',
            'volume_ratio_14',
            'rsi_14'
        ],
        'required_features': ['close'],
        'vif_threshold': 5.0,  # Stricter for minimal set
        'corr_threshold': 0.7,  # Stricter for minimal set
        'similarity_threshold': 0.8
    },
    
    'temporal': {
        'name': 'temporal',
        'description': 'Multi-scale temporal patterns with diverse statistical properties',
        'max_features': 10,
        'candidate_categories': ['ohlcv_basic', 'returns', 'volatility', 'volume', 'onchain', 'technical'],
        'candidate_features': [
            'close', 'volume',
            'returns_7d', 'returns_30d',
            'volatility_14d',
            'volume_ratio_14',
            'rsi_14',
            'transactions_growth',
            'addresses_growth'
        ],
        'required_features': ['close'],
        'vif_threshold': 10.0,
        'corr_threshold': 0.8,
        'similarity_threshold': 0.8
    }
}


def get_feature_set_config(set_name: str) -> Dict:
    """
    Get configuration for a feature set.
    
    Args:
        set_name: Name of feature set
    
    Returns:
        Configuration dictionary
    """
    if set_name not in FEATURE_SET_CONFIGS:
        raise ValueError(f"Unknown feature set: {set_name}. Available: {list(FEATURE_SET_CONFIGS.keys())}")
    
    return FEATURE_SET_CONFIGS[set_name].copy()


def get_all_feature_set_names() -> List[str]:
    """
    Get list of all available feature set names.
    
    Returns:
        List of feature set names
    """
    return list(FEATURE_SET_CONFIGS.keys())


def get_candidate_features_for_set(set_name: str, available_features: Set[str]) -> List[str]:
    """
    Get candidate features for a feature set, filtered to only those available.
    
    Args:
        set_name: Name of feature set
        available_features: Set of available feature names from data
    
    Returns:
        List of candidate feature names
    """
    config = get_feature_set_config(set_name)
    candidates = config['candidate_features']
    
    # Remove duplicates and filter to available features
    candidates_unique = list(dict.fromkeys(candidates))  # Preserve order, remove dupes
    candidates_filtered = [f for f in candidates_unique if f in available_features]
    
    return candidates_filtered



