"""
Feature statistics module for computing per-feature statistics and identifying
statistically diverse features for TimeLLM (where each feature is processed independently).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import pearsonr


def compute_feature_statistics(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Compute comprehensive statistics for each feature.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
    
    Returns:
        DataFrame with statistics for each feature (rows=features, cols=statistics)
    """
    stats_dict = {}
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        series = df[col].dropna()
        if len(series) == 0:
            continue
        
        # Basic statistics
        stats_dict[col] = {
            'min': series.min(),
            'max': series.max(),
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'skewness': series.skew(),
            'kurtosis': series.kurtosis(),
            'range': series.max() - series.min(),
            'iqr': series.quantile(0.75) - series.quantile(0.25),
        }
        
        # Trend (slope of linear fit)
        x = np.arange(len(series))
        if len(series) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, series.values)
            stats_dict[col]['trend_slope'] = slope
            stats_dict[col]['trend_r2'] = r_value ** 2
        else:
            stats_dict[col]['trend_slope'] = 0.0
            stats_dict[col]['trend_r2'] = 0.0
        
        # Lag structure (autocorrelation at different lags)
        max_lag = min(10, len(series) // 4)
        autocorrs = []
        for lag in range(1, max_lag + 1):
            if len(series) > lag:
                corr = series.autocorr(lag=lag)
                if not np.isnan(corr):
                    autocorrs.append(corr)
        
        if len(autocorrs) > 0:
            stats_dict[col]['lag_mean'] = np.mean(autocorrs)
            stats_dict[col]['lag_max'] = np.max(autocorrs)
            stats_dict[col]['lag_std'] = np.std(autocorrs)
        else:
            stats_dict[col]['lag_mean'] = 0.0
            stats_dict[col]['lag_max'] = 0.0
            stats_dict[col]['lag_std'] = 0.0
        
        # Volatility regime (high vs low volatility periods)
        rolling_std = series.rolling(window=min(14, len(series) // 4)).std()
        stats_dict[col]['volatility_mean'] = rolling_std.mean() if len(rolling_std.dropna()) > 0 else series.std()
        
        # Value distribution percentiles
        stats_dict[col]['p25'] = series.quantile(0.25)
        stats_dict[col]['p75'] = series.quantile(0.75)
        stats_dict[col]['p10'] = series.quantile(0.10)
        stats_dict[col]['p90'] = series.quantile(0.90)
    
    return pd.DataFrame(stats_dict).T


def compute_statistical_similarity(stats_df: pd.DataFrame, 
                                  feature1: str, feature2: str,
                                  weight_dict: Optional[Dict[str, float]] = None) -> float:
    """
    Compute statistical similarity between two features using multiple statistics.
    
    Args:
        stats_df: DataFrame with statistics (from compute_feature_statistics)
        feature1: Name of first feature
        feature2: Name of second feature
        weight_dict: Optional weights for different statistics
    
    Returns:
        Similarity score between 0 and 1 (1 = identical, 0 = very different)
    """
    if weight_dict is None:
        weight_dict = {
            'mean': 0.15, 'std': 0.15, 'min': 0.1, 'max': 0.1, 'median': 0.1,
            'trend_slope': 0.15, 'lag_mean': 0.15, 'volatility_mean': 0.1
        }
    
    if feature1 not in stats_df.index or feature2 not in stats_df.index:
        return 0.0
    
    similarities = []
    weights = []
    
    for stat_name, weight in weight_dict.items():
        if stat_name not in stats_df.columns:
            continue
        
        val1 = stats_df.loc[feature1, stat_name]
        val2 = stats_df.loc[feature2, stat_name]
        
        # Normalize by range for comparison
        if stat_name in ['min', 'max', 'mean', 'median', 'trend_slope']:
            # Normalize by overall range
            all_vals = stats_df[stat_name].abs()
            max_range = all_vals.max() if all_vals.max() > 0 else 1.0
            if max_range > 0:
                diff = abs(val1 - val2) / max_range
                similarity = 1.0 - min(diff, 1.0)
            else:
                similarity = 1.0 if abs(val1 - val2) < 1e-8 else 0.0
        else:
            # For std, lag, volatility - use relative difference
            avg_val = (abs(val1) + abs(val2)) / 2.0
            if avg_val > 1e-8:
                diff = abs(val1 - val2) / avg_val
                similarity = 1.0 - min(diff, 1.0)
            else:
                similarity = 1.0 if abs(val1 - val2) < 1e-8 else 0.0
        
        similarities.append(similarity)
        weights.append(weight)
    
    if len(similarities) == 0:
        return 0.0
    
    # Weighted average
    total_weight = sum(weights)
    if total_weight > 0:
        weighted_sim = sum(s * w for s, w in zip(similarities, weights)) / total_weight
    else:
        weighted_sim = np.mean(similarities)
    
    return max(0.0, min(1.0, weighted_sim))


def cluster_features_by_statistics(stats_df: pd.DataFrame, 
                                   similarity_threshold: float = 0.8) -> List[List[str]]:
    """
    Cluster features by statistical similarity.
    
    Args:
        stats_df: DataFrame with statistics (from compute_feature_statistics)
        similarity_threshold: Threshold for considering features similar
    
    Returns:
        List of clusters, where each cluster is a list of feature names
    """
    features = stats_df.index.tolist()
    clusters = []
    assigned = set()
    
    for feature in features:
        if feature in assigned:
            continue
        
        # Start new cluster
        cluster = [feature]
        assigned.add(feature)
        
        # Find similar features
        for other_feature in features:
            if other_feature in assigned:
                continue
            
            similarity = compute_statistical_similarity(stats_df, feature, other_feature)
            if similarity >= similarity_threshold:
                cluster.append(other_feature)
                assigned.add(other_feature)
        
        clusters.append(cluster)
    
    return clusters


def compute_statistical_uniqueness(stats_df: pd.DataFrame, 
                                   feature: str,
                                   selected_features: List[str]) -> float:
    """
    Compute how unique a feature is compared to already selected features.
    
    Args:
        stats_df: DataFrame with statistics
        feature: Feature to evaluate
        selected_features: List of already selected features
    
    Returns:
        Uniqueness score (higher = more unique)
    """
    if len(selected_features) == 0:
        return 1.0
    
    if feature not in stats_df.index:
        return 0.0
    
    # Compute average similarity to selected features
    similarities = []
    for selected in selected_features:
        if selected in stats_df.index:
            sim = compute_statistical_similarity(stats_df, feature, selected)
            similarities.append(sim)
    
    if len(similarities) == 0:
        return 1.0
    
    avg_similarity = np.mean(similarities)
    uniqueness = 1.0 - avg_similarity  # Inverse of similarity
    
    return max(0.0, min(1.0, uniqueness))



