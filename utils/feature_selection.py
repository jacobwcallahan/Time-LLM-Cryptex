"""
Feature selection module using VIF, correlation analysis, mutual information,
and statistical diversity for TimeLLM feature selection.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import warnings
warnings.filterwarnings('ignore')

from utils.feature_statistics import (
    compute_feature_statistics,
    cluster_features_by_statistics,
    compute_statistical_uniqueness
)


def compute_mutual_information(X: pd.DataFrame, y: pd.Series, 
                               random_state: int = 42) -> pd.Series:
    """
    Compute Mutual Information scores for all features.
    
    Args:
        X: DataFrame with features
        y: Target series
        random_state: Random seed for reproducibility
    
    Returns:
        Series with MI scores for each feature
    """
    # Handle NaN values
    X_clean = X.dropna(axis=0, how='any')
    y_clean = y.loc[X_clean.index]
    
    if len(X_clean) == 0:
        return pd.Series(index=X.columns, data=0.0)
    
    # Compute MI
    try:
        mi_scores = mutual_info_regression(
            X_clean, y_clean, 
            random_state=random_state,
            discrete_features=False
        )
    except Exception as e:
        print(f"Warning: Error computing MI: {e}")
        mi_scores = np.zeros(len(X.columns))
    
    return pd.Series(mi_scores, index=X.columns)


def compute_vif(X: pd.DataFrame, max_vif: float = 50.0) -> pd.Series:
    """
    Compute Variance Inflation Factor (VIF) for all features.
    
    Args:
        X: DataFrame with features
        max_vif: Maximum VIF value before considering feature problematic
    
    Returns:
        Series with VIF scores for each feature
    """
    # Handle NaN values by forward fill then drop remaining
    X_clean = X.ffill().dropna(axis=0, how='any')
    
    if len(X_clean) == 0 or len(X_clean.columns) == 0:
        return pd.Series(index=X.columns, data=0.0)
    
    # Check for constant columns
    constant_cols = X_clean.columns[X_clean.nunique() <= 1].tolist()
    X_vif = X_clean.drop(columns=constant_cols, errors='ignore')
    
    if len(X_vif.columns) == 0:
        vif_scores = pd.Series(index=X.columns, data=max_vif)
        return vif_scores
    
    # Add constant for VIF computation
    try:
        X_with_const = add_constant(X_vif)
        
        # Compute VIF
        vif_data = []
        vif_features = []
        
        for i in range(1, X_with_const.shape[1]):  # Skip constant column
            try:
                vif = variance_inflation_factor(X_with_const.values, i)
                if np.isinf(vif) or np.isnan(vif):
                    vif = max_vif
                vif_data.append(vif)
                vif_features.append(X_with_const.columns[i])
            except Exception:
                vif_data.append(max_vif)
                vif_features.append(X_with_const.columns[i])
        
        vif_series = pd.Series(vif_data, index=vif_features)
        
        # Add back constant columns with high VIF
        for col in constant_cols:
            if col not in vif_series.index:
                vif_series[col] = max_vif
        
        # Ensure all original columns are present
        result = pd.Series(index=X.columns, data=max_vif)
        result[vif_series.index] = vif_series
        
        return result
        
    except Exception as e:
        print(f"Warning: Error computing VIF: {e}")
        return pd.Series(index=X.columns, data=max_vif)


def compute_feature_correlations(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise correlation matrix for features.
    
    Args:
        X: DataFrame with features
    
    Returns:
        Correlation matrix
    """
    # Handle NaN values
    X_clean = X.ffill().dropna(axis=0, how='any')
    
    if len(X_clean) == 0:
        return pd.DataFrame(index=X.columns, columns=X.columns, data=0.0)
    
    return X_clean.corr()


def filter_by_vif(features: List[str], X: pd.DataFrame, 
                  vif_threshold: float = 10.0) -> List[str]:
    """
    Remove features with high VIF scores.
    
    Args:
        features: List of feature names
        X: DataFrame with features
        vif_threshold: Maximum VIF threshold
    
    Returns:
        List of features with VIF below threshold
    """
    if len(features) == 0:
        return []
    
    X_subset = X[features]
    vif_scores = compute_vif(X_subset)
    
    valid_features = [f for f in features if f in vif_scores.index and vif_scores[f] < vif_threshold]
    
    return valid_features


def filter_by_correlation(features: List[str], X: pd.DataFrame,
                         corr_threshold: float = 0.8) -> List[str]:
    """
    Remove highly correlated features, keeping the one with highest MI.
    
    Args:
        features: List of feature names
        X: DataFrame with features
        corr_threshold: Maximum correlation threshold
    
    Returns:
        List of features with low inter-correlation
    """
    if len(features) <= 1:
        return features
    
    X_subset = X[features]
    corr_matrix = compute_feature_correlations(X_subset)
    
    # Find highly correlated pairs
    to_remove = set()
    
    for i, feat1 in enumerate(features):
        if feat1 in to_remove:
            continue
        
        for feat2 in features[i+1:]:
            if feat2 in to_remove:
                continue
            
            if feat1 in corr_matrix.index and feat2 in corr_matrix.columns:
                corr_val = abs(corr_matrix.loc[feat1, feat2])
                if corr_val >= corr_threshold:
                    # Keep feature with higher variance (more informative)
                    var1 = X_subset[feat1].var()
                    var2 = X_subset[feat2].var()
                    if var1 < var2:
                        to_remove.add(feat1)
                    else:
                        to_remove.add(feat2)
    
    return [f for f in features if f not in to_remove]


def select_features_for_timellm(candidate_features: List[str],
                               X: pd.DataFrame,
                               y: pd.Series,
                               max_features: int = 10,
                               vif_threshold: float = 10.0,
                               corr_threshold: float = 0.8,
                               similarity_threshold: float = 0.8,
                               random_state: int = 42) -> List[str]:
    """
    Select optimal features for TimeLLM using statistical diversity + MI + correlation.
    
    Args:
        candidate_features: List of candidate feature names
        X: DataFrame with all features
        y: Target series
        max_features: Maximum number of features to select
        vif_threshold: Maximum VIF threshold
        corr_threshold: Maximum correlation threshold
        similarity_threshold: Statistical similarity threshold for clustering
        random_state: Random seed
    
    Returns:
        List of selected feature names
    """
    if len(candidate_features) == 0:
        return []
    
    # Filter candidates to only those that exist in X
    candidate_features = [f for f in candidate_features if f in X.columns]
    
    if len(candidate_features) == 0:
        return []
    
    X_candidates = X[candidate_features]
    
    # Step 1: Compute MI scores
    mi_scores = compute_mutual_information(X_candidates, y, random_state=random_state)
    
    # Step 2: Compute feature statistics for statistical diversity
    stats_df = compute_feature_statistics(X_candidates, candidate_features)
    
    # Step 3: Cluster features by statistical similarity
    clusters = cluster_features_by_statistics(stats_df, similarity_threshold)
    
    # Step 4: Select one feature per cluster (highest MI)
    selected = []
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        
        # Get features in cluster that exist in candidates
        cluster_features = [f for f in cluster if f in candidate_features]
        if len(cluster_features) == 0:
            continue
        
        # Select feature with highest MI
        cluster_mi = mi_scores[cluster_features]
        best_feature = cluster_mi.idxmax()
        selected.append(best_feature)
    
    # Step 5: If not enough features, add most unique features
    remaining = [f for f in candidate_features if f not in selected]
    remaining.sort(key=lambda f: mi_scores.get(f, 0), reverse=True)
    
    while len(selected) < max_features and len(remaining) > 0:
        # Find most statistically unique feature
        best_candidate = None
        best_score = -1
        
        for feature in remaining[:min(50, len(remaining))]:  # Limit search
            uniqueness = compute_statistical_uniqueness(stats_df, feature, selected)
            mi = mi_scores.get(feature, 0)
            # Combined score: uniqueness + MI
            score = 0.6 * uniqueness + 0.4 * (mi / (mi_scores.max() + 1e-8))
            
            if score > best_score:
                best_score = score
                best_candidate = feature
        
        if best_candidate:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            break
    
    # Step 6: Final filter by VIF and correlation
    if len(selected) > 1:
        selected = filter_by_vif(selected, X_candidates, vif_threshold)
        selected = filter_by_correlation(selected, X_candidates, corr_threshold)
    
    # Ensure we don't exceed max_features
    selected = selected[:max_features]
    
    return selected


def select_top_features_by_mi(candidate_features: List[str],
                              X: pd.DataFrame,
                              y: pd.Series,
                              top_k: int = 10,
                              random_state: int = 42) -> List[str]:
    """
    Simple selection: top K features by Mutual Information.
    
    Args:
        candidate_features: List of candidate feature names
        X: DataFrame with all features
        y: Target series
        top_k: Number of top features to select
        random_state: Random seed
    
    Returns:
        List of top K feature names
    """
    candidate_features = [f for f in candidate_features if f in X.columns]
    if len(candidate_features) == 0:
        return []
    
    X_candidates = X[candidate_features]
    mi_scores = compute_mutual_information(X_candidates, y, random_state)
    
    top_features = mi_scores.nlargest(top_k).index.tolist()
    return top_features



