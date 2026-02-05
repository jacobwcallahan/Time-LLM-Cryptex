#!/usr/bin/env python3
"""
Main preprocessing script to generate feature-engineered datasets for TimeLLM.
Creates multiple CSV files with different feature sets, each optimized for
statistical diversity and information gain.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from utils.feature_engineer import engineer_all_features
from utils.feature_selection import (
    select_features_for_timellm,
    compute_mutual_information,
    compute_vif,
    compute_feature_correlations
)
from utils.feature_statistics import compute_feature_statistics
from utils.feature_sets import get_all_feature_set_names, get_feature_set_config, get_candidate_features_for_set
from utils.prompt_generator import generate_and_save_prompt


# Validation tracking
validation_results = {'errors': [], 'warnings': [], 'info': []}


def validate_input_data(ohlcv_df: pd.DataFrame, 
                       transactions_df: Optional[pd.DataFrame],
                       addresses_df: Optional[pd.DataFrame],
                       target_col: str = 'close') -> bool:
    """
    Validate input data files.
    
    Returns:
        True if valid, False otherwise
    """
    is_valid = True
    
    # Validate OHLCV data
    required_ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_ohlcv_cols if col not in ohlcv_df.columns]
    if missing_cols:
        error_msg = f"OHLCV data missing required columns: {missing_cols}"
        validation_results['errors'].append(error_msg)
        print(f"ERROR: {error_msg}")
        is_valid = False
    else:
        validation_results['info'].append(f"OHLCV data has all required columns: {required_ohlcv_cols}")
    
    # Check if DataFrame is empty
    if len(ohlcv_df) == 0:
        error_msg = "OHLCV DataFrame is empty"
        validation_results['errors'].append(error_msg)
        print(f"ERROR: {error_msg}")
        is_valid = False
    else:
        validation_results['info'].append(f"OHLCV data has {len(ohlcv_df)} rows")
    
    # Check target column exists
    if target_col not in ohlcv_df.columns:
        error_msg = f"Target column '{target_col}' not found in OHLCV data"
        validation_results['errors'].append(error_msg)
        print(f"ERROR: {error_msg}")
        is_valid = False
    
    # Check for excessive NaN in OHLCV
    nan_counts = ohlcv_df[required_ohlcv_cols].isna().sum()
    nan_pct = (nan_counts / len(ohlcv_df)) * 100
    high_nan_cols = nan_pct[nan_pct > 50].index.tolist()
    if high_nan_cols:
        warning_msg = f"OHLCV columns with >50% NaN: {high_nan_cols}"
        validation_results['warnings'].append(warning_msg)
        print(f"WARNING: {warning_msg}")
    
    # Validate timestamp column
    if 'timestamp' in ohlcv_df.columns:
        try:
            # Try to convert to numeric (Unix timestamp) or datetime
            pd.to_numeric(ohlcv_df['timestamp'], errors='raise')
            validation_results['info'].append("Timestamp column is numeric (Unix format)")
        except (ValueError, TypeError):
            try:
                pd.to_datetime(ohlcv_df['timestamp'])
                validation_results['info'].append("Timestamp column is datetime format")
            except:
                warning_msg = "Timestamp column format may be invalid"
                validation_results['warnings'].append(warning_msg)
                print(f"WARNING: {warning_msg}")
    
    # Validate transactions data if provided
    if transactions_df is not None:
        if len(transactions_df) == 0:
            warning_msg = "Transactions DataFrame is empty"
            validation_results['warnings'].append(warning_msg)
            print(f"WARNING: {warning_msg}")
        else:
            validation_results['info'].append(f"Transactions data has {len(transactions_df)} rows")
            
            # Check for required columns
            if 'transactions_per_day' not in transactions_df.columns and 'transactions' not in transactions_df.columns:
                warning_msg = "Transactions data missing expected columns (transactions_per_day or transactions)"
                validation_results['warnings'].append(warning_msg)
                print(f"WARNING: {warning_msg}")
    
    # Validate addresses data if provided
    if addresses_df is not None:
        if len(addresses_df) == 0:
            warning_msg = "Addresses DataFrame is empty"
            validation_results['warnings'].append(warning_msg)
            print(f"WARNING: {warning_msg}")
        else:
            validation_results['info'].append(f"Addresses data has {len(addresses_df)} rows")
            
            # Check for required columns
            if 'n_unique_addresses' not in addresses_df.columns and 'addresses' not in addresses_df.columns:
                warning_msg = "Addresses data missing expected columns (n_unique_addresses or addresses)"
                validation_results['warnings'].append(warning_msg)
                print(f"WARNING: {warning_msg}")
    
    return is_valid


def validate_feature_dataframe(df: pd.DataFrame, target_col: str = 'close') -> bool:
    """
    Validate feature-engineered DataFrame.
    
    Returns:
        True if valid, False otherwise
    """
    is_valid = True
    
    # Check DataFrame is not empty
    if len(df) == 0:
        error_msg = "Feature DataFrame is empty"
        validation_results['errors'].append(error_msg)
        print(f"ERROR: {error_msg}")
        is_valid = False
    else:
        validation_results['info'].append(f"Feature DataFrame has {len(df)} rows and {len(df.columns)} columns")
    
    # Check timestamp exists
    if 'timestamp' not in df.columns:
        error_msg = "Feature DataFrame missing 'timestamp' column"
        validation_results['errors'].append(error_msg)
        print(f"ERROR: {error_msg}")
        is_valid = False
    
    # Check target column exists
    if target_col not in df.columns:
        error_msg = f"Feature DataFrame missing target column '{target_col}'"
        validation_results['errors'].append(error_msg)
        print(f"ERROR: {error_msg}")
        is_valid = False
    
    # Check for columns with all NaN values
    all_nan_cols = df.columns[df.isna().all()].tolist()
    if all_nan_cols:
        error_msg = f"Columns with all NaN values: {all_nan_cols}"
        validation_results['errors'].append(error_msg)
        print(f"ERROR: {error_msg}")
        is_valid = False
    
    # Check for columns with high NaN percentage (>50%)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        nan_pct = (df[numeric_cols].isna().sum() / len(df)) * 100
        high_nan_cols = nan_pct[nan_pct > 50].index.tolist()
        if high_nan_cols:
            warning_msg = f"Columns with >50% NaN: {high_nan_cols}"
            validation_results['warnings'].append(warning_msg)
            print(f"WARNING: {warning_msg}")
    
    # Check for infinite values in numeric columns
    if len(numeric_cols) > 0:
        inf_cols = []
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                inf_cols.append(col)
        if inf_cols:
            warning_msg = f"Columns with infinite values: {inf_cols}"
            validation_results['warnings'].append(warning_msg)
            print(f"WARNING: {warning_msg}")
    
    # Verify numeric columns are actually numeric
    non_numeric_cols = []
    for col in df.columns:
        if col not in ['timestamp']:
            try:
                pd.to_numeric(df[col], errors='raise')
            except (ValueError, TypeError):
                non_numeric_cols.append(col)
    if non_numeric_cols:
        warning_msg = f"Columns that should be numeric but aren't: {non_numeric_cols}"
        validation_results['warnings'].append(warning_msg)
        print(f"WARNING: {warning_msg}")
    
    return is_valid


def validate_feature_set_output(csv_path: str, target_col: str = 'close', 
                                expected_features: Optional[List[str]] = None) -> bool:
    """
    Validate output CSV file for a feature set.
    
    Returns:
        True if valid, False otherwise
    """
    is_valid = True
    
    # Check file exists
    if not os.path.exists(csv_path):
        error_msg = f"Output CSV file not found: {csv_path}"
        validation_results['errors'].append(error_msg)
        print(f"ERROR: {error_msg}")
        return False
    
    # Try to read the file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        error_msg = f"Failed to read CSV file {csv_path}: {e}"
        validation_results['errors'].append(error_msg)
        print(f"ERROR: {error_msg}")
        return False
    
    # Check row count
    if len(df) == 0:
        error_msg = f"Output CSV {csv_path} is empty"
        validation_results['errors'].append(error_msg)
        print(f"ERROR: {error_msg}")
        is_valid = False
    
    # Check required columns
    if 'timestamp' not in df.columns:
        error_msg = f"Output CSV {csv_path} missing 'timestamp' column"
        validation_results['errors'].append(error_msg)
        print(f"ERROR: {error_msg}")
        is_valid = False
    
    if target_col not in df.columns:
        error_msg = f"Output CSV {csv_path} missing target column '{target_col}'"
        validation_results['errors'].append(error_msg)
        print(f"ERROR: {error_msg}")
        is_valid = False
    
    # Check column ordering (timestamp first, target last)
    if len(df.columns) > 0:
        if df.columns[0] != 'timestamp':
            warning_msg = f"Output CSV {csv_path} timestamp is not first column (found: {df.columns[0]})"
            validation_results['warnings'].append(warning_msg)
            print(f"WARNING: {warning_msg}")
        
        if df.columns[-1] != target_col:
            warning_msg = f"Output CSV {csv_path} target column is not last column (found: {df.columns[-1]})"
            validation_results['warnings'].append(warning_msg)
            print(f"WARNING: {warning_msg}")
    
    # Check for NaN values
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0].index.tolist()
    if nan_cols:
        warning_msg = f"Output CSV {csv_path} has NaN values in columns: {nan_cols}"
        validation_results['warnings'].append(warning_msg)
        print(f"WARNING: {warning_msg}")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        inf_cols = []
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                inf_cols.append(col)
        if inf_cols:
            warning_msg = f"Output CSV {csv_path} has infinite values in columns: {inf_cols}"
            validation_results['warnings'].append(warning_msg)
            print(f"WARNING: {warning_msg}")
    
    # Check feature count if expected_features provided
    if expected_features is not None:
        feature_cols = [col for col in df.columns if col not in ['timestamp', target_col]]
        if len(feature_cols) != len(expected_features):
            warning_msg = f"Output CSV {csv_path} has {len(feature_cols)} features, expected {len(expected_features)}"
            validation_results['warnings'].append(warning_msg)
            print(f"WARNING: {warning_msg}")
    
    # Check for duplicate columns
    if len(df.columns) != len(set(df.columns)):
        duplicates = [col for col in df.columns if df.columns.tolist().count(col) > 1]
        error_msg = f"Output CSV {csv_path} has duplicate columns: {duplicates}"
        validation_results['errors'].append(error_msg)
        print(f"ERROR: {error_msg}")
        is_valid = False
    
    return is_valid


def validate_prompt_file(prompt_path: str) -> bool:
    """
    Validate prompt file was created correctly.
    
    Returns:
        True if valid, False otherwise
    """
    is_valid = True
    
    # Check file exists
    if not os.path.exists(prompt_path):
        error_msg = f"Prompt file not found: {prompt_path}"
        validation_results['errors'].append(error_msg)
        print(f"ERROR: {error_msg}")
        return False
    
    # Check file is not empty
    try:
        with open(prompt_path, 'r') as f:
            content = f.read().strip()
            if len(content) == 0:
                error_msg = f"Prompt file {prompt_path} is empty"
                validation_results['errors'].append(error_msg)
                print(f"ERROR: {error_msg}")
                is_valid = False
            else:
                validation_results['info'].append(f"Prompt file {prompt_path} has {len(content)} characters")
    except Exception as e:
        error_msg = f"Failed to read prompt file {prompt_path}: {e}"
        validation_results['errors'].append(error_msg)
        print(f"ERROR: {error_msg}")
        is_valid = False
    
    return is_valid


def report_validation_results():
    """
    Print comprehensive validation summary report.
    """
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    print(f"\nErrors: {len(validation_results['errors'])}")
    if validation_results['errors']:
        for i, error in enumerate(validation_results['errors'], 1):
            print(f"  {i}. {error}")
    
    print(f"\nWarnings: {len(validation_results['warnings'])}")
    if validation_results['warnings']:
        for i, warning in enumerate(validation_results['warnings'], 1):
            print(f"  {i}. {warning}")
    
    print(f"\nInfo: {len(validation_results['info'])} items")
    if len(validation_results['info']) > 0:
        print(f"  (First 5: {validation_results['info'][:5]})")
        if len(validation_results['info']) > 5:
            print(f"  ... and {len(validation_results['info']) - 5} more")
    
    print("\n" + "="*60)
    
    if len(validation_results['errors']) > 0:
        print("VALIDATION FAILED: Please fix errors above")
        return False
    elif len(validation_results['warnings']) > 0:
        print("VALIDATION PASSED WITH WARNINGS")
        return True
    else:
        print("VALIDATION PASSED")
        return True


def load_data(root_path: str, data_path: str, 
             transactions_path: Optional[str] = None,
             addresses_path: Optional[str] = None) -> tuple:
    """
    Load OHLCV data and optional on-chain data.
    
    Args:
        root_path: Root directory for data
        data_path: Path to OHLCV CSV file
        transactions_path: Optional path to transactions CSV
        addresses_path: Optional path to addresses CSV
    
    Returns:
        Tuple of (ohlcv_df, transactions_df, addresses_df)
    """
    # Load OHLCV data
    ohlcv_file = os.path.join(root_path, data_path)
    print(f"Loading OHLCV data from: {ohlcv_file}")
    ohlcv_df = pd.read_csv(ohlcv_file)
    
    # Load on-chain data if provided
    transactions_df = None
    if transactions_path:
        trans_file = os.path.join(root_path, transactions_path)
        if os.path.exists(trans_file):
            print(f"Loading transactions data from: {trans_file}")
            transactions_df = pd.read_csv(trans_file)
        else:
            print(f"Warning: Transactions file not found: {trans_file}")
    
    addresses_df = None
    if addresses_path:
        addr_file = os.path.join(root_path, addresses_path)
        if os.path.exists(addr_file):
            print(f"Loading addresses data from: {addr_file}")
            addresses_df = pd.read_csv(addr_file)
        else:
            print(f"Warning: Addresses file not found: {addr_file}")
    
    print(f"Loaded {len(ohlcv_df)} rows of OHLCV data")
    
    # Validate input data
    print("\n" + "="*60)
    print("Validating input data...")
    print("="*60)
    if not validate_input_data(ohlcv_df, transactions_df, addresses_df, target_col='close'):
        print("WARNING: Input data validation failed, but continuing...")
    
    return ohlcv_df, transactions_df, addresses_df


def engineer_candidate_features(ohlcv_df: pd.DataFrame,
                               transactions_df: Optional[pd.DataFrame],
                               addresses_df: Optional[pd.DataFrame],
                               target_col: str = 'close') -> pd.DataFrame:
    """
    Generate all candidate features from raw data.
    
    Args:
        ohlcv_df: OHLCV DataFrame
        transactions_df: Optional transactions DataFrame
        addresses_df: Optional addresses DataFrame
        target_col: Target column name
    
    Returns:
        DataFrame with all engineered features
    """
    print("\n" + "="*60)
    print("Engineering candidate features...")
    print("="*60)
    
    df_features = engineer_all_features(
        ohlcv_df,
        transactions_df=transactions_df,
        addresses_df=addresses_df,
        timestamp_col='timestamp',
        price_col=target_col
    )
    
    print(f"Generated {len(df_features.columns)} total features")
    print(f"Data shape: {df_features.shape}")
    
    # Handle NaN values (causal fill only, then drop remaining)
    initial_rows = len(df_features)
    df_features = df_features.ffill()
    
    # Drop rows that still have NaN (usually initial rows after rolling windows)
    df_features = df_features.dropna()
    dropped_rows = initial_rows - len(df_features)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows with remaining NaN values")
    
    # Validate feature DataFrame
    print("\n" + "="*60)
    print("Validating feature-engineered data...")
    print("="*60)
    if not validate_feature_dataframe(df_features, target_col=target_col):
        print("WARNING: Feature DataFrame validation failed, but continuing...")
    
    return df_features


def compute_global_statistics(df_features: pd.DataFrame, target_col: str,
                             feature_cols: List[str],
                             train_size: float = 0.8) -> Dict:
    """
    Compute global statistics (MI, VIF, correlations) for all features.
    Uses only training set to avoid data leakage.
    
    Args:
        df_features: DataFrame with all features
        target_col: Target column name
        feature_cols: List of feature column names
        train_size: Fraction of data to use for training
    
    Returns:
        Dictionary with statistics
    """
    print("\n" + "="*60)
    print("Computing global feature statistics...")
    print("="*60)
    
    # Split into train/test for statistics computation
    split_idx = int(len(df_features) * train_size)
    df_train = df_features.iloc[:split_idx].copy()
    df_test = df_features.iloc[split_idx:].copy()
    
    print(f"Using {len(df_train)} rows for statistics computation (train set)")
    
    # Prepare feature matrix and target
    feature_cols_available = [f for f in feature_cols if f in df_train.columns]
    X_train = df_train[feature_cols_available]
    y_train = df_train[target_col]
    
    # Compute MI scores
    print("Computing Mutual Information scores...")
    mi_scores = compute_mutual_information(X_train, y_train)
    
    # Compute VIF scores
    print("Computing VIF scores...")
    vif_scores = compute_vif(X_train)
    
    # Compute correlation matrix
    print("Computing feature correlations...")
    corr_matrix = compute_feature_correlations(X_train)
    
    # Compute feature statistics
    print("Computing feature statistics...")
    feature_stats = compute_feature_statistics(X_train, feature_cols_available)
    
    return {
        'mi_scores': mi_scores,
        'vif_scores': vif_scores,
        'corr_matrix': corr_matrix,
        'feature_stats': feature_stats,
        'feature_cols': feature_cols_available
    }


def process_feature_set(set_name: str,
                       df_features: pd.DataFrame,
                       target_col: str,
                       global_stats: Dict,
                       output_dir: str,
                       prompt_dir: str,
                       frequency: str = 'D') -> Dict:
    """
    Process a single feature set: select features, save CSV, generate prompt.
    
    Args:
        set_name: Name of feature set
        df_features: DataFrame with all features
        target_col: Target column name
        global_stats: Global statistics dictionary
        output_dir: Output directory for CSVs
        frequency: Frequency suffix for filename (D, h, etc.)
    
    Returns:
        Dictionary with processing results
    """
    print("\n" + "="*60)
    print(f"Processing feature set: {set_name}")
    print("="*60)
    
    # Get feature set configuration
    config = get_feature_set_config(set_name)
    
    # Get candidate features for this set
    available_features = set(global_stats['feature_cols'])
    candidate_features = get_candidate_features_for_set(set_name, available_features)
    
    print(f"Candidate features for {set_name}: {len(candidate_features)}")
    
    if len(candidate_features) == 0:
        print(f"Warning: No candidate features available for {set_name}")
        return {'set_name': set_name, 'selected_features': [], 'status': 'failed'}
    
    # Select optimal features
    split_idx = int(len(df_features) * 0.8)
    df_train = df_features.iloc[:split_idx].copy()
    
    X_train = df_train[candidate_features]
    y_train = df_train[target_col]
    validation_results['info'].append(
        f"Feature selection for {set_name} computed on train slice only: rows [0:{split_idx})"
    )
    
    selected_features = select_features_for_timellm(
        candidate_features=candidate_features,
        X=X_train,
        y=y_train,
        max_features=config['max_features'],
        vif_threshold=config['vif_threshold'],
        corr_threshold=config['corr_threshold'],
        similarity_threshold=config['similarity_threshold']
    )
    
    # Validate feature selection
    if len(selected_features) == 0:
        error_msg = f"No features selected for {set_name}"
        validation_results['errors'].append(error_msg)
        print(f"ERROR: {error_msg}")
        return {'set_name': set_name, 'selected_features': [], 'status': 'failed'}
    
    # Ensure target and required features are included
    required = config.get('required_features', [])
    for req in required:
        if req in df_features.columns and req not in selected_features:
            selected_features.insert(0, req)  # Add at beginning
    
    # Ensure target is included
    if target_col not in selected_features:
        selected_features.insert(0, target_col)
    
    # Limit to max_features
    selected_features = selected_features[:config['max_features'] + 1]  # +1 for target
    
    # Validate feature count is reasonable
    feature_count = len([f for f in selected_features if f != target_col])
    if feature_count == 0:
        error_msg = f"Feature set {set_name} has no features (only target)"
        validation_results['errors'].append(error_msg)
        print(f"ERROR: {error_msg}")
        return {'set_name': set_name, 'selected_features': selected_features, 'status': 'failed'}
    elif feature_count < 3:
        warning_msg = f"Feature set {set_name} has very few features ({feature_count}), may not provide enough diversity"
        validation_results['warnings'].append(warning_msg)
        print(f"WARNING: {warning_msg}")
    elif feature_count > config['max_features'] + 5:  # Allow some tolerance
        warning_msg = f"Feature set {set_name} has more features ({feature_count}) than expected max ({config['max_features']})"
        validation_results['warnings'].append(warning_msg)
        print(f"WARNING: {warning_msg}")
    
    print(f"Selected {len(selected_features)} features for {set_name} ({feature_count} features + target)")
    print(f"Features: {', '.join(selected_features[:10])}{'...' if len(selected_features) > 10 else ''}")
    
    # Create output DataFrame with selected features
    output_cols = ['timestamp'] + [f for f in selected_features if f != 'timestamp']
    output_df = df_features[output_cols].copy()
    
    # Ensure target is last column (except timestamp)
    if target_col in output_cols:
        cols = ['timestamp'] + [f for f in output_cols if f not in ['timestamp', target_col]] + [target_col]
        output_df = output_df[cols]
    
    # Save CSV
    csv_filename = f'candlesticks-{frequency}_features_{set_name}.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    output_df.to_csv(csv_path, index=False)
    print(f"Saved CSV to: {csv_path}")
    
    # Validate output CSV
    print(f"Validating output CSV for {set_name}...")
    if not validate_feature_set_output(csv_path, target_col=target_col, expected_features=selected_features):
        print(f"WARNING: Output CSV validation failed for {set_name}, but continuing...")
    
    # Generate and save prompt
    prompt_path = generate_and_save_prompt(
        feature_set_name=set_name,
        features=selected_features,
        output_dir=prompt_dir
    )
    print(f"Saved prompt to: {prompt_path}")
    
    # Validate prompt file
    print(f"Validating prompt file for {set_name}...")
    if not validate_prompt_file(prompt_path):
        print(f"WARNING: Prompt file validation failed for {set_name}, but continuing...")
    
    # Compute final statistics for selected features
    # Only include features that exist in global_stats (exclude target and required features that might not have stats)
    features_with_stats = [f for f in selected_features if f in global_stats['mi_scores'].index]
    final_mi = global_stats['mi_scores'][features_with_stats].to_dict() if len(features_with_stats) > 0 else {}
    final_vif = global_stats['vif_scores'][features_with_stats].to_dict() if len(features_with_stats) > 0 else {}
    
    return {
        'set_name': set_name,
        'selected_features': selected_features,
        'num_features': len(selected_features),
        'csv_path': csv_path,
        'prompt_path': prompt_path,
        'mi_scores': final_mi,
        'vif_scores': final_vif,
        'status': 'success'
    }


def save_statistics_reports(global_stats: Dict, results: List[Dict],
                           output_dir: str):
    """
    Save statistics reports to files.
    
    Args:
        global_stats: Global statistics dictionary
        results: List of processing results for each feature set
        output_dir: Output directory
    """
    print("\n" + "="*60)
    print("Saving statistics reports...")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save MI scores
    mi_df = global_stats['mi_scores'].sort_values(ascending=False)
    mi_path = os.path.join(output_dir, 'feature_importance_mi_scores.csv')
    mi_df.to_csv(mi_path, header=['mutual_information'])
    print(f"Saved MI scores to: {mi_path}")
    
    # Save VIF scores
    vif_df = global_stats['vif_scores'].sort_values(ascending=False)
    vif_path = os.path.join(output_dir, 'vif_scores.csv')
    vif_df.to_csv(vif_path, header=['vif'])
    print(f"Saved VIF scores to: {vif_path}")
    
    # Save feature set summary
    summary_rows = []
    for result in results:
        if result['status'] == 'success':
            summary_rows.append({
                'feature_set': result['set_name'],
                'num_features': result['num_features'],
                'csv_file': os.path.basename(result['csv_path']),
                'prompt_file': os.path.basename(result['prompt_path']),
                'avg_mi': np.mean(list(result['mi_scores'].values())) if result['mi_scores'] else 0.0,
                'max_vif': np.max(list(result['vif_scores'].values())) if result['vif_scores'] else 0.0,
                'features': ', '.join(result['selected_features'])
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, 'feature_sets_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved feature sets summary to: {summary_path}")
    
    # Save correlation matrix (top features only)
    corr_matrix = global_stats['corr_matrix']
    top_features = mi_df.head(50).index.tolist()
    corr_subset = corr_matrix.loc[top_features, top_features]
    corr_path = os.path.join(output_dir, 'correlation_matrix_top50.csv')
    corr_subset.to_csv(corr_path)
    print(f"Saved correlation matrix to: {corr_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate feature-engineered datasets for TimeLLM')
    parser.add_argument('--root_path', type=str, default='./dataset',
                       help='Root directory for data files')
    parser.add_argument('--data_path', type=str, default='cryptex/daily/candlesticks-D.csv',
                       help='Path to OHLCV CSV file (relative to root_path)')
    parser.add_argument('--transactions_path', type=str, default='onchain/daily_transactions.csv',
                       help='Path to transactions CSV file (relative to root_path)')
    parser.add_argument('--addresses_path', type=str, default='onchain/unique_addresses.csv',
                       help='Path to addresses CSV file (relative to root_path)')
    parser.add_argument('--target', type=str, default='close',
                       help='Target column name')
    parser.add_argument('--output_dir', type=str, default='./dataset/cryptex/daily',
                       help='Output directory for feature CSV files')
    parser.add_argument('--frequency', type=str, default='D',
                       help='Frequency suffix for filenames (D, h, W, etc.)')
    parser.add_argument('--feature_sets', type=str, nargs='+', default=None,
                       help='Feature sets to process (default: all)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("TimeLLM Feature Engineering Preprocessing")
    print("="*60)
    
    # Load data
    ohlcv_df, transactions_df, addresses_df = load_data(
        args.root_path,
        args.data_path,
        args.transactions_path,
        args.addresses_path
    )
    
    # Engineer all candidate features
    df_features = engineer_candidate_features(
        ohlcv_df,
        transactions_df,
        addresses_df,
        args.target
    )
    
    # Get all feature columns (excluding timestamp and target)
    feature_cols = [col for col in df_features.columns 
                   if col not in ['timestamp', args.target]]
    
    # Compute global statistics
    global_stats = compute_global_statistics(
        df_features,
        args.target,
        feature_cols
    )
    
    # Process each feature set
    feature_sets = args.feature_sets or get_all_feature_set_names()
    results = []
    
    for set_name in feature_sets:
        try:
            result = process_feature_set(
                set_name,
                df_features,
                args.target,
                global_stats,
                args.output_dir,
                os.path.join(args.root_path, 'prompt_bank'),
                args.frequency
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing {set_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({'set_name': set_name, 'status': 'error', 'error': str(e)})
    
    # Save reports
    save_statistics_reports(global_stats, results, args.output_dir)
    
    # Final validation report
    validation_passed = report_validation_results()
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)
    successful_count = len([r for r in results if r.get('status') == 'success'])
    failed_count = len([r for r in results if r.get('status') == 'failed'])
    error_count = len([r for r in results if r.get('status') == 'error'])
    print(f"Processed {successful_count} feature sets successfully")
    if failed_count > 0:
        print(f"  {failed_count} feature sets failed")
    if error_count > 0:
        print(f"  {error_count} feature sets had errors")
    
    # Debug: Print status of each result
    print(f"\nDetailed results:")
    for r in results:
        status = r.get('status', 'unknown')
        name = r.get('set_name', 'unknown')
        print(f"  {name}: {status}")
    
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Prompt directory: {os.path.join(args.root_path, 'prompt_bank')}")
    
    if not validation_passed:
        print("\nWARNING: Some validation errors were found. Please review the validation summary above.")
        sys.exit(1)


if __name__ == '__main__':
    main()



