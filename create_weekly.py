#!/usr/bin/env python3
"""
Script to aggregate daily cryptocurrency data into weekly data using the aggregate_data function.
"""

import os
import pandas as pd
import sys
from pathlib import Path

def aggregate_data(data, multiple):
    """
    Aggregate data to a given granularity.
    For example, if multiple=4, aggregate every 4 rows into 1 row.
    """
    # Group data by chunks of 'multiple' size
    grouped = data.groupby(data.index // multiple)
    
    # Aggregate each group
    new_data = pd.DataFrame()
    if "returns" in data.columns:
        raise ValueError("Cannot aggregate data with returns. Convert to returns after aggregation.")
    new_data['timestamp'] = grouped['timestamp'].first()
    new_data['open'] = grouped['open'].first()
    new_data['high'] = grouped['high'].max()
    new_data['low'] = grouped['low'].min()
    new_data['close'] = grouped['close'].last()
    new_data['volume'] = grouped['volume'].sum()
    
    return new_data

def create_weekly_data(input_file, output_file=None, multiple=7):
    """
    Convert daily data to weekly data using the aggregate_data function.
    
    Args:
        input_file (str): Path to the input daily data CSV file
        output_file (str): Path to save the weekly data (optional)
        multiple (int): Number of days to aggregate (default: 7 for weekly)
    
    Returns:
        pd.DataFrame: Aggregated weekly data
    """
    # Read the daily data
    print(f"Reading daily data from: {input_file}")
    daily_data = pd.read_csv(input_file)
    
    print(f"Original data shape: {daily_data.shape}")
    print(f"Date range: {daily_data['timestamp'].min()} to {daily_data['timestamp'].max()}")
    
    # Aggregate to weekly data (every 7 days)
    print(f"Aggregating data with multiple={multiple}...")
    weekly_data = aggregate_data(daily_data, multiple)
    
    print(f"Weekly data shape: {weekly_data.shape}")
    print(f"Weekly date range: {weekly_data['timestamp'].min()} to {weekly_data['timestamp'].max()}")
    
    # Generate output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_weekly.csv"
    
    # Save the weekly data
    print(f"Saving weekly data to: {output_file}")
    weekly_data.to_csv(output_file, index=False)
    
    return weekly_data

def main():
    """Main function to process available daily data files."""
    
    # Define the dataset directory
    dataset_dir = Path("dataset/cryptex")
    
    # List of daily data files to process
    daily_files = [
        "candlesticks-D.csv",
        "btc_cycle5_bull_1D_yfinance.csv",
        "btc_cycle5_bull_1D_yfinance_reordered.csv",
        "inference_test_btc_D_2024_2025.csv"
    ]
    
    print("=== Weekly Data Aggregation ===")
    print(f"Processing files from: {dataset_dir}")
    print()
    
    for file_name in daily_files:
        input_path = dataset_dir / file_name
        
        if input_path.exists():
            print(f"Processing: {file_name}")
            try:
                # Create weekly data
                weekly_data = create_weekly_data(str(input_path))
                
                # Display sample of the weekly data
                print("Sample of weekly data:")
                print(weekly_data.head())
                print()
                
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                print()
        else:
            print(f"File not found: {input_path}")
            print()

if __name__ == "__main__":
    main()
