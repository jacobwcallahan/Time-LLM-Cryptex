#!/usr/bin/env python3
"""
Clean the hourly cryptocurrency data by removing rows with NaN values.
This fixes the NaN loss issue by ensuring clean data.
"""

import pandas as pd
import numpy as np

def clean_hourly_data():
    """Remove NaN rows from the hourly cryptocurrency data."""
    
    # Load the data
    df = pd.read_csv('dataset/cryptex/hourly/candlesticks-h.csv')
    
    print(f"Original data shape: {df.shape}")
    print(f"NaN count: {df.isnull().sum().sum()}")
    
    # Remove rows with any NaN values
    df_clean = df.dropna()
    
    print(f"Cleaned data shape: {df_clean.shape}")
    print(f"Removed {len(df) - len(df_clean)} rows with NaN values")
    
    # Save cleaned data
    df_clean.to_csv('dataset/cryptex/hourly/candlesticks-h-clean.csv', index=False)
    print("Saved cleaned data to: dataset/cryptex/hourly/candlesticks-h-clean.csv")
    
    # Verify no NaN values remain
    print(f"Verification - NaN count in cleaned data: {df_clean.isnull().sum().sum()}")
    
    return df_clean

if __name__ == "__main__":
    clean_hourly_data()





