import pandas as pd
from pathlib import Path

def clean_ohlcv_nans(data_path: Path):
    """
    Cleans the OHLCV data of any NaN values.
    
    Args:
        data_path: Path to the data

    Returns:
        new_data: Cleaned data
    """
    data = pd.read_csv(data_path)
    print(data.head())
    n = len(data)
    new_data = data.copy()
    last_row = new_data.iloc[0]
    for i, row in data.iterrows():

        if row['close'] is None or row['close'] == "nan" or pd.isna(row['close']) or row['close'] == "":
            new_data.at[i, 'open'] = last_row['close']

            if i < n-1:
                next = data.iloc[i+1]
                if next['close'] is not None and next['close'] != "nan" and not pd.isna(next['close']) and next['close'] != "":
                    new_data.at[i, 'close'] = next['open']
                else:
                    new_data.at[i, 'close'] = last_row['close']
            new_data.at[i, 'high'] = max(new_data.at[i, 'open'], new_data.at[i, 'close'])
            new_data.at[i, 'low'] = min(new_data.at[i, 'open'], new_data.at[i, 'close'])
            new_data.at[i, 'volume'] = 0

        last_row = new_data.iloc[i]

    print(data.head())
    new_data.to_csv(data_path, index=False)
    return new_data
