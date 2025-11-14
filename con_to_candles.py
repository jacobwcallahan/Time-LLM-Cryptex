from pathlib import Path
import pandas as pd
from datetime import datetime
from utils.pipeline import convert_back_to_candlesticks, get_mse_vals

inf_path = Path("15_min_inf.csv")
org_path = Path("dataset/cryptex/candlesticks-Min.csv")

inf_data = pd.read_csv(inf_path)
org_data = pd.read_csv(org_path)
print(datetime.strptime(inf_data['timestamp'].iloc[0], '%Y-%m-%d %H:%M:%S').timestamp())

org_data = org_data[org_data['timestamp'] >= datetime.strptime(inf_data['timestamp'].iloc[0], '%Y-%m-%d %H:%M:%S').timestamp()]
org_data = org_data[org_data['timestamp'] <= datetime.strptime(inf_data['timestamp'].iloc[-1], '%Y-%m-%d %H:%M:%S').timestamp()]
org_data.to_csv("org_data.csv", index=False)

print(inf_data.head())
print(inf_data.columns)
print(org_data.head())
print(org_data.columns)

print(datetime.fromtimestamp(org_data.iloc[0]['timestamp']))

candlesticks = convert_back_to_candlesticks("org_data.csv", "15_min_inf.csv", 24)
candlesticks.to_csv("candlesticks.csv", index=False)


