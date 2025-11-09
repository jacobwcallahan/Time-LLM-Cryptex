import pandas as pd
import numpy as np
from pathlib import Path
from utils.pipeline import inf_analysis, convert_back_to_candlesticks
from datetime import datetime

folder = Path("~/Downloads/all_inference")
csv_files = sorted(folder.glob("*.csv"))

dfs = []
for i in range(1,60):
    if i == 1:
        dfs.append(pd.read_csv(f'~/Downloads/all_inference/inference.csv'))
    else:
        dfs.append(pd.read_csv(f'~/Downloads/all_inference/inference({i}).csv'))

# Stack all files with multi-index
merged = pd.concat(dfs, axis=0, join="outer", keys=range(len(dfs)))

# Compute numeric averages per row position
avg_numeric = merged.groupby(level=1).mean(numeric_only=True)

# Grab timestamp (or any non-numeric column) from the first file
timestamp = dfs[0]["timestamp"].reset_index(drop=True)

# Combine them back together
avg = pd.concat([timestamp, avg_numeric], axis=1)

avg.to_csv("inference_average.csv", index=False)

org_data = pd.read_csv("dataset/cryptex/candlesticks-Min.csv")
org_data = org_data[org_data['timestamp'] >= datetime.strptime(avg['timestamp'].iloc[0], '%Y-%m-%d %H:%M:%S').timestamp()]
org_data = org_data[org_data['timestamp'] <= datetime.strptime(avg['timestamp'].iloc[-1], '%Y-%m-%d %H:%M:%S').timestamp()]
org_data.to_csv("org_data.csv", index=False)

candlesticks_avg = convert_back_to_candlesticks("org_data.csv", "inference_average.csv", 24)
mda_vals = inf_analysis("inference_average.csv", target = "close")
for metric, value in mda_vals.items():
    print(f"{metric}: {value}")

errors = {f'pred_{pred}': [] for pred in range(1, 25)}
pred_len = 24
for i in range(len(avg) - pred_len):
    row = avg.iloc[i]
    for pred in range(1, pred_len+1):
        next_row = avg.iloc[i+pred]
        if pd.notna(row[f'returns_predicted_{pred}']):
            error = row[f'returns_predicted_{pred}'] - next_row['returns']
            sq_error = error ** 2
            errors[f'pred_{pred}'].append(sq_error)

for pred in range(1, 25):
    print(f"MSE for pred_{pred}: {round(np.mean(errors[f'pred_{pred}']) * 100, 6)}%")
    print(f"RMSE for pred_{pred}: {round(np.sqrt(np.mean(errors[f'pred_{pred}'])) * 100, 4)}%")






