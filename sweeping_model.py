#!/usr/bin/env python3
import subprocess
from datetime import datetime, timedelta

# Sliding window settings
window_start = datetime(2020, 1, 1)
stop_date = datetime(2024, 1, 1)

step = timedelta(weeks=1)
window_size = timedelta(weeks=13)

while window_start < stop_date:
    start = window_start
    end = start + window_size
    inf_start = end
    inf_end = end + window_size

    cmd = [
        "python3", "run_hpo.py",
        "--gpu", "4",
        "--study_name", "hour_quarterly",
        "--granularity", "hourly",
        "--start", start.strftime("%Y-%m-%d"),
        "--end", end.strftime("%Y-%m-%d"),
        "--inf_start", inf_start.strftime("%Y-%m-%d"),
        "--inf_end", inf_end.strftime("%Y-%m-%d"),
        "--returns",
        "--backtest",
        "--experiment_name", "hour_quarterly",
        "--trials", "1",
        "--aggregate", "1",
        "--yaml_file", "hour_quarterly.yaml",
        "--log_all_metrics",
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    window_start += step
