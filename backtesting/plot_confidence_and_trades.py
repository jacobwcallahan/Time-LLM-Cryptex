"""
Plot confidence intervals with price, prediction mean, and trade markers.

Loads inference CSV, confidence CSV (from confidence_from_overlapping_windows.py),
and optional trades CSV (or runs backtester to get trades). Produces a figure with
price, prediction mean, CI band, confidence metric, and buy/sell markers.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Run from repo root or backtesting dir; ensure backtesting is on path
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))
if str(_script_dir.parent) not in sys.path:
    sys.path.insert(0, str(_script_dir.parent))

from utils import load_and_prepare_data, _parse_timestamp_series


def _parse_ts(series: pd.Series) -> pd.Series:
    return _parse_timestamp_series(series)


def load_inference(inference_csv: Path, pred_col: str) -> pd.DataFrame:
    df = pd.read_csv(inference_csv)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise ValueError(f"{inference_csv.name}: need timestamp and close")
    df = df.copy()
    df["timestamp"] = _parse_ts(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    if pred_col not in df.columns:
        raise ValueError(f"{inference_csv.name}: missing {pred_col}")
    return df[["close", pred_col]].copy()


def load_confidence(confidence_csv: Path, pred_col: str) -> pd.DataFrame:
    df = pd.read_csv(confidence_csv)
    df = df.copy()
    df["timestamp"] = _parse_ts(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    mean_col = f"{pred_col}_mean"
    lower_col = f"{pred_col}_lower"
    upper_col = f"{pred_col}_upper"
    std_col = f"{pred_col}_std"
    for c in (mean_col, lower_col, upper_col, std_col):
        if c not in df.columns:
            raise ValueError(f"{confidence_csv.name}: missing {c}")
    return df[[mean_col, lower_col, upper_col, std_col]].copy()


def load_trades_csv(trades_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(trades_csv)
    if "timestamp" not in df.columns or "action" not in df.columns:
        raise ValueError(f"{trades_csv.name}: need timestamp and action")
    df = df.copy()
    df["timestamp"] = _parse_ts(df["timestamp"])
    df = df.dropna(subset=["timestamp"])
    return df[["timestamp", "action"]]


def run_backtester_get_trades(
    inference_csv: Path,
    strategy: str,
    prediction_horizon: int,
    train_data_path: Optional[str],
) -> pd.DataFrame:
    from backtest import BacktestRunner, STRATEGIES, OPTIMIZATION_RANGES
    for name, spec in STRATEGIES.items():
        if name != "BuyHold":
            spec.get("params", {})["prediction_horizon"] = prediction_horizon
    for name, ranges in OPTIMIZATION_RANGES.items():
        if "prediction_horizon" in ranges:
            ranges["prediction_horizon"] = [prediction_horizon]
    runner = BacktestRunner(
        str(inference_csv),
        cash=100000.0,
        commission=0.001,
        train_data_path=train_data_path,
    )
    runner.run_strategy(strategy)
    return runner.get_trade_dates(strategy)


def main():
    parser = argparse.ArgumentParser(
        description="Plot confidence intervals with price, prediction, and trades."
    )
    parser.add_argument(
        "--inference_csv",
        type=Path,
        required=True,
        help="Inference CSV (timestamp, close, close_predicted_<h>).",
    )
    parser.add_argument(
        "--confidence_csv",
        type=Path,
        required=True,
        help="Confidence CSV from confidence_from_overlapping_windows.py.",
    )
    parser.add_argument(
        "--trades_csv",
        type=Path,
        default=None,
        help="Optional CSV with timestamp, action (buy/sell). If omitted, run backtester.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="SimpleAI",
        help="Strategy name when running backtester to get trades (default: SimpleAI).",
    )
    parser.add_argument(
        "--prediction_horizon",
        type=int,
        default=1,
        help="Prediction horizon for pred column and backtester (default: 1).",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default=None,
        help="Training CSV for backtest cutoff when running backtester (optional).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output figure path (e.g. PNG).",
    )
    args = parser.parse_args()

    pred_col = f"close_predicted_{args.prediction_horizon}"

    inf = load_inference(args.inference_csv, pred_col)
    conf = load_confidence(args.confidence_csv, pred_col)
    merged = inf.join(conf, how="inner")
    if merged.empty:
        raise ValueError("Inference and confidence have no overlapping timestamps")

    if args.trades_csv is not None and args.trades_csv.exists():
        trades = load_trades_csv(args.trades_csv)
    else:
        trades = run_backtester_get_trades(
            args.inference_csv,
            args.strategy,
            args.prediction_horizon,
            args.train_data,
        )

    mean_col = f"{pred_col}_mean"
    lower_col = f"{pred_col}_lower"
    upper_col = f"{pred_col}_upper"
    std_col = f"{pred_col}_std"

    x = merged.index
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), height_ratios=[1.5, 0.6])

    ax1.plot(x, merged["close"], label="Close", color="black", alpha=0.8)
    ax1.plot(x, merged[mean_col], label="Prediction mean", color="blue", alpha=0.8)
    valid = merged[lower_col].notna() & merged[upper_col].notna()
    if valid.any():
        ax1.fill_between(
            x[valid],
            merged.loc[valid, lower_col],
            merged.loc[valid, upper_col],
            alpha=0.2,
            color="blue",
            label="CI",
        )
    if not trades.empty:
        trades_ts = pd.to_datetime(trades["timestamp"])
        in_index = trades_ts.isin(merged.index)
        trades_aligned = trades[in_index].copy()
        trades_aligned["timestamp"] = trades_ts[in_index]
        if not trades_aligned.empty:
            buys = trades_aligned["action"] == "buy"
            sells = trades_aligned["action"] == "sell"
            if buys.any():
                ax1.scatter(
                    trades_aligned.loc[buys, "timestamp"],
                    merged.loc[trades_aligned.loc[buys, "timestamp"], "close"].values,
                    marker="^",
                    color="green",
                    s=60,
                    zorder=5,
                    label="Buy",
                )
            if sells.any():
                ax1.scatter(
                    trades_aligned.loc[sells, "timestamp"],
                    merged.loc[trades_aligned.loc[sells, "timestamp"], "close"].values,
                    marker="v",
                    color="red",
                    s=60,
                    zorder=5,
                    label="Sell",
                )
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Price, prediction mean, CI and trades")

    ci_width = merged[upper_col] - merged[lower_col]
    ax2.plot(x, ci_width, label="CI width", color="gray", alpha=0.8)
    ax2.set_ylabel("CI width")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Confidence (narrow CI = higher confidence)")

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
