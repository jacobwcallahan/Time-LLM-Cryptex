import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def last_value_baseline_mse(series: Iterable[float], seq_len: int, pred_len: int) -> Optional[float]:
    """
    Compute the naive baseline MSE where each future value is predicted as the last observed value.
    """
    values = np.asarray(series, dtype=float)
    total_mse = 0.0
    n = 0

    for i in range(seq_len, len(values) - pred_len):
        last_val = values[i - 1]
        true_vals = values[i:i + pred_len]
        preds = np.full(pred_len, last_val)
        mse = np.mean((preds - true_vals) ** 2)
        total_mse += mse
        n += 1

    return total_mse / n if n > 0 else None


def evaluate_inference_csv(csv_path: Path, column: str, seq_len: int, pred_len: int) -> Optional[float]:
    """Load a CSV and compute the naive baseline MSE on the specified column."""
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {csv_path}")

    series = df[column].dropna().to_numpy()
    return last_value_baseline_mse(series, seq_len, pred_len)


def generate_naive_predictions(input_csv: Path, output_csv: Path, seq_len: int, pred_len: int) -> None:
    """
    Generate a naive inference CSV where each prediction column replicates the last observed close.
    The structure mirrors run_inference.py output so it can be fed directly into backtesting.
    """
    df_raw = pd.read_csv(input_csv)
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])

    results = []
    # Create prediction rows based on rolling windows
    for i in range(seq_len, len(df_raw)):
        input_window = df_raw.iloc[i - seq_len:i]
        last_row = input_window.iloc[-1].copy()
        last_close = last_row['close']
        for j in range(pred_len):
            last_row[f'close_predicted_{j+1}'] = last_close
        results.append(last_row)

    # Prepend the initial seq_len rows with NaN predictions to match inference format
    for i in range(seq_len):
        row = df_raw.iloc[i].copy()
        for j in range(pred_len):
            row[f'close_predicted_{j+1}'] = np.nan
        results.insert(i, row)

    result_df = pd.DataFrame(results)
    result_df.sort_values('timestamp', inplace=True)
    result_df['timestamp'] = result_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    result_df.to_csv(output_csv, index=False)
    print(f"Naive inference CSV saved to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Naive baseline utilities.")
    parser.add_argument("--csv", required=True, help="Path to the input CSV (raw data or inference).")
    parser.add_argument("--column", default="close", help="Column to evaluate for baseline MSE (default: close).")
    parser.add_argument("--seq-len", type=int, default=168, help="Input sequence length (default: 168).")
    parser.add_argument("--pred-len", type=int, default=24, help="Prediction length (default: 24).")
    parser.add_argument("--output", help="Optional path to save naive inference CSV.")
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    mse = evaluate_inference_csv(csv_path, args.column, args.seq_len, args.pred_len)
    if mse is None:
        print("Not enough data to compute baseline MSE.")
    else:
        print(f"Baseline MSE (last-value) for {csv_path.name}: {mse:.6f}")

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        generate_naive_predictions(csv_path, output_path, args.seq_len, args.pred_len)
        print("You can now run backtesting on the generated naive predictions:")
        print(f"  python backtesting/backtest.py --data {output_path}")


if __name__ == "__main__":
    main()

