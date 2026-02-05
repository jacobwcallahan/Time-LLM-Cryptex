"""
Compute confidence intervals from overlapping-window inferences.

Use when you have N inference runs (e.g. 30), each from a different window end date.
For each calendar date, N windows predict the same target; spread (std) and
intervals are used as confidence. Output can be joined with backtest trades.

Join with backtester: a trade on bar date D uses the prediction for D+1. So join
confidence CSV on timestamp == D (window end) or target_date == D+1 to get the
CI for that prediction (close_predicted_1_mean, _lower, _upper, _std).

Usage:
  # Directory of N inference CSVs (one per window run), same timestamp coverage
  python backtesting/confidence_from_overlapping_windows.py \
    --input_dir path/to/30_inference_csvs \
    --output_csv path/to/confidence_intervals.csv \
    --horizon 1

  # Optional: single CSV with run_id column (e.g. window_end_idx 1..30)
  python backtesting/confidence_from_overlapping_windows.py \
    --input_csv path/to/combined.csv \
    --run_id_col window_id \
    --output_csv path/to/confidence_intervals.csv \
    --horizon 1
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def _parse_ts(series: pd.Series) -> pd.DatetimeIndex:
    return pd.to_datetime(series, errors="coerce")


def load_inference_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path.name}: missing 'timestamp'")
    df = df.copy()
    df["timestamp"] = _parse_ts(df["timestamp"])
    df = df.dropna(subset=["timestamp"])
    return df


def compute_ci_from_directory(
    input_dir: Path,
    pred_col: str,
    horizon: int,
    ci_mult: float = 1.96,
) -> pd.DataFrame:
    """
    Load all CSV files in input_dir; each file = one window run.
    Align on timestamp: for each timestamp T we have one row per file with
    close_predicted_{horizon} = prediction for T+horizon. Aggregate to mean, std, CI.
    """
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {input_dir}")

    frames = []
    for f in files:
        df = load_inference_csv(f)
        if pred_col not in df.columns:
            raise ValueError(f"{f.name}: missing '{pred_col}'")
        df = df[["timestamp", pred_col]].copy()
        df = df.rename(columns={pred_col: "pred"})
        df["_source"] = f.stem
        frames.append(df)

    # Pivot so each timestamp has one row, columns = pred from each file
    combined = pd.concat(frames, ignore_index=True)
    wide = combined.pivot_table(
        index="timestamp",
        columns="_source",
        values="pred",
        aggfunc="first",
    )

    # Drop rows with no predictions
    wide = wide.dropna(how="all")
    n = wide.count(axis=1)
    mean = wide.mean(axis=1)
    std = wide.std(axis=1)
    # Fill std when only one value so CI is still defined
    std = std.fillna(0.0)

    target_date = wide.index + pd.Timedelta(days=horizon)

    out = pd.DataFrame(
        {
            "timestamp": wide.index,
            "target_date": target_date,
            f"{pred_col}_mean": mean,
            f"{pred_col}_std": std,
            f"{pred_col}_lower": mean - ci_mult * std,
            f"{pred_col}_upper": mean + ci_mult * std,
            f"{pred_col}_count": n.astype(int),
        }
    )
    out = out.reset_index(drop=True)
    return out


def compute_ci_from_single_csv(
    input_csv: Path,
    run_id_col: str,
    pred_col: str,
    horizon: int,
    ci_mult: float = 1.96,
) -> pd.DataFrame:
    """
    Single CSV with a column run_id_col (e.g. window_id 1..30).
    Group by timestamp, aggregate pred_col across runs.
    """
    df = pd.read_csv(input_csv)
    for c in ["timestamp", run_id_col, pred_col]:
        if c not in df.columns:
            raise ValueError(f"CSV missing column '{c}'")
    df = df.copy()
    df["timestamp"] = _parse_ts(df["timestamp"])
    df = df.dropna(subset=["timestamp", pred_col])

    agg = df.groupby("timestamp")[pred_col].agg(["mean", "std", "count"])
    agg["std"] = agg["std"].fillna(0.0)
    agg["lower"] = agg["mean"] - ci_mult * agg["std"]
    agg["upper"] = agg["mean"] + ci_mult * agg["std"]
    agg = agg.reset_index()
    agg["target_date"] = agg["timestamp"] + pd.Timedelta(days=horizon)
    agg = agg.rename(columns={
        "mean": f"{pred_col}_mean",
        "std": f"{pred_col}_std",
        "lower": f"{pred_col}_lower",
        "upper": f"{pred_col}_upper",
        "count": f"{pred_col}_count",
    })
    return agg


def main():
    ap = argparse.ArgumentParser(
        description="Compute confidence intervals from overlapping inference windows."
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--input_dir",
        type=Path,
        help="Directory containing one inference CSV per window run.",
    )
    g.add_argument(
        "--input_csv",
        type=Path,
        help="Single CSV with a run/window ID column (use with --run_id_col).",
    )
    ap.add_argument(
        "--run_id_col",
        type=str,
        default="window_id",
        help="Column in --input_csv that identifies the window run (default: window_id).",
    )
    ap.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Output path for confidence-interval CSV.",
    )
    ap.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Prediction horizon (1 = next day). Used for target_date and pred column (default: 1).",
    )
    ap.add_argument(
        "--ci_mult",
        type=float,
        default=1.96,
        help="Multiplier for std to get CI (default 1.96 ~ 95%%).",
    )
    args = ap.parse_args()

    pred_col = f"close_predicted_{args.horizon}"

    if args.input_dir is not None:
        out = compute_ci_from_directory(
            args.input_dir,
            pred_col=pred_col,
            horizon=args.horizon,
            ci_mult=args.ci_mult,
        )
    else:
        out = compute_ci_from_single_csv(
            args.input_csv,
            run_id_col=args.run_id_col,
            pred_col=pred_col,
            horizon=args.horizon,
            ci_mult=args.ci_mult,
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(out)} rows to {args.output_csv}")
    print("Columns:", list(out.columns))
    print("Sample (first 3 rows):")
    print(out.head(3).to_string())


if __name__ == "__main__":
    main()
