#!/usr/bin/env python3
"""
Generate feature-engineered inference input CSVs for the BTC daily holdout window.

This script:
- Creates a post-train holdout OHLCV file by cutting inference_test_btc after the
  end of the training OHLCV history (candlesticks-D.csv).
- Computes engineered features over full history (train history + holdout) using
  utils.feature_engineer.engineer_all_features (causal features only).
- Writes per-feature-set inference CSVs that match the *training feature CSV schema*
  (same columns + order), so trained models can run inference on the holdout.

No future-looking fills are used (ffill only; no bfill).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from utils.feature_engineer import engineer_all_features


FEATURE_SETS: List[str] = [
    "momentum",
    "volatility",
    "onchain_price",
    "volume_price",
    "technical",
    "hybrid",
    "returns",
    "temporal",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _require_cols(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _load_training_headers(train_feature_dir: Path) -> Dict[str, List[str]]:
    headers: Dict[str, List[str]] = {}
    for set_name in FEATURE_SETS:
        p = train_feature_dir / f"candlesticks-D_features_{set_name}.csv"
        df0 = _read_csv(p)
        headers[set_name] = df0.columns.tolist()
    return headers


def _dedupe_by_timestamp_keep_last(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="raise").astype(np.int64)
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate inference feature datasets for BTC daily holdout.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="dataset",
        help="Dataset root directory (default: dataset).",
    )
    parser.add_argument(
        "--train_ohlcv_path",
        type=str,
        default="cryptex/daily/candlesticks-D.csv",
        help="Training/history OHLCV CSV relative to dataset_root.",
    )
    parser.add_argument(
        "--inference_ohlcv_path",
        type=str,
        default="cryptex/daily/inference_test_btc_D_2024_2025.csv",
        help="Original inference OHLCV CSV relative to dataset_root.",
    )
    parser.add_argument(
        "--transactions_path",
        type=str,
        default="onchain/daily_transactions.csv",
        help="On-chain transactions CSV relative to dataset_root.",
    )
    parser.add_argument(
        "--addresses_path",
        type=str,
        default="onchain/unique_addresses.csv",
        help="On-chain addresses CSV relative to dataset_root.",
    )
    parser.add_argument(
        "--train_feature_dir",
        type=str,
        default="cryptex/daily",
        help="Directory (relative to dataset_root) containing candlesticks-D_features_<set>.csv files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cryptex/daily",
        help="Output directory (relative to dataset_root) for generated inference feature CSVs.",
    )
    parser.add_argument(
        "--posttrain_name",
        type=str,
        default="inference_test_btc_D_posttrain.csv",
        help="Filename for the post-train OHLCV holdout (written under output_dir).",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    train_ohlcv_path = dataset_root / args.train_ohlcv_path
    inference_ohlcv_path = dataset_root / args.inference_ohlcv_path
    transactions_path = dataset_root / args.transactions_path
    addresses_path = dataset_root / args.addresses_path
    train_feature_dir = dataset_root / args.train_feature_dir
    output_dir = dataset_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = _read_csv(train_ohlcv_path)
    infer_df = _read_csv(inference_ohlcv_path)
    _require_cols(train_df, ["timestamp", "open", "high", "low", "close", "volume"], "train OHLCV")
    _require_cols(infer_df, ["timestamp", "open", "high", "low", "close", "volume"], "inference OHLCV")

    train_df = _dedupe_by_timestamp_keep_last(train_df)
    infer_df = _dedupe_by_timestamp_keep_last(infer_df)

    train_end_ts = int(train_df["timestamp"].max())
    posttrain_df = infer_df[infer_df["timestamp"] > train_end_ts].copy()
    if posttrain_df.empty:
        raise ValueError(
            "Post-train holdout is empty. inference_ohlcv_path does not extend beyond training end."
        )

    posttrain_out = output_dir / args.posttrain_name
    posttrain_df.to_csv(posttrain_out, index=False)
    print(f"Saved post-train OHLCV holdout: {posttrain_out} ({len(posttrain_df)} rows)")

    # Merge full history for feature computation (prefer inference OHLCV on overlaps).
    merged_ohlcv = pd.concat([train_df, infer_df], axis=0, ignore_index=True)
    merged_ohlcv = _dedupe_by_timestamp_keep_last(merged_ohlcv)

    # Load on-chain data.
    transactions_df = _read_csv(transactions_path)
    addresses_df = _read_csv(addresses_path)

    # Compute full feature pool over history.
    merged_features = engineer_all_features(
        merged_ohlcv,
        transactions_df=transactions_df,
        addresses_df=addresses_df,
        timestamp_col="timestamp",
        price_col="close",
    )
    merged_features = merged_features.sort_values("timestamp")

    # Causal fill only. Do not bfill.
    merged_features = merged_features.ffill()

    # Slice to post-train timestamps.
    posttrain_ts = set(posttrain_df["timestamp"].astype(np.int64).tolist())
    holdout_features = merged_features[merged_features["timestamp"].astype(np.int64).isin(posttrain_ts)].copy()
    holdout_features = holdout_features.sort_values("timestamp")

    # Load training headers so we match schema exactly.
    headers = _load_training_headers(train_feature_dir)

    # Write one inference feature CSV per feature set.
    for set_name in FEATURE_SETS:
        header = headers[set_name]
        missing_cols = [c for c in header if c not in holdout_features.columns]
        if missing_cols:
            raise ValueError(
                f"Holdout feature pool missing columns for {set_name}: {missing_cols}"
            )

        out_df = holdout_features[header].copy()

        # Final cleanup: drop any remaining NaNs/infs.
        out_df = out_df.replace([np.inf, -np.inf], np.nan)
        before = len(out_df)
        out_df = out_df.dropna(axis=0, how="any")
        dropped = before - len(out_df)
        if dropped:
            print(f"{set_name}: dropped {dropped} rows with NaN/Inf after ffill")

        # Enforce timestamp numeric format as in inference input (unix seconds int).
        out_df["timestamp"] = pd.to_numeric(out_df["timestamp"], errors="raise").astype(np.int64)

        out_path = output_dir / f"inference_test_btc_D_posttrain_features_{set_name}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path} ({len(out_df)} rows, {len(out_df.columns)} cols)")


if __name__ == "__main__":
    main()



