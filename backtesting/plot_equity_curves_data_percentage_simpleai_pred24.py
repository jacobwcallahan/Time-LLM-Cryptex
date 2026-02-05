#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import backtrader as bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strategies import SimpleAIStrategy


INFERENCES = [
    ("25", Path("backtesting/inferences/1week_25.csv")),
    ("50", Path("backtesting/inferences/1week_50.csv")),
    ("75", Path("backtesting/inferences/1week_75.csv")),
    ("100", Path("backtesting/inferences/1week_100.csv")),
]

TRAIN_HOURLY_DATA_PATH = Path("dataset/cryptex/hourly/candlesticks-h-clean.csv")
OUT_PATH = Path("data_percentage/equity_curves_simpleai_pred24_data_percentage.png")

PREDICTION_HORIZON_HOURS = 24
CONFIDENCE_THRESHOLD = 0.01
CASH = 100000.0
COMMISSION = 0.001


class EquityCurve(bt.Analyzer):
    def start(self):
        self._dts: List[pd.Timestamp] = []
        self._values: List[float] = []

    def next(self):
        dt = self.strategy.datas[0].datetime.datetime(0)
        self._dts.append(pd.Timestamp(dt))
        self._values.append(float(self.strategy.broker.getvalue()))

    def get_analysis(self):
        if not self._dts:
            return pd.Series(dtype=float)
        return pd.Series(self._values, index=pd.to_datetime(self._dts)).sort_index()


def _parse_timestamp_series(ts: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(ts):
        vals = pd.to_numeric(ts, errors="coerce")
        maxv = vals.dropna().max()
        unit = "s"
        if pd.notna(maxv):
            if maxv > 1e14:
                unit = "ns"
            elif maxv > 1e11:
                unit = "ms"
        return pd.to_datetime(vals, unit=unit, errors="coerce")
    return pd.to_datetime(ts, errors="coerce")


def _read_train_cutoff_timestamp(train_csv: Path) -> Optional[pd.Timestamp]:
    if not train_csv.exists():
        return None
    df = pd.read_csv(train_csv, usecols=["timestamp"])
    ts = _parse_timestamp_series(df["timestamp"]).dropna()
    if ts.empty:
        return None
    return pd.Timestamp(ts.max())


def _load_inference_csv(csv_path: Path, *, train_cutoff: Optional[pd.Timestamp]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{csv_path.name} missing 'timestamp' column")

    df = df.copy()
    df["timestamp"] = _parse_timestamp_series(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {missing}")

    if train_cutoff is not None:
        df = df[df.index > train_cutoff].copy()

    return df


def _make_data_feed_class(pred_col: str):
    class CustomPandasData(bt.feeds.PandasData):
        lines = (pred_col,)
        params = ((pred_col, pred_col),)

    return CustomPandasData


def _run_simpleai_equity_curve(
    df: pd.DataFrame,
    *,
    cash: float,
    commission: float,
    prediction_horizon: int,
    confidence_threshold: float,
) -> Tuple[float, pd.Series]:
    pred_col = f"close_predicted_{prediction_horizon}"
    if pred_col not in df.columns:
        raise ValueError(f"Missing prediction column '{pred_col}'")

    data_feed_class = _make_data_feed_class(pred_col)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        SimpleAIStrategy,
        prediction_horizon=int(prediction_horizon),
        confidence_threshold=float(confidence_threshold),
    )

    data_feed = data_feed_class(dataname=df)
    cerebro.adddata(data_feed)

    cerebro.broker.setcash(float(cash))
    cerebro.broker.setcommission(commission=float(commission))
    cerebro.broker.set_coc(True)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(EquityCurve, _name="equity")

    results = cerebro.run()
    strat = results[0]

    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    try:
        sharpe_val = float(sharpe)
    except (TypeError, ValueError):
        sharpe_val = float("nan")

    series = strat.analyzers.equity.get_analysis()
    return sharpe_val, series


def _overlap_window(series_list: List[pd.Series]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    starts = [s.index.min() for s in series_list if not s.empty]
    ends = [s.index.max() for s in series_list if not s.empty]
    if not starts or not ends:
        raise ValueError("No non-empty equity curves")
    start = max(starts)
    end = min(ends)
    if start >= end:
        raise ValueError(f"No overlapping time window: start={start}, end={end}")
    return start, end


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    cutoff = _read_train_cutoff_timestamp(TRAIN_HOURLY_DATA_PATH)
    if cutoff is not None:
        print(f"Train cutoff: {cutoff}")

    curves: List[Tuple[str, float, pd.Series]] = []
    for pct, csv_path in INFERENCES:
        df = _load_inference_csv(csv_path, train_cutoff=cutoff)
        if df.empty:
            print(f"[Skip] {csv_path.name}: no rows after cutoff")
            continue
        sharpe, series = _run_simpleai_equity_curve(
            df,
            cash=CASH,
            commission=COMMISSION,
            prediction_horizon=PREDICTION_HORIZON_HOURS,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )
        curves.append((pct, sharpe, series))

    if len(curves) != 4:
        raise ValueError(f"Expected 4 curves, got {len(curves)}")

    overlap_start, overlap_end = _overlap_window([s for _, _, s in curves])

    plt.figure(figsize=(12, 6))
    for pct, sharpe, series in curves:
        clipped = series.loc[(series.index >= overlap_start) & (series.index <= overlap_end)]
        if clipped.empty:
            continue
        sharpe_txt = f"{sharpe:.3f}" if not np.isnan(sharpe) else "nan"
        plt.plot(clipped.index, clipped.values, linewidth=1.6, label=f"{pct}% | Sharpe={sharpe_txt}")

    plt.title(
        f"Equity Curves (SimpleAI, pred24) | From {overlap_start.date()} to {overlap_end.date()}"
    )
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200)
    plt.close()

    print(f"Saved: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()


