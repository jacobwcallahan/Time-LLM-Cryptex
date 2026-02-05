import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import backtrader as bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strategies import (
    SimpleAIStrategy,
    SLTPStrategy,
    MomentumAIStrategy,
    RSIAIStrategy,
    BollingerAIStrategy,
    MeanReversionAIStrategy,
    TrendFollowingAIStrategy,
)


AI_STRATEGIES: Dict[str, Tuple[type, Dict]] = {
    "SimpleAI": (SimpleAIStrategy, {"confidence_threshold": 0.01}),
    "SLTP": (
        SLTPStrategy,
        {
            "confidence_threshold": 0.1,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.07,
        },
    ),
    "MomentumAI": (
        MomentumAIStrategy,
        {
            "confidence_threshold": 0.01,
            "momentum_window": 20,
        },
    ),
    "RSIAI": (
        RSIAIStrategy,
        {
            "confidence_threshold": 0.015,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
        },
    ),
    "BollingerAI": (
        BollingerAIStrategy,
        {
            "confidence_threshold": 0.01,
            "bb_period": 20,
            "bb_std": 2.0,
        },
    ),
    "MeanReversionAI": (
        MeanReversionAIStrategy,
        {
            "confidence_threshold": 0.015,
            "lookback_period": 20,
            "mean_reversion_threshold": 1.5,
        },
    ),
    "TrendFollowingAI": (
        TrendFollowingAIStrategy,
        {
            "confidence_threshold": 0.01,
            "ema_short": 5,
            "ema_long": 20,
        },
    ),
}


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


def _is_hourly_index(index: pd.DatetimeIndex) -> bool:
    if len(index) < 3:
        return False
    idx = index.sort_values()
    deltas = idx.to_series().diff().dropna().dt.total_seconds()
    if deltas.empty:
        return False
    return float(deltas.median()) <= 7200


def _read_train_cutoff_timestamp(train_csv: Path) -> Optional[pd.Timestamp]:
    if not train_csv.exists():
        return None
    train_df = pd.read_csv(train_csv, usecols=["timestamp"])
    train_ts = _parse_timestamp_series(train_df["timestamp"]).dropna()
    if train_ts.empty:
        return None
    return pd.Timestamp(train_ts.max())


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


def _display_label(stem: str) -> str:
    if stem == "1week_100":
        return "Hourly model"
    if stem.startswith("iar_"):
        return "HPO Daily"
    if stem.startswith("ohlcv_2weeks_to_1day"):
        return "Daily base model"

    prefix = stem.split("_", 1)[0].lower()
    mapping = {
        "momentum": "Momentum",
        "volatility": "Volatility",
        "onchain": "On-Chain",
        "onchainprice": "On-Chain",
        "onchain_price": "On-Chain",
        "volume": "Volume",
        "volumeprice": "Volume-Price",
        "volume_price": "Volume-Price",
        "technical": "Technical",
        "hybrid": "Hybrid",
        "returns": "Returns",
        "temporal": "Temporal",
        "minimal": "Minimal",
    }
    return mapping.get(prefix, prefix.title())


def _select_inference_files(repo_root: Path, input_dir: Path) -> List[Path]:
    feature_summary = repo_root / "dataset" / "cryptex" / "daily" / "feature_sets_summary.csv"
    feature_sets: List[str] = []
    if feature_summary.exists():
        df = pd.read_csv(feature_summary)
        if "feature_set" in df.columns:
            feature_sets = [str(x) for x in df["feature_set"].dropna().tolist()]

    desired: List[Path] = [input_dir / f"{fs}_1week_to_1day.csv" for fs in feature_sets]
    desired.append(input_dir / "1week_100.csv")
    desired.append(input_dir / "iar_LLAMA3.1_L12_daily_S_seq168_pred7_p6_s4_v1000.csv")
    desired.append(repo_root / "dataset" / "cryptex" / "daily" / "ohlcv_2weeks_to_1day_100pct.csv")

    return [p for p in desired if p.exists()]


def _select_prediction_horizon(stem: str) -> int:
    if stem == "1week_100":
        return 24
    return 1


def _run_strategy_once(
    df: pd.DataFrame,
    *,
    cash: float,
    commission: float,
    prediction_horizon: int,
    strategy_cls: type,
    params: Dict,
) -> Tuple[Optional[float], float, pd.Series]:
    pred_col = f"close_predicted_{prediction_horizon}"
    if pred_col not in df.columns:
        raise ValueError(f"Missing prediction column '{pred_col}'")

    data_feed_class = _make_data_feed_class(pred_col)

    cerebro = bt.Cerebro()
    run_params = dict(params)
    run_params["prediction_horizon"] = int(prediction_horizon)
    cerebro.addstrategy(strategy_cls, **run_params)

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
    sharpe_val: Optional[float]
    if sharpe is None:
        sharpe_val = None
    else:
        try:
            sharpe_val = float(sharpe)
        except (TypeError, ValueError):
            sharpe_val = None
        if sharpe_val is not None and np.isnan(sharpe_val):
            sharpe_val = None

    final_value = float(strat.broker.getvalue())
    total_return = (final_value - float(cash)) / float(cash) if cash != 0 else 0.0

    series = strat.analyzers.equity.get_analysis()
    return sharpe_val, total_return, series


def _pick_best_strategy(
    df: pd.DataFrame,
    *,
    cash: float,
    commission: float,
    prediction_horizon: int,
    metric: str,
) -> Tuple[str, float, float, pd.Series]:
    best_name = ""
    best_sharpe: Optional[float] = None
    best_total_return: Optional[float] = None
    best_series: pd.Series = pd.Series(dtype=float)

    for name, (strategy_cls, params) in AI_STRATEGIES.items():
        try:
            sharpe_val, total_return, series = _run_strategy_once(
                df,
                cash=cash,
                commission=commission,
                prediction_horizon=prediction_horizon,
                strategy_cls=strategy_cls,
                params=params,
            )
        except Exception:
            continue

        is_better = False
        if best_name == "":
            is_better = True
        elif metric == "total_return":
            if best_total_return is None or total_return > best_total_return:
                is_better = True
        else:
            if sharpe_val is not None and (best_sharpe is None or sharpe_val > best_sharpe):
                is_better = True
            elif sharpe_val is None and best_sharpe is None:
                if best_total_return is None or total_return > best_total_return:
                    is_better = True

        if is_better:
            best_name = name
            best_sharpe = sharpe_val
            best_total_return = total_return
            best_series = series

    if best_name == "" or best_series.empty:
        raise RuntimeError("No strategy produced a valid result")

    sharpe_out = float(best_sharpe) if best_sharpe is not None else float("nan")
    return_out = float(best_total_return) if best_total_return is not None else float("nan")
    return best_name, sharpe_out, return_out, best_series


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


def _plot_best(
    curves: List[Tuple[str, str, float, float, pd.Series]],
    *,
    output_path: Path,
    title: str,
    show_return: bool,
) -> None:
    overlap_start, overlap_end = _overlap_window([s for *_, s in curves])

    plt.figure(figsize=(12, 6))
    for stem, strat_name, sharpe, total_return, series in curves:
        clipped = series.loc[(series.index >= overlap_start) & (series.index <= overlap_end)]
        if clipped.empty:
            continue
        sharpe_str = f"{sharpe:.3f}" if not np.isnan(sharpe) else "nan"
        label = f"{_display_label(stem)} | {strat_name} | Sharpe={sharpe_str}"
        if show_return:
            if not np.isnan(total_return):
                label = f"{label} | Return={total_return * 100:.1f}%"
        plt.plot(clipped.index, clipped.values, linewidth=1.6, label=label)

    plt.title(f"{title} | From {overlap_start.date()} to {overlap_end.date()}")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate best-by-Sharpe equity plot and also plot best-by-total-return equity curves."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "inferences"),
        help="Directory containing inference CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("histogram_reports")),
        help="Directory to write PNG outputs.",
    )
    parser.add_argument("--cash", type=float, default=100000.0)
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument(
        "--train-data-hourly",
        type=str,
        default=str(Path("dataset/cryptex/hourly/candlesticks-h-clean.csv")),
    )
    parser.add_argument(
        "--train-data-daily",
        type=str,
        default=str(Path("dataset/cryptex/daily/candlesticks-D.csv")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_hourly = (
        (repo_root / args.train_data_hourly).resolve()
        if not Path(args.train_data_hourly).is_absolute()
        else Path(args.train_data_hourly).resolve()
    )
    train_daily = (
        (repo_root / args.train_data_daily).resolve()
        if not Path(args.train_data_daily).is_absolute()
        else Path(args.train_data_daily).resolve()
    )

    hourly_cutoff = _read_train_cutoff_timestamp(train_hourly)
    daily_cutoff = _read_train_cutoff_timestamp(train_daily)

    files = _select_inference_files(repo_root, input_dir)
    if not files:
        raise ValueError(f"No matching inference files found in {input_dir}")

    best_by_sharpe: List[Tuple[str, str, float, float, pd.Series]] = []
    best_by_return: List[Tuple[str, str, float, float, pd.Series]] = []

    for csv_path in files:
        stem = csv_path.stem

        df_raw = pd.read_csv(csv_path, usecols=["timestamp"])
        idx = _parse_timestamp_series(df_raw["timestamp"]).dropna()
        is_hourly = _is_hourly_index(pd.DatetimeIndex(idx))
        cutoff = hourly_cutoff if is_hourly else daily_cutoff

        df = _load_inference_csv(csv_path, train_cutoff=cutoff)
        if df.empty:
            print(f"[Skip] {csv_path.name}: no rows after cutoff")
            continue

        horizon = _select_prediction_horizon(stem)

        try:
            strat_name, sharpe, total_return, series = _pick_best_strategy(
                df,
                cash=args.cash,
                commission=args.commission,
                prediction_horizon=horizon,
                metric="sharpe",
            )
            best_by_sharpe.append((stem, strat_name, sharpe, total_return, series))
        except Exception as exc:
            print(f"[Skip] {csv_path.name} (best Sharpe): {exc}")

        try:
            strat_name, sharpe, total_return, series = _pick_best_strategy(
                df,
                cash=args.cash,
                commission=args.commission,
                prediction_horizon=horizon,
                metric="total_return",
            )
            best_by_return.append((stem, strat_name, sharpe, total_return, series))
        except Exception as exc:
            print(f"[Skip] {csv_path.name} (best Return): {exc}")

    if not best_by_sharpe:
        raise ValueError("No curves produced for best-by-Sharpe")
    if not best_by_return:
        raise ValueError("No curves produced for best-by-Return")

    out_best_sharpe = output_dir / "equity_curves_best_non_buyhold.png"
    _plot_best(
        best_by_sharpe,
        output_path=out_best_sharpe,
        title="Equity Curves (Best Strategy by Sharpe)",
        show_return=False,
    )

    out_best_return = output_dir / "equity_curves_best_total_return.png"
    _plot_best(
        best_by_return,
        output_path=out_best_return,
        title="Equity Curves (Best Strategy by Total Return)",
        show_return=True,
    )

    print(f"Saved: {out_best_sharpe}")
    print(f"Saved: {out_best_return}")


if __name__ == "__main__":
    main()

