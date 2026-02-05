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


def _read_train_cutoff_timestamp(train_csv: Path) -> Optional[pd.Timestamp]:
    if not train_csv.exists():
        return None
    df = pd.read_csv(train_csv, usecols=["timestamp"])
    ts = _parse_timestamp_series(df["timestamp"]).dropna()
    if ts.empty:
        return None
    return pd.Timestamp(ts.max())


def _make_data_feed_class(pred_col: str):
    class CustomPandasData(bt.feeds.PandasData):
        lines = (pred_col,)
        params = ((pred_col, pred_col),)

    return CustomPandasData


def _load_timellm_inference(csv_path: Path, *, cutoff: Optional[pd.Timestamp]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{csv_path.name} missing timestamp")
    df = df.copy()
    df["timestamp"] = _parse_timestamp_series(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {missing}")

    if cutoff is not None:
        df = df[df.index > cutoff].copy()

    return df


def _load_ohlcv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.copy()
    if "timestamp" not in df.columns:
        raise ValueError(f"{csv_path.name} missing timestamp")
    df["timestamp"] = _parse_timestamp_series(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {missing}")

    return df


def _build_daily_ohlcv_full(train_csv: Path, inference_csv: Path) -> pd.DataFrame:
    train = _load_ohlcv(train_csv)
    inf = _load_ohlcv(inference_csv)
    merged = pd.concat([train, inf], axis=0)
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.sort_index()
    return merged


def _load_lstm_predictions(
    lstm_csv: Path,
    *,
    ohlcv_df: pd.DataFrame,
    cutoff: Optional[pd.Timestamp],
) -> pd.DataFrame:
    dfp = pd.read_csv(lstm_csv)
    needed = ["timestamp", "y_pred_next_close"]
    missing = [c for c in needed if c not in dfp.columns]
    if missing:
        raise ValueError(f"{lstm_csv.name} missing columns: {missing}")

    dfp = dfp.copy()
    dfp["timestamp"] = _parse_timestamp_series(dfp["timestamp"])
    dfp = dfp.dropna(subset=["timestamp"]).sort_values("timestamp")
    dfp = dfp.drop_duplicates(subset=["timestamp"], keep="last")
    dfp = dfp.set_index("timestamp")

    merged = ohlcv_df.join(dfp[["y_pred_next_close"]], how="inner")
    merged = merged.rename(columns={"y_pred_next_close": "close_predicted_1"})

    if cutoff is not None:
        merged = merged[merged.index > cutoff].copy()

    return merged


def _run_simpleai_equity(df: pd.DataFrame, *, horizon: int, cash: float, commission: float) -> Tuple[float, pd.Series]:
    pred_col = f"close_predicted_{int(horizon)}"
    if pred_col not in df.columns:
        raise ValueError(f"Missing {pred_col}")

    data_feed_class = _make_data_feed_class(pred_col)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(SimpleAIStrategy, prediction_horizon=int(horizon), confidence_threshold=0.01)
    cerebro.adddata(data_feed_class(dataname=df))
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


def _run_best_by_return_equity(
    df: pd.DataFrame,
    *,
    horizon: int,
    cash: float,
    commission: float,
) -> Tuple[str, float, float, pd.Series]:
    pred_col = f"close_predicted_{int(horizon)}"
    if pred_col not in df.columns:
        raise ValueError(f"Missing {pred_col}")

    data_feed_class = _make_data_feed_class(pred_col)

    best_name = ""
    best_sharpe = float("nan")
    best_return = float("-inf")
    best_series = pd.Series(dtype=float)

    for name, (strategy_cls, base_params) in AI_STRATEGIES.items():
        params = dict(base_params)
        params["prediction_horizon"] = int(horizon)

        cerebro = bt.Cerebro()
        cerebro.addstrategy(strategy_cls, **params)
        cerebro.adddata(data_feed_class(dataname=df))
        cerebro.broker.setcash(float(cash))
        cerebro.broker.setcommission(commission=float(commission))
        cerebro.broker.set_coc(True)

        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(EquityCurve, _name="equity")

        try:
            results = cerebro.run()
        except Exception:
            continue

        strat = results[0]
        final_value = float(strat.broker.getvalue())
        total_return = (final_value - float(cash)) / float(cash) if cash != 0 else 0.0

        sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio")
        try:
            sharpe_val = float(sharpe)
        except (TypeError, ValueError):
            sharpe_val = float("nan")

        series = strat.analyzers.equity.get_analysis()

        if total_return > best_return:
            best_name = name
            best_return = float(total_return)
            best_sharpe = sharpe_val
            best_series = series

    if best_name == "" or best_series.empty:
        raise RuntimeError("No valid strategy result")

    return best_name, float(best_sharpe), float(best_return), best_series


def _overlap_window(series_list: List[pd.Series]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    starts = [s.index.min() for s in series_list if not s.empty]
    ends = [s.index.max() for s in series_list if not s.empty]
    if not starts or not ends:
        raise ValueError("No non-empty equity curves")
    start = max(starts)
    end = min(ends)
    if start >= end:
        raise ValueError(f"No overlap window: start={start}, end={end}")
    return start, end


def _plot_equity_curves(
    curves: List[Tuple[str, pd.Series, str]],
    *,
    output_path: Path,
    title: str,
) -> None:
    overlap_start, overlap_end = _overlap_window([s for _, s, _ in curves])

    plt.figure(figsize=(12, 6))
    for label, series, _ in curves:
        clipped = series.loc[(series.index >= overlap_start) & (series.index <= overlap_end)]
        if clipped.empty:
            continue
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
    parser = argparse.ArgumentParser(description="Equity curve comparison: TimeLLM vs LSTM")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("paper_outputs_timellm_vs_lstm")),
        help="Output directory for plots.",
    )
    parser.add_argument("--cash", type=float, default=100000.0)
    parser.add_argument("--commission", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    inf_dir = repo_root / "backtesting" / "inferences"

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sources
    timellm_100 = inf_dir / "1week_100.csv"
    timellm_25 = inf_dir / "1week_25.csv"
    timellm_test = inf_dir / "test.csv"
    lstm_hourly = inf_dir / "preds_hourly_lstm_seq168.csv"
    lstm_daily = inf_dir / "preds_daily_lstm_seq30.csv"
    hpo_daily = inf_dir / "iar_LLAMA3.1_L12_daily_S_seq168_pred7_p6_s4_v1000.csv"
    daily_base = inf_dir / "ohlcv_2weeks_to_1day_100pct.csv"

    # Training cutoffs
    hourly_train = repo_root / "dataset" / "cryptex" / "hourly" / "candlesticks-h-clean.csv"
    daily_train = repo_root / "dataset" / "cryptex" / "daily" / "candlesticks-D.csv"
    hourly_cutoff = _read_train_cutoff_timestamp(hourly_train)
    daily_cutoff = _read_train_cutoff_timestamp(daily_train)

    # OHLCV for LSTM joins (must cover evaluation windows).
    daily_inference_ohlcv = repo_root / "dataset" / "cryptex" / "daily" / "inference_test_btc_D_2024_2025.csv"

    # Load datasets
    df_timellm_100_full = _load_timellm_inference(timellm_100, cutoff=None)
    df_timellm_100 = (
        df_timellm_100_full[df_timellm_100_full.index > hourly_cutoff].copy()
        if hourly_cutoff is not None
        else df_timellm_100_full.copy()
    )
    df_timellm_25 = _load_timellm_inference(timellm_25, cutoff=hourly_cutoff)
    df_timellm_test = _load_timellm_inference(timellm_test, cutoff=hourly_cutoff)

    hourly_ohlcv_eval = df_timellm_100_full[["open", "high", "low", "close", "volume"]].copy()
    daily_ohlcv_full = _build_daily_ohlcv_full(daily_train, daily_inference_ohlcv)

    df_lstm_hourly = _load_lstm_predictions(lstm_hourly, ohlcv_df=hourly_ohlcv_eval, cutoff=hourly_cutoff)
    df_lstm_daily = _load_lstm_predictions(lstm_daily, ohlcv_df=daily_ohlcv_full, cutoff=daily_cutoff)
    df_hpo_daily = _load_timellm_inference(hpo_daily, cutoff=daily_cutoff)
    df_daily_base = _load_timellm_inference(daily_base, cutoff=daily_cutoff)

    if (
        df_timellm_100.empty
        or df_timellm_25.empty
        or df_timellm_test.empty
        or df_lstm_hourly.empty
        or df_lstm_daily.empty
        or df_hpo_daily.empty
        or df_daily_base.empty
    ):
        raise ValueError("One or more datasets are empty after cutoff")

    cash = float(args.cash)
    commission = float(args.commission)

    def make_simpleai_plot(*, timellm_horizon: int, filename: str) -> None:
        s100_sh, s100 = _run_simpleai_equity(df_timellm_100, horizon=timellm_horizon, cash=cash, commission=commission)
        s25_sh, s25 = _run_simpleai_equity(df_timellm_25, horizon=timellm_horizon, cash=cash, commission=commission)
        stest_sh, stest = _run_simpleai_equity(df_timellm_test, horizon=timellm_horizon, cash=cash, commission=commission)
        lh_sh, lh = _run_simpleai_equity(df_lstm_hourly, horizon=1, cash=cash, commission=commission)
        ld_sh, ld = _run_simpleai_equity(df_lstm_daily, horizon=1, cash=cash, commission=commission)
        hpo_sh, hpo = _run_simpleai_equity(df_hpo_daily, horizon=1, cash=cash, commission=commission)
        base_sh, base = _run_simpleai_equity(df_daily_base, horizon=1, cash=cash, commission=commission)

        curves = [
            (f"TimeLLM 1week_100 | SimpleAI | h={timellm_horizon} | Sharpe={s100_sh:.3f}", s100, "timellm"),
            (f"TimeLLM 1week_25 | SimpleAI | h={timellm_horizon} | Sharpe={s25_sh:.3f}", s25, "timellm"),
            (f"TimeLLM test | SimpleAI | h={timellm_horizon} | Sharpe={stest_sh:.3f}", stest, "timellm"),
            (f"LSTM hourly | SimpleAI | h=1 | Sharpe={lh_sh:.3f}", lh, "lstm"),
            (f"LSTM daily | SimpleAI | h=1 | Sharpe={ld_sh:.3f}", ld, "lstm"),
            (f"HPO Daily (IAR) | SimpleAI | h=1 | Sharpe={hpo_sh:.3f}", hpo, "iar"),
            (f"Daily base | SimpleAI | h=1 | Sharpe={base_sh:.3f}", base, "base"),
        ]

        _plot_equity_curves(
            curves,
            output_path=out_dir / filename,
            title=f"Equity curves (SimpleAI) | TimeLLM horizon={timellm_horizon}",
        )

    def make_best_by_return_plot(*, timellm_horizon: int, filename: str) -> None:
        s100_name, s100_sh, s100_ret, s100 = _run_best_by_return_equity(
            df_timellm_100, horizon=timellm_horizon, cash=cash, commission=commission
        )
        s25_name, s25_sh, s25_ret, s25 = _run_best_by_return_equity(
            df_timellm_25, horizon=timellm_horizon, cash=cash, commission=commission
        )
        stest_name, stest_sh, stest_ret, stest = _run_best_by_return_equity(
            df_timellm_test, horizon=timellm_horizon, cash=cash, commission=commission
        )
        lh_name, lh_sh, lh_ret, lh = _run_best_by_return_equity(
            df_lstm_hourly, horizon=1, cash=cash, commission=commission
        )
        ld_name, ld_sh, ld_ret, ld = _run_best_by_return_equity(
            df_lstm_daily, horizon=1, cash=cash, commission=commission
        )
        hpo_name, hpo_sh, hpo_ret, hpo = _run_best_by_return_equity(
            df_hpo_daily, horizon=1, cash=cash, commission=commission
        )
        base_name, base_sh, base_ret, base = _run_best_by_return_equity(
            df_daily_base, horizon=1, cash=cash, commission=commission
        )

        curves = [
            (
                f"TimeLLM 1week_100 | {s100_name} | h={timellm_horizon} | Return={s100_ret*100:.1f}% | Sharpe={s100_sh:.3f}",
                s100,
                "timellm",
            ),
            (
                f"TimeLLM 1week_25 | {s25_name} | h={timellm_horizon} | Return={s25_ret*100:.1f}% | Sharpe={s25_sh:.3f}",
                s25,
                "timellm",
            ),
            (
                f"TimeLLM test | {stest_name} | h={timellm_horizon} | Return={stest_ret*100:.1f}% | Sharpe={stest_sh:.3f}",
                stest,
                "timellm",
            ),
            (
                f"LSTM hourly | {lh_name} | h=1 | Return={lh_ret*100:.1f}% | Sharpe={lh_sh:.3f}",
                lh,
                "lstm",
            ),
            (
                f"LSTM daily | {ld_name} | h=1 | Return={ld_ret*100:.1f}% | Sharpe={ld_sh:.3f}",
                ld,
                "lstm",
            ),
            (
                f"HPO Daily (IAR) | {hpo_name} | h=1 | Return={hpo_ret*100:.1f}% | Sharpe={hpo_sh:.3f}",
                hpo,
                "iar",
            ),
            (
                f"Daily base | {base_name} | h=1 | Return={base_ret*100:.1f}% | Sharpe={base_sh:.3f}",
                base,
                "base",
            ),
        ]

        _plot_equity_curves(
            curves,
            output_path=out_dir / filename,
            title=f"Equity curves (Best by Return) | TimeLLM horizon={timellm_horizon}",
        )

    make_simpleai_plot(timellm_horizon=24, filename="equity_simpleai_timellmPred24.png")
    make_simpleai_plot(timellm_horizon=1, filename="equity_simpleai_timellmPred1.png")
    make_best_by_return_plot(timellm_horizon=24, filename="equity_bestByReturn_timellmPred24.png")
    make_best_by_return_plot(timellm_horizon=1, filename="equity_bestByReturn_timellmPred1.png")

    print(f"Saved plots in: {out_dir}")


if __name__ == "__main__":
    main()

