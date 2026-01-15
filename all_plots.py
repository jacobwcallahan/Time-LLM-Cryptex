#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob

import backtrader as bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------- Strategies ----------

class _BasePredStrategy(bt.Strategy):
    params = (
        ("prediction_horizon", 1),
        ("confidence_threshold", 0.01),
        ("trade_size", 1.0),  # fraction of available cash
    )

    def _pred_col(self) -> str:
        return f"close_predicted_{int(self.p.prediction_horizon)}"

    def _get_pred(self) -> Optional[float]:
        line = getattr(self.data, self._pred_col(), None)
        if line is None:
            return None
        v = float(line[0])
        if np.isnan(v) or np.isinf(v):
            return None
        return v

    def _signal_from_pred(self) -> int:
        # +1 long, -1 short, 0 flat
        pred = self._get_pred()
        if pred is None:
            return 0
        close = float(self.data.close[0])
        if close == 0:
            return 0
        delta = (pred - close) / close
        if abs(delta) < float(self.p.confidence_threshold):
            return 0
        return 1 if delta > 0 else -1

    def _rebalance_to_signal(self, sig: int) -> None:
        pos = self.getposition(self.data).size
        if sig == 0:
            if pos != 0:
                self.close()
            return

        # Backtrader target percent is simplest for long-only, but we want optional short.
        # We'll approximate with manual sizing.
        cash = float(self.broker.getcash())
        close = float(self.data.close[0])
        if close <= 0:
            return

        target_value = cash * float(self.p.trade_size)
        size = int(target_value / close)
        if size <= 0:
            return

        if sig > 0:
            if pos <= 0:
                if pos != 0:
                    self.close()
                self.buy(size=size)
        else:
            if pos >= 0:
                if pos != 0:
                    self.close()
                self.sell(size=size)


class SimpleAIStrategy(_BasePredStrategy):
    def next(self):
        sig = self._signal_from_pred()
        self._rebalance_to_signal(sig)


class SLTPStrategy(_BasePredStrategy):
    params = (
        ("stop_loss_pct", 0.02),
        ("take_profit_pct", 0.07),
    )

    def _init_(self):
        self.entry_price = None

    def next(self):
        pos = self.getposition(self.data).size
        close = float(self.data.close[0])

        # If in position, apply SL/TP.
        if pos != 0 and self.entry_price is not None:
            pnl = (close - self.entry_price) / self.entry_price if self.entry_price != 0 else 0.0
            if pos < 0:
                pnl = -pnl
            if pnl <= -float(self.p.stop_loss_pct) or pnl >= float(self.p.take_profit_pct):
                self.close()
                self.entry_price = None
                return

        sig = self._signal_from_pred()
        prev_pos = pos
        self._rebalance_to_signal(sig)
        new_pos = self.getposition(self.data).size
        if prev_pos == 0 and new_pos != 0:
            self.entry_price = close


class MomentumAIStrategy(_BasePredStrategy):
    params = (("momentum_window", 20),)

    def _init_(self):
        self.mom = bt.indicators.Momentum(self.data.close, period=int(self.p.momentum_window))

    def next(self):
        sig = self._signal_from_pred()
        m = float(self.mom[0])
        if np.isnan(m):
            sig = 0
        else:
            if sig > 0 and m < 0:
                sig = 0
            if sig < 0 and m > 0:
                sig = 0
        self._rebalance_to_signal(sig)


class RSIAIStrategy(_BasePredStrategy):
    params = (
        ("rsi_period", 14),
        ("rsi_oversold", 30),
        ("rsi_overbought", 70),
    )

    def _init_(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=int(self.p.rsi_period))

    def next(self):
        sig = self._signal_from_pred()
        r = float(self.rsi[0])
        if np.isnan(r):
            sig = 0
        else:
            if sig > 0 and r > float(self.p.rsi_overbought):
                sig = 0
            if sig < 0 and r < float(self.p.rsi_oversold):
                sig = 0
        self._rebalance_to_signal(sig)


class BollingerAIStrategy(_BasePredStrategy):
    params = (
        ("bb_period", 20),
        ("bb_std", 2.0),
    )

    def _init_(self):
        bb = bt.indicators.BollingerBands(self.data.close, period=int(self.p.bb_period), devfactor=float(self.p.bb_std))
        self.bb_top = bb.top
        self.bb_bot = bb.bot

    def next(self):
        sig = self._signal_from_pred()
        close = float(self.data.close[0])
        top = float(self.bb_top[0])
        bot = float(self.bb_bot[0])
        if any(np.isnan(x) for x in (top, bot)):
            sig = 0
        else:
            if sig > 0 and close > top:
                sig = 0
            if sig < 0 and close < bot:
                sig = 0
        self._rebalance_to_signal(sig)


class MeanReversionAIStrategy(_BasePredStrategy):
    params = (
        ("lookback_period", 20),
        ("mean_reversion_threshold", 1.5),
    )

    def _init_(self):
        self.sma = bt.indicators.SMA(self.data.close, period=int(self.p.lookback_period))
        self.std = bt.indicators.StdDev(self.data.close, period=int(self.p.lookback_period))

    def next(self):
        sig = self._signal_from_pred()
        close = float(self.data.close[0])
        mu = float(self.sma[0])
        sd = float(self.std[0])
        if any(np.isnan(x) for x in (mu, sd)) or sd == 0:
            sig = 0
        else:
            z = (close - mu) / sd
            thr = float(self.p.mean_reversion_threshold)
            # If price is far above mean, avoid new longs; far below mean, avoid new shorts.
            if sig > 0 and z > thr:
                sig = 0
            if sig < 0 and z < -thr:
                sig = 0
        self._rebalance_to_signal(sig)


class TrendFollowingAIStrategy(_BasePredStrategy):
    params = (
        ("ema_short", 5),
        ("ema_long", 20),
    )

    def _init_(self):
        self.ema_s = bt.indicators.EMA(self.data.close, period=int(self.p.ema_short))
        self.ema_l = bt.indicators.EMA(self.data.close, period=int(self.p.ema_long))

    def next(self):
        sig = self._signal_from_pred()
        es = float(self.ema_s[0])
        el = float(self.ema_l[0])
        if any(np.isnan(x) for x in (es, el)):
            sig = 0
        else:
            # Align with trend.
            if sig > 0 and es < el:
                sig = 0
            if sig < 0 and es > el:
                sig = 0
        self._rebalance_to_signal(sig)


AI_STRATEGIES: Dict[str, Tuple[type, Dict]] = {
    "SimpleAI": (SimpleAIStrategy, dict(confidence_threshold=0.01)),
    "SLTP": (SLTPStrategy, dict(confidence_threshold=0.1, stop_loss_pct=0.02, take_profit_pct=0.07)),
    "MomentumAI": (MomentumAIStrategy, dict(confidence_threshold=0.01, momentum_window=20)),
    "RSIAI": (RSIAIStrategy, dict(confidence_threshold=0.015, rsi_period=14, rsi_oversold=30, rsi_overbought=70)),
    "BollingerAI": (BollingerAIStrategy, dict(confidence_threshold=0.01, bb_period=20, bb_std=2.0)),
    "MeanReversionAI": (MeanReversionAIStrategy, dict(confidence_threshold=0.015, lookback_period=20, mean_reversion_threshold=1.5)),
    "TrendFollowingAI": (TrendFollowingAIStrategy, dict(confidence_threshold=0.01, ema_short=5, ema_long=20)),
}


# ---------- Backtrader utilities ----------

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
        # Add a dynamic prediction line with the real column name.
        lines = (pred_col,)
        params = (
            ("datetime", None),
            ("open", "open"),
            ("high", "high"),
            ("low", "low"),
            ("close", "close"),
            ("volume", "volume"),
            ("openinterest", None),
            (pred_col, pred_col),
        )

    return CustomPandasData


def _load_timellm_inference(csv_path: Path, *, cutoff: Optional[pd.Timestamp]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{csv_path} missing timestamp")
    df = df.copy()
    df["timestamp"] = _parse_timestamp_series(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    if cutoff is not None:
        df = df[df.index > cutoff].copy()

    return df


def _load_ohlcv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{csv_path} missing timestamp")
    df = df.copy()
    df["timestamp"] = _parse_timestamp_series(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")
    return df


def _build_daily_ohlcv_full(train_csv: Path, inference_csv: Path) -> pd.DataFrame:
    train = _load_ohlcv(train_csv)
    inf = _load_ohlcv(inference_csv)
    merged = pd.concat([train, inf], axis=0)
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
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
        raise ValueError(f"{lstm_csv} missing columns: {missing}")

    dfp = dfp.copy()
    dfp["timestamp"] = _parse_timestamp_series(dfp["timestamp"])
    dfp = dfp.dropna(subset=["timestamp"]).sort_values("timestamp")
    dfp = dfp.drop_duplicates(subset=["timestamp"], keep="last").set_index("timestamp")

    merged = ohlcv_df.join(dfp[["y_pred_next_close"]], how="inner")
    merged = merged.rename(columns={"y_pred_next_close": "close_predicted_1"})

    if cutoff is not None:
        merged = merged[merged.index > cutoff].copy()
    return merged


def _run_strategy_equity(df: pd.DataFrame, *, strategy_cls: type, strategy_params: Dict, horizon: int, cash: float, commission: float) -> Tuple[float, float, pd.Series]:
    pred_col = f"close_predicted_{int(horizon)}"
    if pred_col not in df.columns:
        raise ValueError(f"Missing {pred_col}")

    data_feed_class = _make_data_feed_class(pred_col)

    cerebro = bt.Cerebro()
    params = dict(strategy_params)
    params["prediction_horizon"] = int(horizon)
    cerebro.addstrategy(strategy_cls, **params)
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

    final_value = float(strat.broker.getvalue())
    total_return = (final_value - float(cash)) / float(cash) if cash != 0 else 0.0

    series = strat.analyzers.equity.get_analysis()
    return sharpe_val, total_return, series


def _run_simpleai_equity(df: pd.DataFrame, *, horizon: int, cash: float, commission: float) -> Tuple[float, pd.Series]:
    sh, _ret, series = _run_strategy_equity(
        df,
        strategy_cls=SimpleAIStrategy,
        strategy_params=dict(confidence_threshold=0.01),
        horizon=horizon,
        cash=cash,
        commission=commission,
    )
    return sh, series


def _run_best_by_return_equity(df: pd.DataFrame, *, horizon: int, cash: float, commission: float) -> Tuple[str, float, float, pd.Series]:
    best_name = ""
    best_sharpe = float("nan")
    best_return = float("-inf")
    best_series = pd.Series(dtype=float)

    for name, (strategy_cls, base_params) in AI_STRATEGIES.items():
        try:
            sh, tr, series = _run_strategy_equity(
                df,
                strategy_cls=strategy_cls,
                strategy_params=base_params,
                horizon=horizon,
                cash=cash,
                commission=commission,
            )
        except Exception:
            continue

        if tr > best_return and not series.empty:
            best_name = name
            best_return = float(tr)
            best_sharpe = float(sh)
            best_series = series

    if best_name == "" or best_series.empty:
        raise RuntimeError("No valid strategy result")

    return best_name, best_sharpe, best_return, best_series


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


def _plot_equity_curves(curves: List[Tuple[str, pd.Series]], *, output_path: Path, title: str, normalize: bool = False) -> None:
    if normalize:
        # Normalize each curve to start at 100% for comparison across different time periods
        plt.figure(figsize=(14, 8))
        
        # Use a colormap with distinct colors
        n_curves = len(curves)
        if n_curves <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_curves]
        elif n_curves <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_curves]
        else:
            colors = plt.cm.hsv(np.linspace(0, 0.9, n_curves))

        for idx, (label, series) in enumerate(curves):
            if series.empty:
                continue
            # Normalize to 100 at start
            normalized = (series / series.iloc[0]) * 100
            plt.plot(normalized.index, normalized.values, linewidth=2.0, label=label, color=colors[idx], alpha=0.8)

        plt.title(f"{title} (Normalized)", fontsize=14, fontweight='bold')
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Portfolio Value (% of Initial)", fontsize=12)
        plt.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Initial Value')
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", fontsize=9, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
    else:
        # Original behavior - require overlap
        overlap_start, overlap_end = _overlap_window([s for _, s in curves])

        # Use a colormap with distinct colors
        n_curves = len(curves)
        if n_curves <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_curves]
        elif n_curves <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_curves]
        else:
            colors = plt.cm.hsv(np.linspace(0, 0.9, n_curves))

        plt.figure(figsize=(14, 8))
        for idx, (label, series) in enumerate(curves):
            clipped = series.loc[(series.index >= overlap_start) & (series.index <= overlap_end)]
            if clipped.empty:
                continue
            plt.plot(clipped.index, clipped.values, linewidth=2.0, label=label, color=colors[idx], alpha=0.8)

        plt.title(f"{title} | From {overlap_start.date()} to {overlap_end.date()}", fontsize=14, fontweight='bold')
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Portfolio Value ($)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", fontsize=9, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()


# ---------- Main ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Equity curve comparison for multiple assets")
    p.add_argument("--infs-dir", type=str, default=str(Path("dataset/other_assets/infs")),
                   help="Directory containing inference CSV files")
    p.add_argument("--output-dir", type=str, default=str(Path("paper_outputs_other_assets")))
    p.add_argument("--cash", type=float, default=100000.0)
    p.add_argument("--commission", type=float, default=0.001)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    infs_dir = Path(args.infs_dir).expanduser().resolve()
    if not infs_dir.exists():
        raise ValueError(f"Directory does not exist: {infs_dir}")

    # Find all CSV files in the infs directory
    csv_files = sorted(infs_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {infs_dir}")

    print(f"Found {len(csv_files)} CSV files to process")

    cash = float(args.cash)
    commission = float(args.commission)

    # Storage for all results across all assets
    results = {
        "simpleai_h1": [],
        "simpleai_h24": [],
        "best_h1": [],
        "best_h24": [],
        "all_strategies_h1": [],
        "all_strategies_h24": [],
    }

    # Process each CSV file and collect results
    for csv_file in csv_files:
        asset_name = csv_file.stem  # Get filename without extension
        # Clean up the asset name (remove -inf suffix if present)
        if asset_name.endswith("-inf"):
            asset_name = asset_name[:-4]
        
        print(f"\nProcessing {asset_name}...")

        try:
            # Load the inference data (no cutoff needed for other_assets)
            df = _load_timellm_inference(csv_file, cutoff=None)
            
            if df.empty:
                print(f"  Skipping {asset_name}: empty dataframe")
                continue

            # SimpleAI h=1
            try:
                sh, series = _run_simpleai_equity(df, horizon=1, cash=cash, commission=commission)
                if not series.empty:
                    results["simpleai_h1"].append((f"{asset_name} (Sharpe={sh:.3f})", series))
                    print(f"  ✓ SimpleAI h=1")
            except Exception as e:
                print(f"  ✗ SimpleAI h=1: {e}")

            # SimpleAI h=24
            try:
                sh, series = _run_simpleai_equity(df, horizon=24, cash=cash, commission=commission)
                if not series.empty:
                    results["simpleai_h24"].append((f"{asset_name} (Sharpe={sh:.3f})", series))
                    print(f"  ✓ SimpleAI h=24")
            except Exception as e:
                print(f"  ✗ SimpleAI h=24: {e}")

            # Best by Return h=1
            try:
                name, sh, ret, series = _run_best_by_return_equity(df, horizon=1, cash=cash, commission=commission)
                if not series.empty:
                    results["best_h1"].append((f"{asset_name} ({name}, Ret={ret*100:.1f}%, Sharpe={sh:.3f})", series))
                    print(f"  ✓ Best-by-Return h=1: {name}")
            except Exception as e:
                print(f"  ✗ Best-by-Return h=1: {e}")

            # Best by Return h=24
            try:
                name, sh, ret, series = _run_best_by_return_equity(df, horizon=24, cash=cash, commission=commission)
                if not series.empty:
                    results["best_h24"].append((f"{asset_name} ({name}, Ret={ret*100:.1f}%, Sharpe={sh:.3f})", series))
                    print(f"  ✓ Best-by-Return h=24: {name}")
            except Exception as e:
                print(f"  ✗ Best-by-Return h=24: {e}")

            # All strategies h=1
            try:
                strategy_results = []
                for strat_name, (strategy_cls, base_params) in AI_STRATEGIES.items():
                    try:
                        sh, tr, series = _run_strategy_equity(
                            df,
                            strategy_cls=strategy_cls,
                            strategy_params=base_params,
                            horizon=1,
                            cash=cash,
                            commission=commission,
                        )
                        if not series.empty:
                            strategy_results.append((strat_name, sh, tr, series))
                    except Exception:
                        continue
                
                if strategy_results:
                    # Find the best strategy for this asset
                    best_strat = max(strategy_results, key=lambda x: x[2])  # Max return
                    strat_name, sh, tr, series = best_strat
                    results["all_strategies_h1"].append((f"{asset_name} ({strat_name}, Ret={tr*100:.1f}%)", series))
                    print(f"  ✓ All Strategies h=1: Best={strat_name}")
            except Exception as e:
                print(f"  ✗ All Strategies h=1: {e}")

            # All strategies h=24
            try:
                strategy_results = []
                for strat_name, (strategy_cls, base_params) in AI_STRATEGIES.items():
                    try:
                        sh, tr, series = _run_strategy_equity(
                            df,
                            strategy_cls=strategy_cls,
                            strategy_params=base_params,
                            horizon=24,
                            cash=cash,
                            commission=commission,
                        )
                        if not series.empty:
                            strategy_results.append((strat_name, sh, tr, series))
                    except Exception:
                        continue
                
                if strategy_results:
                    # Find the best strategy for this asset
                    best_strat = max(strategy_results, key=lambda x: x[2])  # Max return
                    strat_name, sh, tr, series = best_strat
                    results["all_strategies_h24"].append((f"{asset_name} ({strat_name}, Ret={tr*100:.1f}%)", series))
                    print(f"  ✓ All Strategies h=24: Best={strat_name}")
            except Exception as e:
                print(f"  ✗ All Strategies h=24: {e}")

        except Exception as e:
            print(f"  ✗ Error processing {asset_name}: {e}")
            continue

    # Create combined plots
    print("\n\nCreating combined plots...")
    
    if results["simpleai_h1"]:
        _plot_equity_curves(
            results["simpleai_h1"],
            output_path=out_dir / "combined_simpleai_h1.png",
            title="All Assets - SimpleAI Strategy (horizon=1)",
            normalize=True
        )
        print(f"  ✓ Created combined_simpleai_h1.png ({len(results['simpleai_h1'])} assets)")

    if results["simpleai_h24"]:
        _plot_equity_curves(
            results["simpleai_h24"],
            output_path=out_dir / "combined_simpleai_h24.png",
            title="All Assets - SimpleAI Strategy (horizon=24)",
            normalize=True
        )
        print(f"  ✓ Created combined_simpleai_h24.png ({len(results['simpleai_h24'])} assets)")

    if results["best_h1"]:
        _plot_equity_curves(
            results["best_h1"],
            output_path=out_dir / "combined_bestByReturn_h1.png",
            title="All Assets - Best Strategy by Return (horizon=1)",
            normalize=True
        )
        print(f"  ✓ Created combined_bestByReturn_h1.png ({len(results['best_h1'])} assets)")

    if results["best_h24"]:
        _plot_equity_curves(
            results["best_h24"],
            output_path=out_dir / "combined_bestByReturn_h24.png",
            title="All Assets - Best Strategy by Return (horizon=24)",
            normalize=True
        )
        print(f"  ✓ Created combined_bestByReturn_h24.png ({len(results['best_h24'])} assets)")

    if results["all_strategies_h1"]:
        _plot_equity_curves(
            results["all_strategies_h1"],
            output_path=out_dir / "combined_allStrategies_h1.png",
            title="All Assets - Best Strategy per Asset (horizon=1)",
            normalize=True
        )
        print(f"  ✓ Created combined_allStrategies_h1.png ({len(results['all_strategies_h1'])} assets)")

    if results["all_strategies_h24"]:
        _plot_equity_curves(
            results["all_strategies_h24"],
            output_path=out_dir / "combined_allStrategies_h24.png",
            title="All Assets - Best Strategy per Asset (horizon=24)",
            normalize=True
        )
        print(f"  ✓ Created combined_allStrategies_h24.png ({len(results['all_strategies_h24'])} assets)")

    print(f"\n✓ All combined plots saved in: {out_dir}")


if __name__ == "__main__":
    main()