import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import backtrader as bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_equity_curves_best_return as pe


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


def _rmse_horizon(df: pd.DataFrame, pred_col: str, horizon: int) -> float:
    close = df["close"].astype(float)
    pred = df[pred_col].astype(float)
    future_close = close.shift(-int(horizon))
    tmp = pd.concat([future_close, pred], axis=1, join="inner").dropna()
    if tmp.empty:
        return float("nan")
    diff = tmp.iloc[:, 0] - tmp.iloc[:, 1]
    return float(np.sqrt((diff ** 2).mean()))


def _abs_error_usd(df: pd.DataFrame, pred_col: str, horizon: int) -> pd.Series:
    close = df["close"].astype(float)
    pred = df[pred_col].astype(float)
    future_close = close.shift(-int(horizon))
    tmp = pd.concat([future_close, pred], axis=1, join="inner").dropna()
    if tmp.empty:
        return pd.Series(dtype=float)
    err = (tmp.iloc[:, 0] - tmp.iloc[:, 1]).abs()
    return err.dropna()


def _daily_mae_from_hourly_pred24(df: pd.DataFrame) -> pd.Series:
    pred_col = "close_predicted_24"
    if pred_col not in df.columns:
        return pd.Series(dtype=float)
    hourly_err = _abs_error_usd(df, pred_col, 24)
    if hourly_err.empty:
        return hourly_err
    daily = hourly_err.groupby(hourly_err.index.floor("D")).mean()
    return daily.dropna()


def _plot_error_histogram(
    errors: pd.Series,
    *,
    output_path: Path,
    title: str,
    bins: int = 50,
) -> None:
    if errors.empty:
        return

    plt.figure(figsize=(10, 5))
    plt.hist(errors.values, bins=bins, edgecolor="black", linewidth=0.5, alpha=0.8)
    plt.title(title)
    plt.xlabel("Absolute error (USD)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_error_histogram_binned(
    errors: pd.Series,
    *,
    output_path: Path,
    title: str,
    bins: int = 20,
) -> None:
    if errors.empty:
        return

    values = errors.values.astype(float)
    counts, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = edges[1:] - edges[:-1]

    plt.figure(figsize=(12, 5))
    plt.bar(
        centers,
        counts,
        width=widths * 0.9,
        align="center",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
    )
    plt.title(title)
    plt.xlabel("Absolute error (USD)")
    plt.ylabel("Frequency")

    labels = [f"{edges[i]:,.0f}-{edges[i + 1]:,.0f}" for i in range(len(edges) - 1)]
    plt.xticks(centers, labels, rotation=45, ha="right", fontsize=8)

    y_max = float(max(counts)) if len(counts) else 0.0
    if y_max > 0:
        plt.ylim(0, y_max * 1.15)
        for x, c in zip(centers, counts):
            if c <= 0:
                continue
            plt.text(
                x,
                c + y_max * 0.02,
                str(int(c)),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _mda(actual: pd.Series, predicted: pd.Series) -> float:
    df = pd.concat([actual.astype(float), predicted.astype(float)], axis=1, join="inner").dropna()
    if df.empty:
        return float("nan")
    actual_diff = df.iloc[:, 0].diff().fillna(0)
    predicted_diff = df.iloc[:, 1].diff().fillna(0)
    directional_match = (actual_diff * predicted_diff) >= 0
    return float(directional_match.mean())


def _mda_horizon(df: pd.DataFrame, pred_col: str, horizon: int) -> float:
    close = df["close"].astype(float)
    pred = df[pred_col].astype(float)

    future_close = close.shift(-int(horizon))
    actual_move = future_close - close
    pred_move = pred - close

    tmp = pd.concat([actual_move, pred_move], axis=1, join="inner").dropna()
    if tmp.empty:
        return float("nan")

    # Ignore zero moves to avoid inflating directional accuracy.
    tmp = tmp[(tmp.iloc[:, 0] != 0) & (tmp.iloc[:, 1] != 0)]
    if tmp.empty:
        return float("nan")

    actual_sign = np.sign(tmp.iloc[:, 0])
    pred_sign = np.sign(tmp.iloc[:, 1])
    return float((actual_sign == pred_sign).mean())


def _read_train_cutoffs(repo_root: Path, train_hourly: Path, train_daily: Path) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    hourly_path = (repo_root / train_hourly).resolve() if not train_hourly.is_absolute() else train_hourly.resolve()
    daily_path = (repo_root / train_daily).resolve() if not train_daily.is_absolute() else train_daily.resolve()
    return pe._read_train_cutoff_timestamp(hourly_path), pe._read_train_cutoff_timestamp(daily_path)


def _load_with_cutoff(
    csv_path: Path,
    *,
    hourly_cutoff: Optional[pd.Timestamp],
    daily_cutoff: Optional[pd.Timestamp],
) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path, usecols=["timestamp"])
    idx = pe._parse_timestamp_series(df_raw["timestamp"]).dropna()
    is_hourly = pe._is_hourly_index(pd.DatetimeIndex(idx))
    cutoff = hourly_cutoff if is_hourly else daily_cutoff
    return pe._load_inference_csv(csv_path, train_cutoff=cutoff)


def _run_strategy_metrics(
    df: pd.DataFrame,
    *,
    strategy_cls: type,
    strategy_params: Dict,
    prediction_horizon: int,
    cash: float,
    commission: float,
) -> Dict:
    pred_col = f"close_predicted_{int(prediction_horizon)}"
    if pred_col not in df.columns:
        raise ValueError(f"Missing prediction column '{pred_col}'")

    data_feed_class = pe._make_data_feed_class(pred_col)

    cerebro = bt.Cerebro()
    params = dict(strategy_params)
    params["prediction_horizon"] = int(prediction_horizon)
    cerebro.addstrategy(strategy_cls, **params)

    data_feed = data_feed_class(dataname=df)
    cerebro.adddata(data_feed)

    cerebro.broker.setcash(float(cash))
    cerebro.broker.setcommission(commission=float(commission))
    cerebro.broker.set_coc(True)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(EquityCurve, _name="equity")

    results = cerebro.run()
    strat = results[0]

    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    try:
        sharpe_val = float(sharpe)
    except (TypeError, ValueError):
        sharpe_val = float("nan")

    drawdown = strat.analyzers.drawdown.get_analysis()
    max_dd = drawdown.get("max", {}).get("drawdown")
    try:
        max_dd_pct = float(max_dd) * 100 if max_dd is not None else float("nan")
    except (TypeError, ValueError):
        max_dd_pct = float("nan")

    trades = strat.analyzers.trades.get_analysis()
    total_trades = int(trades.get("total", {}).get("total", 0) or 0)
    won_trades = int(trades.get("won", {}).get("total", 0) or 0)
    win_rate = (won_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    final_value = float(strat.broker.getvalue())
    total_return = (final_value - float(cash)) / float(cash) if cash != 0 else 0.0

    eq = strat.analyzers.equity.get_analysis()

    rmse_val = _rmse_horizon(df, pred_col, prediction_horizon)
    mda_val = _mda_horizon(df, pred_col, prediction_horizon)

    return {
        "final_value": final_value,
        "total_return": float(total_return),
        "sharpe": sharpe_val,
        "max_drawdown_pct": max_dd_pct,
        "total_trades": total_trades,
        "win_rate_pct": float(win_rate),
        "rmse": float(rmse_val),
        "mda_pct": float(mda_val) * 100 if not np.isnan(mda_val) else float("nan"),
        "equity": eq,
    }


def _feature_engineered_files(repo_root: Path, input_dir: Path) -> List[Path]:
    feature_summary = repo_root / "dataset" / "cryptex" / "daily" / "feature_sets_summary.csv"
    if not feature_summary.exists():
        return []
    df = pd.read_csv(feature_summary)
    if "feature_set" not in df.columns:
        return []
    feature_sets = [str(x) for x in df["feature_set"].dropna().tolist()]
    files = [input_dir / f"{fs}_1week_to_1day.csv" for fs in feature_sets]
    return [p for p in files if p.exists()]


def _all_runs(repo_root: Path, input_dir: Path) -> List[Path]:
    runs: List[Path] = []
    runs.extend(_feature_engineered_files(repo_root, input_dir))
    runs.append(repo_root / "dataset" / "cryptex" / "daily" / "ohlcv_2weeks_to_1day_100pct.csv")
    runs.append(input_dir / "iar_LLAMA3.1_L12_daily_S_seq168_pred7_p6_s4_v1000.csv")
    runs.append(input_dir / "1week_100.csv")
    return [p for p in runs if p.exists()]


def _best_by_return_for_file(
    df: pd.DataFrame,
    *,
    horizon: int,
    cash: float,
    commission: float,
) -> Tuple[str, float, float, pd.Series]:
    return pe._pick_best_strategy(
        df,
        cash=cash,
        commission=commission,
        prediction_horizon=int(horizon),
        metric="total_return",
    )


def _plot_curves_best_return(
    curves: List[Tuple[str, str, float, float, pd.Series]],
    *,
    output_path: Path,
    title: str,
) -> None:
    # Reuse the existing overlap-only plotting.
    pe._plot_best(curves, output_path=output_path, title=title, show_return=True)


def _write_table(rows: List[Dict], out_path: Path) -> None:
    df = pd.DataFrame(rows)
    cols = [
        "Run",
        "Strategy",
        "TotalReturn(%)",
        "Sharpe",
        "MaxDrawdown(%)",
        "TotalTrades",
        "WinRate(%)",
        "RMSE",
        "MDA(%)",
    ]
    df = df[cols]

    # Stable ordering: feature-engineered first (alphabetical), then base/HPO/hourly.
    df = df.sort_values(["Run"]).reset_index(drop=True)

    lines: List[str] = []
    lines.append("\t".join(cols))
    for _, r in df.iterrows():
        lines.append(
            "\t".join(
                [
                    str(r["Run"]),
                    str(r["Strategy"]),
                    f"{float(r['TotalReturn(%)']):.2f}",
                    f"{float(r['Sharpe']):.3f}" if pd.notna(r["Sharpe"]) else "nan",
                    f"{float(r['MaxDrawdown(%)']):.2f}" if pd.notna(r["MaxDrawdown(%)"]) else "nan",
                    str(int(r["TotalTrades"]) if pd.notna(r["TotalTrades"]) else 0),
                    f"{float(r['WinRate(%)']):.2f}" if pd.notna(r["WinRate(%)"]) else "nan",
                    f"{float(r['RMSE']):.6g}" if pd.notna(r["RMSE"]) else "nan",
                    f"{float(r['MDA(%)']):.2f}" if pd.notna(r["MDA(%)"]) else "nan",
                ]
            )
        )

    out_path.write_text("\n".join(lines) + "\n")


def _generate_variant(
    *,
    repo_root: Path,
    input_dir: Path,
    output_dir: Path,
    hourly_horizon: int,
    cash: float,
    commission: float,
    hourly_cutoff: Optional[pd.Timestamp],
    daily_cutoff: Optional[pd.Timestamp],
    top_k: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_files = _feature_engineered_files(repo_root, input_dir)
    feature_curves: List[Tuple[str, str, float, float, pd.Series]] = []
    feature_rank: List[Tuple[float, str, str, float, float, pd.Series]] = []

    for p in feature_files:
        df = _load_with_cutoff(p, hourly_cutoff=hourly_cutoff, daily_cutoff=daily_cutoff)
        if df.empty:
            continue
        # Feature engineered are daily, horizon 1.
        try:
            strat_name, sharpe, total_return, series = _best_by_return_for_file(
                df,
                horizon=1,
                cash=cash,
                commission=commission,
            )
        except Exception:
            continue
        feature_curves.append((p.stem, strat_name, sharpe, total_return, series))
        feature_rank.append((total_return, p.stem, strat_name, sharpe, total_return, series))

    fig1 = output_dir / "fig1_equity_feature_engineered_best_return.png"
    _plot_curves_best_return(
        feature_curves,
        output_path=fig1,
        title="Equity Curves (Feature-engineered, Best by Total Return)",
    )

    feature_rank.sort(key=lambda x: (x[0] if not np.isnan(x[0]) else float("-inf")), reverse=True)
    top_features = [x[1:] for x in feature_rank[:top_k]]

    # Add base/HPO/hourly.
    extra_paths = {
        "ohlcv_2weeks_to_1day_100pct": repo_root / "dataset" / "cryptex" / "daily" / "ohlcv_2weeks_to_1day_100pct.csv",
        "iar": input_dir / "iar_LLAMA3.1_L12_daily_S_seq168_pred7_p6_s4_v1000.csv",
        "1week_100": input_dir / "1week_100.csv",
    }

    plot2_curves: List[Tuple[str, str, float, float, pd.Series]] = []
    for stem, strat_name, sharpe, total_return, series in top_features:
        plot2_curves.append((stem, strat_name, sharpe, total_return, series))

    for key, p in extra_paths.items():
        if not p.exists():
            continue
        df = _load_with_cutoff(p, hourly_cutoff=hourly_cutoff, daily_cutoff=daily_cutoff)
        if df.empty:
            continue
        horizon = hourly_horizon if key == "1week_100" else 1
        try:
            strat_name, sharpe, total_return, series = _best_by_return_for_file(
                df,
                horizon=horizon,
                cash=cash,
                commission=commission,
            )
        except Exception:
            continue
        plot2_curves.append((p.stem, strat_name, sharpe, total_return, series))

    fig2 = output_dir / "fig2_equity_top3_features_plus_bases_best_return.png"
    _plot_curves_best_return(
        plot2_curves,
        output_path=fig2,
        title=f"Equity Curves (Top {top_k} Features + Bases, Best by Total Return)",
    )

    # Error histograms (absolute USD error, horizon-aligned).
    base_path = repo_root / "dataset" / "cryptex" / "daily" / "ohlcv_2weeks_to_1day_100pct.csv"
    if base_path.exists():
        base_df = _load_with_cutoff(base_path, hourly_cutoff=hourly_cutoff, daily_cutoff=daily_cutoff)
        if not base_df.empty:
            base_pred = "close_predicted_1"
            base_err = _abs_error_usd(base_df, base_pred, 1)
            _plot_error_histogram(
                base_err,
                output_path=output_dir / "error_hist_daily_base_abs_usd.png",
                title="Daily base model | Abs error (USD) | horizon=1",
            )
            _plot_error_histogram_binned(
                base_err,
                output_path=output_dir / "error_hist_daily_base_abs_usd_binned.png",
                title="Daily base model | Abs error (USD) | horizon=1",
            )

    hpo_path = input_dir / "iar_LLAMA3.1_L12_daily_S_seq168_pred7_p6_s4_v1000.csv"
    if hpo_path.exists():
        hpo_df = _load_with_cutoff(hpo_path, hourly_cutoff=hourly_cutoff, daily_cutoff=daily_cutoff)
        if not hpo_df.empty:
            hpo_pred = "close_predicted_1"
            hpo_err = _abs_error_usd(hpo_df, hpo_pred, 1)
            _plot_error_histogram(
                hpo_err,
                output_path=output_dir / "error_hist_hpo_daily_abs_usd.png",
                title="HPO Daily | Abs error (USD) | horizon=1",
            )
            _plot_error_histogram_binned(
                hpo_err,
                output_path=output_dir / "error_hist_hpo_daily_abs_usd_binned.png",
                title="HPO Daily | Abs error (USD) | horizon=1",
            )

    hourly_path = input_dir / "1week_100.csv"
    if hourly_path.exists():
        hourly_df = _load_with_cutoff(hourly_path, hourly_cutoff=hourly_cutoff, daily_cutoff=daily_cutoff)
        if not hourly_df.empty:
            h = int(hourly_horizon)
            hourly_pred = f"close_predicted_{h}"
            if hourly_pred in hourly_df.columns:
                hourly_err = _abs_error_usd(hourly_df, hourly_pred, h)
                suffix = f"pred{h}"
                _plot_error_histogram(
                    hourly_err,
                    output_path=output_dir / f"error_hist_hourly_model_{suffix}_abs_usd.png",
                    title=f"Hourly model | Abs error (USD) | horizon={h}",
                )
                _plot_error_histogram_binned(
                    hourly_err,
                    output_path=output_dir / f"error_hist_hourly_model_{suffix}_abs_usd_binned.png",
                    title=f"Hourly model | Abs error (USD) | horizon={h}",
                )

            if h == 24:
                daily_mae = _daily_mae_from_hourly_pred24(hourly_df)
                _plot_error_histogram(
                    daily_mae,
                    output_path=output_dir / "error_hist_hourly_model_pred24_daily_aligned_mae_usd.png",
                    title="Hourly model | Daily-aligned MAE (USD) | horizon=24",
                )
                _plot_error_histogram_binned(
                    daily_mae,
                    output_path=output_dir / "error_hist_hourly_model_pred24_daily_aligned_mae_usd_binned.png",
                    title="Hourly model | Daily-aligned MAE (USD) | horizon=24",
                )

    # Table.
    rows: List[Dict] = []
    for p in _all_runs(repo_root, input_dir):
        df = _load_with_cutoff(p, hourly_cutoff=hourly_cutoff, daily_cutoff=daily_cutoff)
        if df.empty:
            continue

        horizon = hourly_horizon if p.stem == "1week_100" else 1

        # Pick best strategy by return.
        try:
            best_name, best_sharpe, best_total_return, _ = _best_by_return_for_file(
                df,
                horizon=horizon,
                cash=cash,
                commission=commission,
            )
        except Exception:
            continue

        strategy_cls, strategy_params = pe.AI_STRATEGIES[best_name]
        metrics = _run_strategy_metrics(
            df,
            strategy_cls=strategy_cls,
            strategy_params=strategy_params,
            prediction_horizon=horizon,
            cash=cash,
            commission=commission,
        )

        run_label = pe._display_label(p.stem)
        rows.append(
            {
                "Run": run_label,
                "Strategy": best_name,
                "TotalReturn(%)": metrics["total_return"] * 100.0,
                "Sharpe": metrics["sharpe"],
                "MaxDrawdown(%)": metrics["max_drawdown_pct"],
                "TotalTrades": metrics["total_trades"],
                "WinRate(%)": metrics["win_rate_pct"],
                "RMSE": metrics["rmse"],
                "MDA(%)": metrics["mda_pct"],
            }
        )

    _write_table(rows, output_dir / "table_backtest_summary.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready figures and tables for pred24 and pred1 hourly horizons."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "inferences"),
    )
    parser.add_argument("--cash", type=float, default=100000.0)
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=3)
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
    parser.add_argument(
        "--output-root",
        type=str,
        default=".",
        help="Where to create paper_outputs_hourly_pred24/ and paper_outputs_hourly_pred1/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_dir = Path(args.input_dir).expanduser().resolve()

    hourly_cutoff, daily_cutoff = _read_train_cutoffs(
        repo_root,
        Path(args.train_data_hourly),
        Path(args.train_data_daily),
    )

    out_root = Path(args.output_root).expanduser().resolve()

    _generate_variant(
        repo_root=repo_root,
        input_dir=input_dir,
        output_dir=out_root / "paper_outputs_hourly_pred24",
        hourly_horizon=24,
        cash=args.cash,
        commission=args.commission,
        hourly_cutoff=hourly_cutoff,
        daily_cutoff=daily_cutoff,
        top_k=args.top_k,
    )

    _generate_variant(
        repo_root=repo_root,
        input_dir=input_dir,
        output_dir=out_root / "paper_outputs_hourly_pred1",
        hourly_horizon=1,
        cash=args.cash,
        commission=args.commission,
        hourly_cutoff=hourly_cutoff,
        daily_cutoff=daily_cutoff,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()

