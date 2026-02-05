#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_equity_curves_timellm_vs_lstm as pvtl
import paper_report_artifacts as pra
from strategies import SimpleAIStrategy


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    df: pd.DataFrame
    horizon: int


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _pretty_name(key: str) -> str:
    mapping = {
        # Section 1/2/3 baselines
        "llm_hourly_pred24": "Hourly (ours) 100",
        "daily_2week": "Daily 2-week",
        "hpo_daily": "HPO daily",
        "lstm_daily": "LSTM daily",
        "lstm_hourly": "LSTM hourly",
        # Section 2 percentages
        "hourly_100": "Hourly (ours) 100%",
        "hourly_75": "Hourly (ours) 75%",
        "hourly_50": "Hourly (ours) 50%",
        "hourly_25": "Hourly (ours) 25%",
        # Feature engineered sets
        "technical": "Technical",
        "momentum": "Momentum",
        "volume_price": "Volume",
        "volatility": "Volatility",
        "onchain_price": "On-chain + Price",
        "hybrid": "Hybrid",
        "returns": "Returns",
        "temporal": "Temporal",
    }
    return mapping.get(key, key)


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
    curves: List[Tuple[str, pd.Series]],
    *,
    output_path: Path,
    title: str,
) -> None:
    overlap_start, overlap_end = _overlap_window([s for _, s in curves])

    plt.figure(figsize=(12, 6))
    for label, series in curves:
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


def _write_table(df: pd.DataFrame, *, csv_path: Path, tex_path: Path) -> None:
    df.to_csv(csv_path, index=False)
    try:
        tex = df.to_latex(index=False, float_format=lambda x: f"{x:.4f}")
        tex_path.write_text(tex)
    except Exception:
        tex_path.write_text(df.to_string(index=False))


def _abs_pct_error_horizon(df: pd.DataFrame, *, pred_col: str, horizon: int) -> pd.Series:
    close = df["close"].astype(float)
    pred = df[pred_col].astype(float)
    future_close = close.shift(-int(horizon))
    tmp = pd.concat([future_close, pred, close], axis=1, join="inner").dropna()
    if tmp.empty:
        return pd.Series(dtype=float)
    fc = tmp.iloc[:, 0]
    pr = tmp.iloc[:, 1]
    denom = fc.replace(0.0, np.nan)
    err = (fc - pr).abs() / denom * 100.0
    return err.dropna()


def _fmt_sharpe(x: float) -> str:
    try:
        if np.isnan(float(x)):
            return "nan"
        return f"{float(x):.3f}"
    except Exception:
        return "nan"


def _direction_miss_horizon(df: pd.DataFrame, *, pred_col: str, horizon: int) -> pd.Series:
    close = df["close"].astype(float)
    pred = df[pred_col].astype(float)
    future_close = close.shift(-int(horizon))
    actual_move = future_close - close
    pred_move = pred - close
    tmp = pd.concat([actual_move, pred_move], axis=1, join="inner").dropna()
    if tmp.empty:
        return pd.Series(dtype=float)
    tmp = tmp[(tmp.iloc[:, 0] != 0) & (tmp.iloc[:, 1] != 0)]
    if tmp.empty:
        return pd.Series(dtype=float)
    miss = (np.sign(tmp.iloc[:, 0]) != np.sign(tmp.iloc[:, 1])).astype(int)
    miss.index = tmp.index
    return miss


def _plot_monthly_lines(
    series_by_label: Dict[str, pd.Series],
    *,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(12, 5))
    for label, s in series_by_label.items():
        if s.empty:
            continue
        plt.plot(s.index, s.values, marker="o", linewidth=1.8, label=label)
    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_error_distribution_overlay(
    errors_by_label: Dict[str, pd.Series],
    *,
    output_path: Path,
    title: str,
    bins: int = 60,
) -> None:
    plt.figure(figsize=(12, 5))
    for label, s in errors_by_label.items():
        if s.empty:
            continue
        plt.hist(s.values, bins=bins, alpha=0.35, label=label, density=True)
    plt.title(title)
    plt.xlabel("Absolute percent error (%)")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _load_section1_models(repo_root: Path) -> List[ModelSpec]:
    inf_dir = repo_root / "backtesting" / "inferences"

    hourly_train = repo_root / "dataset" / "cryptex" / "hourly" / "candlesticks-h-clean.csv"
    daily_train = repo_root / "dataset" / "cryptex" / "daily" / "candlesticks-D.csv"
    hourly_cutoff = pvtl._read_train_cutoff_timestamp(hourly_train)
    daily_cutoff = pvtl._read_train_cutoff_timestamp(daily_train)

    timellm_hourly = inf_dir / "1week_100.csv"
    hpo_daily = inf_dir / "iar_LLAMA3.1_L12_daily_S_seq168_pred7_p6_s4_v1000.csv"
    daily_2week = inf_dir / "ohlcv_2weeks_to_1day_100pct.csv"
    lstm_hourly = inf_dir / "preds_hourly_lstm_seq168.csv"
    lstm_daily = inf_dir / "preds_daily_lstm_seq30.csv"

    df_timellm_100_full = pvtl._load_timellm_inference(timellm_hourly, cutoff=None)
    df_timellm_hourly = (
        df_timellm_100_full[df_timellm_100_full.index > hourly_cutoff].copy()
        if hourly_cutoff is not None
        else df_timellm_100_full.copy()
    )

    daily_inference_ohlcv = repo_root / "dataset" / "cryptex" / "daily" / "inference_test_btc_D_2024_2025.csv"
    hourly_ohlcv_eval = df_timellm_100_full[["open", "high", "low", "close", "volume"]].copy()
    daily_ohlcv_full = pvtl._build_daily_ohlcv_full(daily_train, daily_inference_ohlcv)

    df_lstm_hourly = pvtl._load_lstm_predictions(lstm_hourly, ohlcv_df=hourly_ohlcv_eval, cutoff=hourly_cutoff)
    df_lstm_daily = pvtl._load_lstm_predictions(lstm_daily, ohlcv_df=daily_ohlcv_full, cutoff=daily_cutoff)

    df_hpo_daily = pvtl._load_timellm_inference(hpo_daily, cutoff=daily_cutoff)
    df_daily_2week = pvtl._load_timellm_inference(daily_2week, cutoff=daily_cutoff)

    return [
        ModelSpec(
            key="llm_hourly_pred24",
            label="Hourly 100",
            df=df_timellm_hourly,
            horizon=24,
        ),
        ModelSpec(
            key="hpo_daily",
            label="HPO daily",
            df=df_hpo_daily,
            horizon=1,
        ),
        ModelSpec(
            key="daily_2week",
            label="Daily 2-week",
            df=df_daily_2week,
            horizon=1,
        ),
        ModelSpec(
            key="lstm_daily",
            label="LSTM daily",
            df=df_lstm_daily,
            horizon=1,
        ),
        ModelSpec(
            key="lstm_hourly",
            label="LSTM hourly",
            df=df_lstm_hourly,
            horizon=1,
        ),
    ]


def _load_section2_models(repo_root: Path) -> List[ModelSpec]:
    inf_dir = repo_root / "backtesting" / "inferences"
    hourly_train = repo_root / "dataset" / "cryptex" / "hourly" / "candlesticks-h-clean.csv"
    hourly_cutoff = pvtl._read_train_cutoff_timestamp(hourly_train)

    models: List[Tuple[str, str]] = [
        ("hourly_100", "1week_100.csv"),
        ("hourly_75", "1week_75.csv"),
        ("hourly_50", "1week_50.csv"),
        ("hourly_25", "1week_25.csv"),
    ]
    out: List[ModelSpec] = []
    for key, fname in models:
        df_full = pvtl._load_timellm_inference(inf_dir / fname, cutoff=None)
        df = df_full[df_full.index > hourly_cutoff].copy() if hourly_cutoff is not None else df_full
        out.append(ModelSpec(key=key, label=_pretty_name(key), df=df, horizon=24))
    return out


def _load_feature_set_models(repo_root: Path) -> List[ModelSpec]:
    inf_dir = repo_root / "backtesting" / "inferences"
    daily_train = repo_root / "dataset" / "cryptex" / "daily" / "candlesticks-D.csv"
    daily_cutoff = pvtl._read_train_cutoff_timestamp(daily_train)

    feature_sets = [
        ("momentum", "momentum_1week_to_1day.csv"),
        ("volatility", "volatility_1week_to_1day.csv"),
        ("onchain_price", "onchain_price_1week_to_1day.csv"),
        ("volume_price", "volume_price_1week_to_1day.csv"),
        ("technical", "technical_1week_to_1day.csv"),
        ("hybrid", "hybrid_1week_to_1day.csv"),
        ("returns", "returns_1week_to_1day.csv"),
        ("temporal", "temporal_1week_to_1day.csv"),
    ]

    out: List[ModelSpec] = []
    for key, fname in feature_sets:
        df = pvtl._load_timellm_inference(inf_dir / fname, cutoff=daily_cutoff)
        out.append(ModelSpec(key=key, label=_pretty_name(key), df=df, horizon=1))
    return out


def _run_simpleai_metrics(
    spec: ModelSpec,
    *,
    cash: float,
    commission: float,
) -> Dict:
    return pra._run_strategy_metrics(
        spec.df,
        strategy_cls=SimpleAIStrategy,
        strategy_params={"confidence_threshold": 0.01},
        prediction_horizon=int(spec.horizon),
        cash=float(cash),
        commission=float(commission),
    )


def _section1(out_root: Path, repo_root: Path, cash: float, commission: float) -> None:
    section_dir = out_root / "section1"
    equity_dir = section_dir / "equity"
    table_dir = section_dir / "tables"
    hist_dir = section_dir / "histograms"
    _ensure_dir(equity_dir)
    _ensure_dir(table_dir)
    _ensure_dir(hist_dir)

    specs = _load_section1_models(repo_root)
    rows = []
    curves = []
    errors_abs_usd: Dict[str, pd.Series] = {}

    for spec in specs:
        m = _run_simpleai_metrics(spec, cash=cash, commission=commission)
        pretty = _pretty_name(spec.key)
        curve_label = f"{pretty} | h={spec.horizon} | Sharpe={_fmt_sharpe(m['sharpe'])}"
        rows.append(
            {
                "model": pretty,
                "horizon": spec.horizon,
                "total_return_pct": m["total_return"] * 100.0,
                "sharpe": m["sharpe"],
                "max_drawdown_pct": m["max_drawdown_pct"],
                "total_trades": m["total_trades"],
                "win_rate_pct": m["win_rate_pct"],
                "rmse_usd": m["rmse"],
                "mda_pct": m["mda_pct"],
            }
        )
        eq = m["equity"]
        curves.append((curve_label, eq))

        pred_col = f"close_predicted_{int(spec.horizon)}"
        err = pra._abs_error_usd(spec.df, pred_col, int(spec.horizon))
        errors_abs_usd[curve_label] = err

    _plot_equity_curves(
        curves,
        output_path=equity_dir / "equity_section1_all_models_simpleai.png",
        title="Equity curves (SimpleAI)",
    )

    summary = pd.DataFrame(rows).sort_values("total_return_pct", ascending=False)
    _write_table(
        summary,
        csv_path=table_dir / "backtest_summary_section1.csv",
        tex_path=table_dir / "backtest_summary_section1.tex",
    )

    for label, err in errors_abs_usd.items():
        out_name = label.replace(" ", "_").replace("|", "").replace("(", "").replace(")", "")
        pra._plot_error_histogram(
            err,
            output_path=hist_dir / f"abs_error_usd_hist_{out_name}.png",
            title=f"Abs error (USD): {label}",
        )
        pra._plot_error_histogram_binned(
            err,
            output_path=hist_dir / f"abs_error_usd_hist_{out_name}_binned.png",
            title=f"Abs error (USD): {label}",
        )


def _section2(out_root: Path, repo_root: Path, cash: float, commission: float) -> None:
    section_dir = out_root / "section2"
    equity_dir = section_dir / "equity"
    table_dir = section_dir / "tables"
    hist_dir = section_dir / "histograms"
    _ensure_dir(equity_dir)
    _ensure_dir(table_dir)
    _ensure_dir(hist_dir)

    specs = _load_section2_models(repo_root)
    rows = []
    curves = []

    monthly_err: Dict[str, pd.Series] = {}
    monthly_miss_rate: Dict[str, pd.Series] = {}
    dist_errors: Dict[str, pd.Series] = {}

    for spec in specs:
        m = _run_simpleai_metrics(spec, cash=cash, commission=commission)
        pretty = _pretty_name(spec.key)
        curve_label = f"{pretty} | h={spec.horizon} | Sharpe={_fmt_sharpe(m['sharpe'])}"
        rows.append(
            {
                "model": pretty,
                "horizon": spec.horizon,
                "total_return_pct": m["total_return"] * 100.0,
                "sharpe": m["sharpe"],
                "max_drawdown_pct": m["max_drawdown_pct"],
                "total_trades": m["total_trades"],
                "win_rate_pct": m["win_rate_pct"],
                "rmse_usd": m["rmse"],
                "mda_pct": m["mda_pct"],
            }
        )
        curves.append((curve_label, m["equity"]))

        pred_col = f"close_predicted_{int(spec.horizon)}"
        err_pct = _abs_pct_error_horizon(spec.df, pred_col=pred_col, horizon=int(spec.horizon))
        dist_errors[curve_label] = err_pct

        miss = _direction_miss_horizon(spec.df, pred_col=pred_col, horizon=int(spec.horizon))
        if not miss.empty:
            miss_rate = miss.groupby(miss.index.to_period("M")).mean() * 100.0
            miss_rate.index = miss_rate.index.to_timestamp()
            monthly_miss_rate[curve_label] = miss_rate

        if not err_pct.empty:
            by_month = err_pct.groupby(err_pct.index.to_period("M")).mean()
            by_month.index = by_month.index.to_timestamp()
            monthly_err[curve_label] = by_month

    _plot_equity_curves(
        curves,
        output_path=equity_dir / "equity_section2_hourly_percentages_simpleai.png",
        title="Equity curves (SimpleAI) | Hourly pred24",
    )

    summary = pd.DataFrame(rows).sort_values("total_return_pct", ascending=False)
    _write_table(
        summary,
        csv_path=table_dir / "backtest_summary_section2.csv",
        tex_path=table_dir / "backtest_summary_section2.tex",
    )

    _plot_monthly_lines(
        monthly_miss_rate,
        output_path=hist_dir / "direction_inaccuracy_by_month_pct.png",
        title="Directional inaccuracy by month (miss rate)",
        ylabel="Direction miss rate (%)",
    )
    _plot_monthly_lines(
        monthly_err,
        output_path=hist_dir / "prediction_error_by_month_pct.png",
        title="Prediction error by month (abs % error)",
        ylabel="Mean absolute percent error (%)",
    )
    _plot_error_distribution_overlay(
        dist_errors,
        output_path=hist_dir / "error_distribution_overlay_abs_pct_error.png",
        title="Error distribution (abs % error) | Hourly pred24",
    )


def _section3(out_root: Path, repo_root: Path, cash: float, commission: float) -> None:
    section_dir = out_root / "section3"
    equity_dir = section_dir / "equity"
    table_dir = section_dir / "tables"
    hist_dir = section_dir / "histograms"
    _ensure_dir(equity_dir)
    _ensure_dir(table_dir)
    _ensure_dir(hist_dir)

    feature_specs = _load_feature_set_models(repo_root)

    # Baselines
    inf_dir = repo_root / "backtesting" / "inferences"
    hourly_train = repo_root / "dataset" / "cryptex" / "hourly" / "candlesticks-h-clean.csv"
    daily_train = repo_root / "dataset" / "cryptex" / "daily" / "candlesticks-D.csv"
    hourly_cutoff = pvtl._read_train_cutoff_timestamp(hourly_train)
    daily_cutoff = pvtl._read_train_cutoff_timestamp(daily_train)

    df_hourly_full = pvtl._load_timellm_inference(inf_dir / "1week_100.csv", cutoff=None)
    df_hourly = df_hourly_full[df_hourly_full.index > hourly_cutoff].copy() if hourly_cutoff is not None else df_hourly_full
    df_hpo = pvtl._load_timellm_inference(inf_dir / "iar_LLAMA3.1_L12_daily_S_seq168_pred7_p6_s4_v1000.csv", cutoff=daily_cutoff)

    daily_inference_ohlcv = repo_root / "dataset" / "cryptex" / "daily" / "inference_test_btc_D_2024_2025.csv"
    daily_ohlcv_full = pvtl._build_daily_ohlcv_full(daily_train, daily_inference_ohlcv)
    df_lstm_daily = pvtl._load_lstm_predictions(inf_dir / "preds_daily_lstm_seq30.csv", ohlcv_df=daily_ohlcv_full, cutoff=daily_cutoff)

    baselines = [
        ModelSpec(key="llm_hourly_pred24", label=_pretty_name("llm_hourly_pred24"), df=df_hourly, horizon=24),
        ModelSpec(key="hpo_daily", label=_pretty_name("hpo_daily"), df=df_hpo, horizon=1),
        ModelSpec(key="lstm_daily", label=_pretty_name("lstm_daily"), df=df_lstm_daily, horizon=1),
    ]

    rows = []
    curves_all_features = []

    feature_returns: List[Tuple[float, ModelSpec, Dict]] = []
    for spec in feature_specs:
        m = _run_simpleai_metrics(spec, cash=cash, commission=commission)
        feature_returns.append((float(m["total_return"]), spec, m))

        pretty = _pretty_name(spec.key)
        curve_label = f"{pretty} | h={spec.horizon} | Sharpe={_fmt_sharpe(m['sharpe'])}"
        rows.append(
            {
                "model": pretty,
                "horizon": spec.horizon,
                "total_return_pct": m["total_return"] * 100.0,
                "sharpe": m["sharpe"],
                "max_drawdown_pct": m["max_drawdown_pct"],
                "total_trades": m["total_trades"],
                "win_rate_pct": m["win_rate_pct"],
                "rmse_usd": m["rmse"],
                "mda_pct": m["mda_pct"],
            }
        )
        curves_all_features.append((curve_label, m["equity"]))

    _plot_equity_curves(
        curves_all_features,
        output_path=equity_dir / "equity_section3_all_feature_sets_simpleai.png",
        title="Equity curves (SimpleAI) | Feature-engineered sets",
    )

    feature_returns.sort(key=lambda x: x[0], reverse=True)
    top3 = feature_returns[:3]

    top3_curves = []
    top3_rows = []
    top3_err: Dict[str, pd.Series] = {}

    for _ret, spec, m in top3:
        pretty = _pretty_name(spec.key)
        curve_label = f"{pretty} | h={spec.horizon} | Sharpe={_fmt_sharpe(m['sharpe'])}"
        top3_curves.append((curve_label, m["equity"]))
        top3_rows.append(spec)
        pred_col = f"close_predicted_{int(spec.horizon)}"
        top3_err[curve_label] = pra._abs_error_usd(spec.df, pred_col, int(spec.horizon))

    baseline_curves = []
    for spec in baselines:
        m = _run_simpleai_metrics(spec, cash=cash, commission=commission)
        pretty = _pretty_name(spec.key)
        curve_label = f"{pretty} | h={spec.horizon} | Sharpe={_fmt_sharpe(m['sharpe'])}"
        rows.append(
            {
                "model": pretty,
                "horizon": spec.horizon,
                "total_return_pct": m["total_return"] * 100.0,
                "sharpe": m["sharpe"],
                "max_drawdown_pct": m["max_drawdown_pct"],
                "total_trades": m["total_trades"],
                "win_rate_pct": m["win_rate_pct"],
                "rmse_usd": m["rmse"],
                "mda_pct": m["mda_pct"],
            }
        )
        baseline_curves.append((curve_label, m["equity"]))

    _plot_equity_curves(
        top3_curves + baseline_curves,
        output_path=equity_dir / "equity_section3_top3_vs_baselines_simpleai.png",
        title="Top 3 feature sets vs baselines (SimpleAI)",
    )

    summary = pd.DataFrame(rows).sort_values("total_return_pct", ascending=False)
    _write_table(
        summary,
        csv_path=table_dir / "backtest_summary_section3.csv",
        tex_path=table_dir / "backtest_summary_section3.tex",
    )

    for label, err in top3_err.items():
        out_name = label.replace(" ", "_").replace("|", "").replace("(", "").replace(")", "")
        pra._plot_error_histogram(
            err,
            output_path=hist_dir / f"abs_error_usd_hist_{out_name}.png",
            title=f"Abs error (USD): {label}",
        )

    if top3_err:
        plt.figure(figsize=(12, 5))
        for label, err in top3_err.items():
            if err.empty:
                continue
            plt.hist(err.values, bins=60, alpha=0.35, density=True, label=label)
        plt.title("Error distribution (abs USD) | Top 3 feature sets")
        plt.xlabel("Absolute error (USD)")
        plt.ylabel("Density")
        plt.grid(True, alpha=0.25)
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(hist_dir / "error_distribution_overlay_top3_abs_usd.png", dpi=200)
        plt.close()

    # Best-by-total-return plots (AI strategies only; no BuyHold).
    def _best_by_return_curve(spec: ModelSpec) -> Tuple[str, pd.Series, float, float]:
        strat_name, sharpe_val, total_return, series = pvtl._run_best_by_return_equity(
            spec.df,
            horizon=int(spec.horizon),
            cash=float(cash),
            commission=float(commission),
        )
        label = (
            f"{_pretty_name(spec.key)} | {strat_name} | "
            f"Return={float(total_return)*100:.1f}% | Sharpe={_fmt_sharpe(sharpe_val)}"
        )
        return label, series, float(total_return), float(sharpe_val)

    # Plot: all feature sets, each with best-by-return strategy.
    feature_best: List[Tuple[float, ModelSpec, str, pd.Series]] = []
    curves_best_all: List[Tuple[str, pd.Series]] = []
    for spec in feature_specs:
        lbl, series, total_ret, _sh = _best_by_return_curve(spec)
        curves_best_all.append((lbl, series))
        feature_best.append((total_ret, spec, lbl, series))

    _plot_equity_curves(
        curves_best_all,
        output_path=equity_dir / "equity_section3_all_feature_sets_bestByReturn.png",
        title="Equity curves (Best by Return) | Feature-engineered sets",
    )

    # Plot: top-3 feature sets (by best-by-return total return) vs baselines (also best-by-return).
    feature_best.sort(key=lambda x: x[0], reverse=True)
    top3_best = feature_best[:3]
    top3_curves_best = [(lbl, series) for _ret, _spec, lbl, series in top3_best]

    baseline_curves_best: List[Tuple[str, pd.Series]] = []
    for spec in baselines:
        lbl, series, _ret, _sh = _best_by_return_curve(spec)
        baseline_curves_best.append((lbl, series))

    _plot_equity_curves(
        top3_curves_best + baseline_curves_best,
        output_path=equity_dir / "equity_section3_top3_vs_baselines_bestByReturn.png",
        title="Equity curves (Best by Return) | Top 3 vs baselines",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Cryptex report artifacts")
    p.add_argument(
        "--output-dir",
        default=str(Path("cryptex-report")),
        help="Output directory under repo root.",
    )
    p.add_argument("--cash", type=float, default=100000.0)
    p.add_argument("--commission", type=float, default=0.001)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_root = (repo_root / args.output_dir).resolve()
    _ensure_dir(out_root)

    cash = float(args.cash)
    commission = float(args.commission)

    _section1(out_root, repo_root, cash, commission)
    _section2(out_root, repo_root, cash, commission)
    _section3(out_root, repo_root, cash, commission)

    print(f"Wrote report artifacts to: {out_root}")


if __name__ == "__main__":
    main()



