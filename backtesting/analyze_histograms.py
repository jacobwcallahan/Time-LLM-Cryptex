import argparse
from pathlib import Path
import tempfile
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from backtest import BacktestRunner, STRATEGIES
from utils import load_and_prepare_data

sns.set_theme(style="whitegrid")

TRAIN_HOURLY_DATA_PATH = Path("dataset/cryptex/hourly/candlesticks-h-clean.csv")
RETURNS_PREDICTION_HORIZON = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate histograms summarizing backtest prediction errors, directional misses, and returns."
    )
    parser.add_argument(
        "--input-dir",
        default=str(Path(__file__).resolve().parent / "inferences"),
        help="Directory containing inference CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="histogram_reports",
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=100000.0,
        help="Initial cash for each backtest run.",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.001,
        help="Commission rate applied during backtests.",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip running backtests (only error and direction histograms will be produced).",
    )
    parser.add_argument(
        "--strategies",
        default=None,
        help="Comma-separated list of strategies to evaluate (defaults to all except BuyHold).",
    )
    return parser.parse_args()


def determine_prediction_column(columns: List[str]) -> Optional[str]:
    for suffix in ["_1", ""]:
        target = f"close_predicted{suffix}"
        for col in columns:
            if col == target or col.startswith("close_predicted_"):
                return col if suffix else "close_predicted_1"
    for col in columns:
        if col.startswith("close_predicted_"):
            return col
    return None


BASELINE_SOURCE_PATH = Path("backtesting/data/candlesticks-hourly-inference_set.csv")


def _to_datetime_series(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.number):
        return pd.to_datetime(series, unit="s")
    return pd.to_datetime(series)


def generate_naive_baseline_predictions(
    source_path: Path,
    *,
    window_hours: int = 168,
    horizon: int = 24,
) -> Path:
    if not source_path.exists():
        raise FileNotFoundError(f"Baseline source data not found: {source_path}")

    df = pd.read_csv(source_path)
    if "timestamp" not in df.columns:
        raise ValueError("Baseline source must contain 'timestamp' column.")
    if "close" not in df.columns:
        raise ValueError("Baseline source must contain 'close' column.")

    df = df.copy()
    df["timestamp"] = _to_datetime_series(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    close = df["close"].astype(float)
    hourly_returns = close.pct_change()
    avg_return = hourly_returns.rolling(window_hours, min_periods=window_hours).mean()
    growth = 1 + avg_return

    for h in range(1, horizon + 1):
        col = f"close_predicted_{h}"
        df[col] = close * (growth ** h)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return Path(tmp.name)


def collect_all_errors(
    file_path: Path,
    *,
    train_data_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Collect all individual prediction errors (not aggregated) from a CSV file."""
    df, _ = load_and_prepare_data(
        str(file_path),
        train_data_path=str(train_data_path) if train_data_path is not None else None,
    )
    pred_col = determine_prediction_column(df.columns)
    if not pred_col or pred_col not in df.columns:
        raise ValueError(f"No prediction columns found in {file_path.name}")

    df = df.copy()
    df["abs_pct_error"] = ((df["close"] - df[pred_col]).abs() / df["close"]) * 100
    df["inference"] = file_path.stem
    
    result_df = pd.DataFrame({
        "inference": df["inference"],
        "abs_pct_error": df["abs_pct_error"]
    })
    
    return result_df


def _compute_monthly_returns(df: pd.DataFrame, value_column: str) -> pd.Series:
    df = df.copy()
    df["month"] = df.index.to_period("M")

    def calc_monthly_return(series: pd.Series) -> float:
        series = series.dropna()
        if series.empty:
            return 0.0
        start_price = series.iloc[0]
        end_price = series.iloc[-1]
        if start_price == 0:
            return 0.0
        return ((end_price - start_price) / start_price) * 100

    returns_by_month = df.groupby("month")[value_column].apply(calc_monthly_return)

    returns_by_month.index = returns_by_month.index.to_timestamp()
    returns_by_month.index.name = "month"

    return returns_by_month


def collect_returns_by_month(
    file_path: Path,
    prediction_column: str,
    *,
    train_data_path: Optional[Path] = None,
) -> pd.Series:
    """Collect monthly returns from a specific prediction column."""
    df, _ = load_and_prepare_data(
        str(file_path),
        train_data_path=str(train_data_path) if train_data_path is not None else None,
    )

    if prediction_column not in df.columns:
        raise ValueError(
            f"Prediction column '{prediction_column}' not found in {file_path.name}"
        )

    return _compute_monthly_returns(df, prediction_column)


def collect_actual_returns_by_month(
    file_path: Path,
    *,
    train_data_path: Optional[Path] = None,
) -> pd.Series:
    """Collect monthly returns from actual closing prices."""
    df, _ = load_and_prepare_data(
        str(file_path),
        train_data_path=str(train_data_path) if train_data_path is not None else None,
    )
    return _compute_monthly_returns(df, "close")


def collect_error_and_direction_data(
    file_path: Path,
    *,
    train_data_path: Optional[Path] = None,
) -> Tuple[pd.Series, pd.Series]:
    df, _ = load_and_prepare_data(
        str(file_path),
        train_data_path=str(train_data_path) if train_data_path is not None else None,
    )
    pred_col = determine_prediction_column(df.columns)
    if not pred_col or pred_col not in df.columns:
        raise ValueError(f"No prediction columns found in {file_path.name}")

    df = df.copy()
    df["abs_pct_error"] = ((df["close"] - df[pred_col]).abs() / df["close"]) * 100
    df["month"] = df.index.to_period("M")
    error_by_month = df.groupby("month")["abs_pct_error"].mean()

    actual_diff = df["close"].diff()
    predicted_diff = df[pred_col].diff()
    direction_mismatch = (np.sign(actual_diff) != np.sign(predicted_diff)).astype(int)
    direction_mismatch = direction_mismatch.fillna(0)
    df["direction_miss"] = direction_mismatch
    direction_by_month = df.groupby("month")["direction_miss"].sum()

    error_by_month.index = error_by_month.index.to_timestamp()
    direction_by_month.index = direction_by_month.index.to_timestamp()

    return error_by_month, direction_by_month


def run_backtests_for_returns(
    file_path: Path,
    cash: float,
    commission: float,
    strategy_names: List[str],
    *,
    train_data_path: Optional[Path] = None,
) -> pd.DataFrame:
    runner = BacktestRunner(
        str(file_path),
        cash=cash,
        commission=commission,
        train_data_path=str(train_data_path) if train_data_path is not None else None,
    )

    records: List[Dict[str, float]] = []
    for name in strategy_names:
        # Ensure returns use the intended forecast horizon.
        spec = STRATEGIES.get(name)
        if spec is not None:
            spec.setdefault("params", {})["prediction_horizon"] = RETURNS_PREDICTION_HORIZON
        try:
            runner.run_strategy(name)
        except Exception as exc:
            print(f"[Warning] Failed to run {name} on {file_path.name}: {exc}")
            continue

        result = runner.results.get(name)
        if not result:
            continue

        records.append(
            {
                "inference": file_path.stem,
                "strategy": name,
                "total_return": result["total_return"],
            }
        )

    return pd.DataFrame(records)


def plot_error_histogram(
    aggregated: pd.DataFrame,
    output_dir: Path,
) -> None:
    if aggregated.empty:
        return

    aggregated_sorted = aggregated.sort_values("month")

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=aggregated_sorted,
        x="month",
        y="mean_abs_error",
        hue="inference",
    )

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Absolute % Error")
    plt.xlabel("Month")
    plt.title("Prediction Error by Month")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title="Inference", bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_error_by_month.png", dpi=200)
    plt.close()


def plot_direction_histogram(
    aggregated: pd.DataFrame,
    output_dir: Path,
) -> None:
    if aggregated.empty:
        return

    aggregated_sorted = aggregated.sort_values("month")

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=aggregated_sorted,
        x="month",
        y="direction_miss_count",
        hue="inference",
    )

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Directional Miss Count")
    plt.xlabel("Month")
    plt.title("Directional Inaccuracy by Month")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title="Inference", bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "direction_inaccuracy_by_month.png", dpi=200)
    plt.close()


def plot_returns_by_month(
    aggregated: pd.DataFrame,
    output_dir: Path,
    *,
    title: str,
    filename: str,
) -> None:
    if aggregated.empty:
        return

    aggregated_sorted = aggregated.sort_values("month")

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=aggregated_sorted,
        x="month",
        y="return_pct",
        hue="inference",
    )

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Return (%)")
    plt.xlabel("Month")
    plt.title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title="Inference", bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=200)
    plt.close()


def plot_combined_returns(
    actual_returns: pd.DataFrame,
    prediction_plots: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
) -> None:
    if actual_returns.empty or not prediction_plots:
        return

    actual_sorted = actual_returns.sort_values("month")
    total_plots = 1 + len(prediction_plots)
    fig, axes = plt.subplots(total_plots, 1, figsize=(12, total_plots * 4), sharex=True)
    if total_plots == 1:
        axes = [axes]

    actual_ax = axes[0]
    sns.barplot(
        data=actual_sorted,
        x="month",
        y="return_pct",
        color="tab:blue",
        ax=actual_ax,
    )
    actual_ax.set_title("Actual Market Returns by Month")
    actual_ax.set_ylabel("Return (%)")
    actual_ax.set_xlabel("")
    actual_ax.tick_params(axis="x", rotation=45)

    for idx, (title, df) in enumerate(prediction_plots, start=1):
        df_sorted = df.sort_values("month")
        ax = axes[idx]
        sns.barplot(
            data=df_sorted,
            x="month",
            y="return_pct",
            hue="inference",
            ax=ax,
        )
        ax.set_title(title)
        ax.set_ylabel("Return (%)")
        ax.tick_params(axis="x", rotation=45)
        if idx == total_plots - 1:
            ax.set_xlabel("Month")
        else:
            ax.set_xlabel("")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(title="Inference", bbox_to_anchor=(1.04, 1), loc="upper left")

    fig.tight_layout()
    fig.savefig(output_dir / "returns_by_month_combined.png", dpi=200)
    plt.close(fig)


def plot_all_errors_histogram(
    all_errors_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create a distribution histogram showing all individual prediction errors."""
    if all_errors_df.empty:
        print("[Info] No error data to plot.")
        return

    plt.figure(figsize=(12, 6))

    unique_inferences = all_errors_df["inference"].unique()
    ax = None
    if len(unique_inferences) == 1:
        ax = sns.histplot(
            data=all_errors_df,
            x="abs_pct_error",
            bins=50,
            kde=True,
            label=unique_inferences[0],
        )
        ax.legend(title="Inference", bbox_to_anchor=(1.04, 1), loc="upper left")
    else:
        ax = sns.histplot(
            data=all_errors_df,
            x="abs_pct_error",
            hue="inference",
            bins=50,
            alpha=0.6,
            kde=True,
        )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles=handles,
                labels=labels,
                title="Inference",
                bbox_to_anchor=(1.04, 1),
                loc="upper left",
            )

    plt.xlabel("Absolute Percentage Error (%)")
    plt.ylabel("Frequency")
    plt.title("Distribution of All Prediction Errors")
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_error_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 6))
    log_ax = sns.histplot(
        data=all_errors_df,
        x="abs_pct_error",
        bins=50,
        hue="inference" if len(unique_inferences) > 1 else None,
        alpha=0.6,
    )
    if len(unique_inferences) == 1:
        log_ax.legend(title="Inference", labels=unique_inferences)
    else:
        handles, labels = log_ax.get_legend_handles_labels()
        if handles:
            log_ax.legend(
                handles=handles,
                labels=labels,
                title="Inference",
                bbox_to_anchor=(1.04, 1),
                loc="upper left",
            )
    log_ax.set_yscale("log")
    log_ax.set_ylim(bottom=1e-1)
    log_ax.set_ylabel("Frequency (log scale)")
    plt.xlabel("Absolute Percentage Error (%)")
    plt.title("Distribution of All Prediction Errors (Log Scale)")
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_error_distribution_log.png", dpi=200)
    plt.close()
def plot_returns_histogram(
    returns_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    if returns_df.empty:
        print("[Info] No returns data to plot.")
        return

    idx = returns_df.groupby("inference")["total_return"].idxmax()
    best = returns_df.loc[idx].copy()

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=best.reset_index(drop=True),
        x="inference",
        y="total_return",
        color="tab:green",
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Total Return (%)")
    plt.xlabel("Inference Run")
    plt.title("Best Strategy Return per Inference")
    plt.tight_layout()
    plt.savefig(output_dir / "best_worst_returns.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_names = ["1week_25.csv", "1week_50.csv", "1week_75.csv", "1week_100.csv"]
    csv_files = [input_dir / name for name in selected_names if (input_dir / name).exists()]
    if not csv_files:
        raise ValueError(
            f"No selected 1-week CSV files found in '{input_dir}'. Expected: {selected_names}"
        )

    error_records: List[Dict[str, float]] = []
    direction_records: List[Dict[str, float]] = []
    all_errors_list: List[pd.DataFrame] = []

    prediction_horizons = [
        (
            "pred24",
            "close_predicted_24",
            "Returns by Month (Prediction +24h)",
            "returns_by_month_pred24.png",
        ),
    ]
    returns_records: Dict[str, List[Dict[str, float]]] = {
        label: [] for label, *_ in prediction_horizons
    }
    prediction_plot_data: List[Tuple[str, pd.DataFrame]] = []
    baseline_temp_paths: Dict[str, Path] = {}

    returns_df = pd.DataFrame()
    strategy_names = list(STRATEGIES.keys())
    if "BuyHold" in strategy_names:
        strategy_names.remove("BuyHold")
    if args.strategies:
        requested = [s.strip() for s in args.strategies.split(",") if s.strip()]
        unknown = [s for s in requested if s not in STRATEGIES]
        if unknown:
            raise ValueError(f"Unknown strategies: {unknown}")
        strategy_names = [s for s in requested if s != "BuyHold"]

    actual_source = next(
        (path for path in csv_files if path.stem.lower() != "naive_baseline"),
        csv_files[0],
    )
    train_data_path = TRAIN_HOURLY_DATA_PATH if TRAIN_HOURLY_DATA_PATH.exists() else None
    actual_returns_series = collect_actual_returns_by_month(
        actual_source, train_data_path=train_data_path
    )
    actual_returns_df = actual_returns_series.rename("return_pct").reset_index()

    for file_path in csv_files:
        run_name = file_path.stem
        path_for_processing = file_path
        effective_name = run_name
        if run_name.lower().startswith("naive_baseline"):
            if run_name not in baseline_temp_paths:
                baseline_temp_paths[run_name] = generate_naive_baseline_predictions(
                    BASELINE_SOURCE_PATH
                )
            path_for_processing = baseline_temp_paths[run_name]
            effective_name = "naive_baseline"
        try:
            error_series, direction_series = collect_error_and_direction_data(
                path_for_processing,
                train_data_path=train_data_path,
            )
            all_errors_df = collect_all_errors(
                path_for_processing,
                train_data_path=train_data_path,
            )
            all_errors_df["inference"] = effective_name
            all_errors_list.append(all_errors_df)
        except Exception as exc:
            print(f"[Warning] Skipping {run_name} due to error: {exc}")
            continue

        for month, value in error_series.items():
            error_records.append(
                {
                    "inference": effective_name,
                    "month": month,
                    "mean_abs_error": value,
                }
            )

        for month, value in direction_series.items():
            direction_records.append(
                {
                    "inference": effective_name,
                    "month": month,
                    "direction_miss_count": value,
                }
            )

        for label, column_name, _, _ in prediction_horizons:
            try:
                returns_series = collect_returns_by_month(
                    path_for_processing,
                    column_name,
                    train_data_path=train_data_path,
                )
            except ValueError as exc:
                print(f"[Warning] {run_name}: {exc}")
                continue

            for month, value in returns_series.items():
                returns_records[label].append(
                    {
                        "inference": effective_name,
                        "month": month,
                        "return_pct": value,
                    }
                )

        if not args.skip_backtest:
            returns_subset = run_backtests_for_returns(
                file_path,
                cash=args.cash,
                commission=args.commission,
                strategy_names=strategy_names,
                train_data_path=train_data_path,
            )
            returns_df = pd.concat([returns_df, returns_subset], ignore_index=True)

    if all_errors_list:
        all_errors_combined = pd.concat(all_errors_list, ignore_index=True)
        plot_all_errors_histogram(all_errors_combined, output_dir)

    if error_records:
        error_df = pd.DataFrame(error_records)
        plot_error_histogram(error_df, output_dir)

    if direction_records:
        direction_df = pd.DataFrame(direction_records)
        plot_direction_histogram(direction_df, output_dir)

    # Ensure stale plot from prior runs doesn't linger.
    stale_pred1 = output_dir / "returns_by_month_pred1.png"
    if stale_pred1.exists():
        stale_pred1.unlink()

    for label, _, title, filename in prediction_horizons:
        horizon_records = returns_records.get(label, [])
        if horizon_records:
            returns_monthly_df = pd.DataFrame(horizon_records)
            prediction_plot_data.append((title, returns_monthly_df))
            plot_returns_by_month(
                returns_monthly_df,
                output_dir,
                title=title,
                filename=filename,
            )

    if not actual_returns_df.empty and prediction_plot_data:
        plot_combined_returns(actual_returns_df, prediction_plot_data, output_dir)

    if not args.skip_backtest and not returns_df.empty:
        plot_returns_histogram(returns_df, output_dir)

    print(f"[Done] Histograms saved in {output_dir}")


if __name__ == "__main__":
    main()



