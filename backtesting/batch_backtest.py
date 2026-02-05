import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from backtest import BacktestRunner, STRATEGIES

DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "inferences"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run backtests against locally stored inference CSV files."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing inference CSV files to backtest.",
    )
    parser.add_argument(
        "--output-dir",
        default="inference_backtests",
        help="Directory where summary text files will be stored.",
    )
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()),
        default=None,
        help="Specific strategy to backtest. If omitted, all strategies are executed.",
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
        help="Broker commission rate applied during backtests.",
    )
    return parser.parse_args()


def format_summary(runner: BacktestRunner) -> pd.DataFrame:
    summary_rows = []
    for strategy_name, result in runner.results.items():
        analyzers = result["analyzers"]
        sharpe = analyzers.get("sharpe", {}).get("sharperatio", 0) or 0
        max_dd = analyzers.get("drawdown", {}).get("max", {}).get("drawdown", 0) or 0
        trades = analyzers.get("trades", {})
        total_trades = trades.get("total", {}).get("total", 0)
        won_trades = trades.get("won", {}).get("total", 0)
        win_rate = (won_trades / total_trades * 100) if total_trades else 0

        mda = result.get("mda", float("nan"))

        summary_rows.append(
            {
                "Strategy": strategy_name,
                "Total Return (%)": result["total_return"],
                "Sharpe Ratio": sharpe,
                "Max Drawdown (%)": max_dd * 100,
                "Total Trades": total_trades,
                "Win Rate (%)": win_rate,
                "MDA (%)": mda * 100 if pd.notna(mda) else float("nan"),
                "Final Value ($)": result["final_value"],
            }
        )

    if not summary_rows:
        return pd.DataFrame()

    df = pd.DataFrame(summary_rows)
    df.sort_values("Sharpe Ratio", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def write_summary_file(
    run_name: str,
    run_id: str,
    csv_path: Path,
    summary_df: pd.DataFrame,
    runner: BacktestRunner,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / f"{run_name}_summary.txt"

    with summary_file.open("w") as f:
        f.write(f"Run Name: {run_name}\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"CSV Path: {csv_path}\n\n")

        if summary_df.empty:
            f.write("No strategies produced results.\n")
            return

        f.write("[Summary]\n")
        f.write(summary_df.to_string(index=False, float_format="%.4f"))
        f.write("\n\n[Analyzer Details]\n")

        for strategy_name, result in runner.results.items():
            analyzers = result["analyzers"]
            mda = result.get("mda")
            f.write(f"- {strategy_name}:\n")
            f.write(json.dumps(analyzers, indent=2))
            if mda is not None and not pd.isna(mda):
                f.write(f"\n  MDA: {mda:.4f}")
            f.write("\n\n")


def backtest_run(
    file_stem: str,
    csv_path: Path,
    strategy: str | None,
    cash: float,
    commission: float,
) -> BacktestRunner:
    print(f"[Debug] Starting backtest for {file_stem} using {csv_path}")
    runner = BacktestRunner(str(csv_path), cash=cash, commission=commission)

    if strategy:
        print(f"[Debug] Running strategy {strategy}")
        runner.run_strategy(strategy)
    else:
        print("[Debug] Running all strategies")
        runner.run_all_strategies()

    return runner


def main() -> None:
    print("[Debug] Parsing arguments")
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

    csv_files: List[Path] = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in '{input_dir}'.")

    base_output_dir = Path(args.output_dir).expanduser().resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Debug] Output directory: {base_output_dir}")

    successes: List[str] = []
    failures: List[tuple[str, str]] = []

    for csv_path in csv_files:
        file_stem = csv_path.stem
        print(f"[Processing] {file_stem}")
        run_output_dir = base_output_dir / file_stem

        try:
            runner = backtest_run(
                file_stem=file_stem,
                csv_path=csv_path,
                strategy=args.strategy,
                cash=args.cash,
                commission=args.commission,
            )
        except Exception as exc:
            message = f"Backtest failed: {exc}"
            print(f"  ! {message}")
            failures.append((file_stem, message))
            continue

        summary_df = format_summary(runner)
        write_summary_file(
            run_name=file_stem,
            run_id="local",
            csv_path=csv_path,
            summary_df=summary_df,
            runner=runner,
            output_dir=run_output_dir,
        )
        print(f"  Results written to {run_output_dir}")
        successes.append(file_stem)

    print("\n[Summary]")
    print(f"Successful files: {len(successes)}")
    if successes:
        print("  " + ", ".join(successes))
    print(f"Failed files: {len(failures)}")
    for file_name, reason in failures:
        print(f"  - {file_name}: {reason}")


if __name__ == "__main__":
    main()

