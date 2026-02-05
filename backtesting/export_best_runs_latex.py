#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math

import pandas as pd

from backtest import BacktestRunner, STRATEGIES


TRAIN_HOURLY_DATA_PATH = Path("dataset/cryptex/hourly/candlesticks-h-clean.csv")
INFERENCES = [
    ("25", Path("backtesting/inferences/1week_25.csv")),
    ("50", Path("backtesting/inferences/1week_50.csv")),
    ("75", Path("backtesting/inferences/1week_75.csv")),
    ("100", Path("backtesting/inferences/1week_100.csv")),
]
PREDICTION_HORIZON_HOURS = 24
CONFIDENCE_THRESHOLD = 0.01


def _as_float(x) -> float:
    try:
        val = float(x)
        if math.isnan(val) or math.isinf(val):
            return float("nan")
        return val
    except Exception:
        return float("nan")


def _run_all_strategies(
    inference_csv: Path,
    *,
    cash: float = 100000.0,
    commission: float = 0.001,
) -> pd.DataFrame:
    if not inference_csv.exists():
        raise FileNotFoundError(inference_csv)

    train_path: Optional[str] = None
    if TRAIN_HOURLY_DATA_PATH.exists():
        train_path = str(TRAIN_HOURLY_DATA_PATH)

    runner = BacktestRunner(
        str(inference_csv),
        cash=cash,
        commission=commission,
        train_data_path=train_path,
    )

    strategy_names = ["SimpleAI"]
    rows: List[Dict[str, object]] = []

    for name in strategy_names:
        spec = STRATEGIES.get(name, {})
        spec.setdefault("params", {})["prediction_horizon"] = PREDICTION_HORIZON_HOURS
        spec.setdefault("params", {})["confidence_threshold"] = CONFIDENCE_THRESHOLD

        try:
            runner.run_strategy(name)
        except Exception:
            continue

        result = runner.results.get(name)
        if not result:
            continue

        analyzers = result.get("analyzers", {}) or {}
        sharpe = None
        sharpe_dict = analyzers.get("sharpe")
        if isinstance(sharpe_dict, dict):
            sharpe = sharpe_dict.get("sharperatio")

        rows.append(
            {
                "strategy": name,
                "total_return_pct": _as_float(result.get("total_return")),
                "sharpe": _as_float(sharpe),
            }
        )

    return pd.DataFrame(rows)


def _pick_best_by_return(df: pd.DataFrame) -> Tuple[Optional[str], float, float]:
    if df.empty:
        return None, float("nan"), float("nan")
    row = df.iloc[0]
    return (
        str(row["strategy"]),
        float(row["total_return_pct"]),
        float(row["sharpe"]),
    )


def _pick_best_by_sharpe(df: pd.DataFrame) -> Tuple[Optional[str], float, float]:
    if df.empty:
        return None, float("nan"), float("nan")
    row = df.iloc[0]
    return (
        str(row["strategy"]),
        float(row["sharpe"]),
        float(row["total_return_pct"]),
    )


def _fmt(x: float, *, ndigits: int = 3) -> str:
    if x is None:
        return ""
    try:
        x = float(x)
    except Exception:
        return ""
    if math.isnan(x) or math.isinf(x):
        return ""
    return f"{x:.{ndigits}f}"


def main() -> None:
    out_dir = Path("data_percentage").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    best_by_return_rows: List[Dict[str, str]] = []
    best_by_sharpe_rows: List[Dict[str, str]] = []

    for pct, path in INFERENCES:
        df = _run_all_strategies(path)

        best_r_name, best_r_ret, best_r_sharpe = _pick_best_by_return(df)
        best_by_return_rows.append(
            {
                "Percent": pct,
                "BestStrategy": best_r_name or "",
                "TotalReturnPct": _fmt(best_r_ret, ndigits=3),
                "Sharpe": _fmt(best_r_sharpe, ndigits=3),
            }
        )

        best_s_name, best_s_sharpe, best_s_ret = _pick_best_by_sharpe(df)
        best_by_sharpe_rows.append(
            {
                "Percent": pct,
                "BestStrategy": best_s_name or "",
                "Sharpe": _fmt(best_s_sharpe, ndigits=3),
                "TotalReturnPct": _fmt(best_s_ret, ndigits=3),
            }
        )

    best_by_return = pd.DataFrame(best_by_return_rows)
    best_by_sharpe = pd.DataFrame(best_by_sharpe_rows)

    (out_dir / "best_by_return.tex").write_text(
        best_by_return.to_latex(index=False, escape=True),
        encoding="utf-8",
    )
    (out_dir / "best_by_sharpe.tex").write_text(
        best_by_sharpe.to_latex(index=False, escape=True),
        encoding="utf-8",
    )

    print(f"[Done] Wrote LaTeX tables to {out_dir}")
    print("\nBest by return:")
    print(best_by_return.to_string(index=False))
    print("\nBest by Sharpe:")
    print(best_by_sharpe.to_string(index=False))


if __name__ == "__main__":
    main()



