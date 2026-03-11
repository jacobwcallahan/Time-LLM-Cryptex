"""Backtest tab helpers - run backtest, fetch summary table from MLflow."""

import tempfile
import traceback
from pathlib import Path

import mlflow
import pandas as pd
import plotly.graph_objects as go

from .common import MLFLOW_TRACKING_URI, logger


def _backtest_plot_from_results(data_df, trade_log_df, strategy_name):
    """Create plotly figure: candlestick + buy/sell markers from backtest results."""
    if data_df is None or data_df.empty:
        return None
    dates = data_df.index.tolist() if hasattr(data_df.index, "tolist") else list(range(len(data_df)))
    n = len(dates)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=dates,
        open=data_df["open"].values,
        high=data_df["high"].values,
        low=data_df["low"].values,
        close=data_df["close"].values,
        name="OHLC",
        increasing_line_color="green",
        decreasing_line_color="red",
    ))

    if trade_log_df is not None and not trade_log_df.empty and "bar_open" in trade_log_df.columns:
        buy_dates, buy_prices = [], []
        for _, row in trade_log_df.iterrows():
            b = int(row["bar_open"])
            if 0 <= b < n:
                buy_dates.append(dates[b])
                buy_prices.append(float(row.get("entry_price", row.get("close", 0))))
        if buy_dates:
            fig.add_trace(go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode="markers",
                name="Buy",
                marker=dict(symbol="triangle-up", size=12, color="lime", line=dict(width=1, color="darkgreen")),
            ))
        if "bar_close" in trade_log_df.columns:
            sell_dates, sell_prices = [], []
            for _, row in trade_log_df.iterrows():
                bc = row.get("bar_close")
                if pd.notna(bc):
                    b = int(bc)
                    if 0 <= b < n:
                        sell_dates.append(dates[b])
                        sell_prices.append(float(row.get("exit_price", row.get("close", 0))))
            if sell_dates:
                fig.add_trace(go.Scatter(
                    x=sell_dates,
                    y=sell_prices,
                    mode="markers",
                    name="Sell",
                    marker=dict(symbol="triangle-down", size=12, color="red", line=dict(width=1, color="darkred")),
                ))

    fig.update_layout(
        title=f"Backtest: {strategy_name} (Buys & Sells)",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
    )
    return fig


def fetch_summary_table_from_mlflow(experiment_name, run_id):
    """Fetch backtest summary_table from MLflow run artifacts. Returns (summary_df, error_msg)."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return None, f"Experiment '{experiment_name}' not found"
        runs = client.search_runs(experiment_ids=[exp.experiment_id], filter_string=f"run_id = '{run_id}'", max_results=1)
        if not runs:
            runs = client.search_runs(experiment_ids=[exp.experiment_id], filter_string=f"tags.mlflow.runName = '{run_id}'", max_results=1)
        if not runs:
            return None, "Run not found"
        run_uuid = runs[0].info.run_id

        for artifact_path in ["summary_table.csv", "summary_table"]:
            try:
                downloaded = mlflow.artifacts.download_artifacts(run_id=run_uuid, artifact_path=artifact_path)
                path = Path(downloaded)
                if path.is_dir():
                    csv_path = path / "summary_table.csv"
                else:
                    csv_path = path
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    return df, None
            except Exception:
                continue
        return None, None
    except Exception as e:
        return None, str(e)


def run_backtest(experiment_name, run_id, strategy, initial_capital, start_date, end_date, threshold, log_to_mlflow=False):
    """Run backtest on inference data from MLflow. Returns (plot, total_return, sharpe, max_dd, win_rate, num_trades, profit_factor)."""
    null_result = (None, None, None, None, None, None, None, None, None)

    try:
        if not run_id:
            return (None, "Error: Run ID is required", None, None, None, None, None, None, None)
        if not experiment_name:
            return (None, "Error: Experiment Name is required", None, None, None, None, None, None, None)

        from backtesting.backtest import BacktestRunner, STRATEGIES

        if strategy and strategy not in STRATEGIES:
            return (None, f"Error: Strategy '{strategy}' not found", None, None, None, None, None, None, None)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return (None, f"Error: Experiment '{experiment_name}' not found", None, None, None, None, None, None, None)
        runs = client.search_runs(experiment_ids=[exp.experiment_id], filter_string=f"run_id = '{run_id}'", max_results=1)
        if not runs:
            runs = client.search_runs(experiment_ids=[exp.experiment_id], filter_string=f"tags.mlflow.runName = '{run_id}'", max_results=1)
        if not runs:
            return (None, f"Error: No run found with ID/name '{run_id}'", None, None, None, None, None, None, None)

        run_uuid = runs[0].info.run_id
        try:
            downloaded = mlflow.artifacts.download_artifacts(run_id=run_uuid, artifact_path="ohlcv_inference.csv")
            artifact_path = Path(downloaded)
            if artifact_path.is_dir():
                artifact_path = artifact_path / "ohlcv_inference.csv"
            if not artifact_path.exists():
                return (None, "Error: ohlcv_inference.csv not found in run artifacts", None, None, None, None, None, None, None)
        except Exception as e:
            return (None, f"Error: Could not download ohlcv_inference.csv: {e}", None, None, None, None, None, None, None)

        runner = BacktestRunner(
            str(artifact_path),
            cash=float(initial_capital or 10000),
            commission=0.001,
            pipeline=True,
        )

        if strategy:
            runner.run_strategy(strategy)
        else:
            runner.run_all_strategies()

        summary_df = runner.create_summary_table()
        if summary_df is None or summary_df.empty:
            return (None, "Backtest completed but no results to display.", None, None, None, None, None, None, None)

        strat_name = str(summary_df.iloc[0]["Strategy"])
        row = summary_df.iloc[0]
        total_return = float(row.get("Total Return (%)", 0))
        sharpe = float(row.get("Sharpe Ratio", 0))
        max_dd = float(row.get("Max Drawdown (%)", 0))
        win_rate = float(row.get("Win Rate (%)", 0))
        num_trades = int(row.get("Total Trades", 0))

        trade_log_df = runner.results[strat_name]["analyzers"].get("trade_log", pd.DataFrame())
        if not isinstance(trade_log_df, pd.DataFrame):
            trade_log_df = pd.DataFrame()
        if not trade_log_df.empty and "pnl" in trade_log_df.columns:
            gross_profit = trade_log_df[trade_log_df["pnl"] > 0]["pnl"].sum()
            gross_loss = abs(trade_log_df[trade_log_df["pnl"] < 0]["pnl"].sum())
            profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
        else:
            profit_factor = 0.0

        fig = _backtest_plot_from_results(runner.data, trade_log_df, strat_name)
        summary_str = summary_df.to_string(index=False, float_format="%.2f")

        if log_to_mlflow:
            try:
                tmpdir = Path(tempfile.mkdtemp())
                summary_path = tmpdir / "summary_table.csv"
                summary_df.to_csv(summary_path, index=False)
                client.log_artifact(run_id=run_uuid, local_path=str(summary_path), artifact_path="summary_table")
                summary_path.unlink(missing_ok=True)
                tmpdir.rmdir()
            except Exception as log_err:
                logger.debug(f"Failed to log summary_table to MLflow: {log_err}")

        return (fig, summary_str, total_return, sharpe, max_dd, win_rate, num_trades, profit_factor, summary_df)
    except Exception as e:
        return (None, f"Error: {str(e)}\n{traceback.format_exc()}", None, None, None, None, None, None, None)
