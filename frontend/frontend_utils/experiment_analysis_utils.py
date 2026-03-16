"""Experiment Analysis tab helpers - aggregate runs with inference + backtest by granularity."""

import traceback
from pathlib import Path

import mlflow
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from .common import MLFLOW_TRACKING_URI, logger
from .experiment_runs_utils import _get_run_artifact_status
from .inference_utils import _fetch_metrics_from_mlflow
from .backtest_utils import fetch_summary_table_from_mlflow


# Params to display (user-specified order)
DISPLAY_PARAMS = [
    "granularity",
    "features",
    "target",
    "seq_len",
    "pred_len",
    "enc_in",
    "d_model",
    "n_heads",
    "d_ff",
    "dropout",
    "patch_len",
    "stride",
    "llm_model",
    "num_workers",
    "train_epochs",
    "batch_size",
    "eval_batch_size",
    "patience",
    "learning_rate",
    "loss",
    "metric",
    "lradj",
    "pct_start",
    "use_amp",
    "llm_layers",
]


def _get_run_params(run) -> dict:
    """Extract display params from MLflow run. Returns dict of param_key -> value."""
    params = {}
    if hasattr(run, "data") and hasattr(run.data, "params"):
        for k, v in run.data.params.items():
            if k in DISPLAY_PARAMS:
                params[k] = str(v)
    return params


def fetch_experiment_analysis_data(experiment_name, pred_horizon=1):
    """
    Fetch all runs that have BOTH inference and backtest done.
    Returns data grouped by granularity for analysis.

    Returns:
        dict: {
            "runs": [{"run_id", "run_name", "granularity", "params", "mda", "mse", "mae",
                      "bt_total_return", "bt_sharpe", "bt_max_dd", "bt_win_rate", "bt_trades", "bt_strategy"}, ...],
            "by_granularity": {"h": [...], "daily": [...], ...},
            "error": str or None,
        }
    """
    result = {"runs": [], "by_granularity": {}, "error": None}
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            result["error"] = f"Experiment '{experiment_name}' not found"
            return result

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string='attributes.status = "FINISHED"',
            order_by=["start_time DESC"],
        )

        for r in runs:
            run_uuid = r.info.run_id
            run_name = r.data.tags.get("mlflow.runName", run_uuid)
            has_inf, has_bt = _get_run_artifact_status(client, run_uuid)
            if not (has_inf and has_bt):
                continue

            params = _get_run_params(r)
            granularity = params.get("granularity", "unknown")
            target = params.get("target", "close")
            # Map target to data type: returns, volatility, or ohlcv
            if target == "returns":
                data_type = "returns"
            elif target == "volatility":
                data_type = "volatility"
            else:
                data_type = "ohlcv"

            mae, mse, mda = _fetch_metrics_from_mlflow(client, run_uuid, pred_horizon)
            summary_df, bt_err = fetch_summary_table_from_mlflow(experiment_name, run_uuid)

            bt_total_return = bt_sharpe = bt_max_dd = bt_win_rate = bt_trades = bt_strategy = None
            if summary_df is not None and not summary_df.empty:
                row = summary_df.iloc[0]
                bt_strategy = str(row.get("Strategy", ""))
                bt_total_return = float(row.get("Total Return (%)", 0))
                bt_sharpe = float(row.get("Sharpe Ratio", 0))
                bt_max_dd = float(row.get("Max Drawdown (%)", 0))
                bt_win_rate = float(row.get("Win Rate (%)", 0))
                bt_trades = int(row.get("Total Trades", 0))

            run_data = {
                "run_id": run_uuid,
                "run_name": run_name,
                "granularity": granularity,
                "data_type": data_type,
                "params": params,
                "mda": mda,
                "mse": mse,
                "mae": mae,
                "bt_total_return": bt_total_return,
                "bt_sharpe": bt_sharpe,
                "bt_max_dd": bt_max_dd,
                "bt_win_rate": bt_win_rate,
                "bt_trades": bt_trades,
                "bt_strategy": bt_strategy,
            }
            result["runs"].append(run_data)
            if granularity not in result["by_granularity"]:
                result["by_granularity"][granularity] = []
            result["by_granularity"][granularity].append(run_data)

        return result
    except Exception as e:
        result["error"] = f"Error: {str(e)}\n{traceback.format_exc()}"
        return result


def build_analysis_summary(data):
    """Build HTML/text summary of best runs by each metric."""
    runs = data.get("runs", [])
    if not runs:
        return "No runs with both inference and backtest found."

    lines = []
    lines.append(f"**Total runs analyzed:** {len(runs)}")
    lines.append("")

    # Best by MDA (higher is better)
    valid_mda = [r for r in runs if r.get("mda") is not None]
    if valid_mda:
        best_mda = max(valid_mda, key=lambda x: x["mda"])
        lines.append(f"**Best MDA:** {best_mda['mda']:.4f} — Run: {best_mda['run_name'][:50]} (granularity: {best_mda['granularity']})")
    lines.append("")

    # Best by MSE (lower is better)
    valid_mse = [r for r in runs if r.get("mse") is not None]
    if valid_mse:
        best_mse = min(valid_mse, key=lambda x: x["mse"])
        lines.append(f"**Best MSE:** {best_mse['mse']:.6f} — Run: {best_mse['run_name'][:50]} (granularity: {best_mse['granularity']})")
    lines.append("")

    # Best by MAE (lower is better)
    valid_mae = [r for r in runs if r.get("mae") is not None]
    if valid_mae:
        best_mae = min(valid_mae, key=lambda x: x["mae"])
        lines.append(f"**Best MAE:** {best_mae['mae']:.6f} — Run: {best_mae['run_name'][:50]} (granularity: {best_mae['granularity']})")
    lines.append("")

    # Best by Backtest Sharpe (higher is better)
    valid_bt = [r for r in runs if r.get("bt_sharpe") is not None]
    if valid_bt:
        best_bt = max(valid_bt, key=lambda x: x["bt_sharpe"])
        lines.append(f"**Best Backtest Sharpe:** {best_bt['bt_sharpe']:.4f} — Run: {best_bt['run_name'][:50]} (granularity: {best_bt['granularity']}, strategy: {best_bt.get('bt_strategy', 'N/A')})")
    lines.append("")

    # Best by Total Return
    if valid_bt:
        best_ret = max(valid_bt, key=lambda x: x["bt_total_return"] or -999)
        lines.append(f"**Best Total Return (%):** {best_ret['bt_total_return']:.2f}% — Run: {best_ret['run_name'][:50]} (granularity: {best_ret['granularity']})")

    return "\n".join(lines)


def _filter_runs(runs, granularity_filter=None, data_type_filter=None):
    """Apply granularity and data_type filters to runs list."""
    if granularity_filter:
        runs = [r for r in runs if r.get("granularity") == granularity_filter]
    if data_type_filter:
        runs = [r for r in runs if r.get("data_type") == data_type_filter]
    return runs


def build_metrics_dataframe(data, granularity_filter=None, data_type_filter=None):
    """Build a DataFrame of runs with inference + backtest metrics for display."""
    runs = _filter_runs(data.get("runs", []), granularity_filter, data_type_filter)
    if not runs:
        return pd.DataFrame()

    rows = []
    for r in runs:
        row = {
            "Run": r.get("run_name", r["run_id"])[:40],
            "Granularity": r.get("granularity", ""),
            "Data Type": r.get("data_type", ""),
            "MDA": r.get("mda"),
            "MSE": r.get("mse"),
            "MAE": r.get("mae"),
            "BT Return (%)": r.get("bt_total_return"),
            "BT Sharpe": r.get("bt_sharpe"),
            "BT Max DD (%)": r.get("bt_max_dd"),
            "BT Win Rate (%)": r.get("bt_win_rate"),
            "BT Trades": r.get("bt_trades"),
            "BT Strategy": r.get("bt_strategy", ""),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_params_dataframe(data, granularity_filter=None, data_type_filter=None):
    """Build a DataFrame of run params for comparison."""
    runs = _filter_runs(data.get("runs", []), granularity_filter, data_type_filter)
    if not runs:
        return pd.DataFrame()

    all_keys = set()
    for r in runs:
        all_keys.update(r.get("params", {}).keys())
    ordered_keys = [k for k in DISPLAY_PARAMS if k in all_keys]
    if not ordered_keys:
        ordered_keys = sorted(all_keys)

    rows = []
    for r in runs:
        params = r.get("params", {})
        row = {
            "Run": r.get("run_name", r["run_id"])[:40],
            "Granularity": r.get("granularity", ""),
            "Data Type": r.get("data_type", ""),
        }
        for k in ordered_keys:
            row[k] = params.get(k, "")
        rows.append(row)
    return pd.DataFrame(rows)


def build_metrics_chart(data, metric_key, granularity_filter=None, data_type_filter=None, title=None):
    """Build a bar chart for a given metric (mda, mse, mae, bt_sharpe, bt_total_return).
    Only includes runs that have this metric (skips runs with None) so we don't show fake zeros.
    """
    runs = _filter_runs(data.get("runs", []), granularity_filter, data_type_filter)
    # Only include runs that have this metric - partial data is fine, don't show fake zeros
    runs = [r for r in runs if r.get(metric_key) is not None]
    if not runs:
        return None

    labels = [r.get("run_name", r["run_id"])[:30] for r in runs]
    values = [r.get(metric_key) for r in runs]

    higher_better = metric_key in ("mda", "bt_sharpe", "bt_total_return", "bt_win_rate")
    if higher_better:
        best_val = max(values) if values else 0
        colors = ["#22c55e" if v == best_val else "#3b82f6" for v in values]
    else:
        valid = [x for x in values if x is not None]
        best_val = min(valid) if valid else 0
        colors = ["#22c55e" if v == best_val else "#3b82f6" for v in values]

    fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=colors)])
    fig.update_layout(
        title=title or f"{metric_key} by Run",
        xaxis_title="Run",
        yaxis_title=metric_key,
        template="plotly_white",
        height=400,
        xaxis_tickangle=-45,
    )
    return fig


def build_multi_metric_chart(data, granularity_filter=None, data_type_filter=None):
    """Build a grouped bar chart for MDA, MSE, MAE.
    Includes runs that have at least one of the three metrics; missing values shown as 0
    (only for the comparison chart where we want to show partial data).
    """
    runs = _filter_runs(data.get("runs", []), granularity_filter, data_type_filter)
    # Include runs that have at least one inference metric
    runs = [r for r in runs if any(r.get(k) is not None for k in ("mda", "mse", "mae"))]
    if not runs:
        return None

    labels = [r.get("run_name", r["run_id"])[:25] for r in runs]
    mda_vals = [r.get("mda") if r.get("mda") is not None else 0 for r in runs]
    mse_vals = [r.get("mse") if r.get("mse") is not None else 0 for r in runs]
    mae_vals = [r.get("mae") if r.get("mae") is not None else 0 for r in runs]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="MDA", x=labels, y=mda_vals, marker_color="#22c55e"))
    fig.add_trace(go.Bar(name="MSE", x=labels, y=mse_vals, marker_color="#3b82f6"))
    fig.add_trace(go.Bar(name="MAE", x=labels, y=mae_vals, marker_color="#f59e0b"))
    fig.update_layout(
        title="Inference Metrics (MDA, MSE, MAE) by Run",
        xaxis_title="Run",
        yaxis_title="Value",
        barmode="group",
        template="plotly_white",
        height=450,
        xaxis_tickangle=-45,
    )
    return fig
