"""
Experiment Analysis tab for Time-LLM-Cryptex.
Analyzes all runs with both inference and backtesting, grouped by granularity.
Shows MDA, MSE, MAE, backtest metrics, and hyperparameter comparison.
"""

import gradio as gr

from frontend_utils.experiment_analysis_utils import (
    fetch_experiment_analysis_data,
    build_analysis_summary,
    build_metrics_dataframe,
    build_params_dataframe,
    build_metrics_chart,
    build_multi_metric_chart,
)


def on_analyze(experiment_name, pred_horizon, granularity_filter, data_type_filter):
    """Load and analyze experiment runs. Returns (status, summary, metrics_df, params_df, charts)."""
    if not experiment_name or not str(experiment_name).strip():
        return (
            "Enter an experiment name and click Analyze.",
            "",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    pred_horizon = int(pred_horizon) if pred_horizon else 1
    granularity_filter = str(granularity_filter).strip() or None
    if granularity_filter == "All":
        granularity_filter = None
    data_type_filter = str(data_type_filter).strip() or None
    if data_type_filter == "All":
        data_type_filter = None

    data = fetch_experiment_analysis_data(experiment_name, pred_horizon=pred_horizon)

    if data.get("error"):
        return (
            data["error"],
            "",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    runs = data.get("runs", [])
    if not runs:
        return (
            f"No runs with both inference and backtest found in experiment '{experiment_name}'.",
            "",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    status = f"Found {len(runs)} run(s) with inference + backtest. Prediction horizon: {pred_horizon}."
    summary = build_analysis_summary(data)
    metrics_df = build_metrics_dataframe(data, granularity_filter, data_type_filter)
    params_df = build_params_dataframe(data, granularity_filter, data_type_filter)
    mda_chart = build_metrics_chart(data, "mda", granularity_filter, data_type_filter, title="MDA by Run")
    mse_chart = build_metrics_chart(data, "mse", granularity_filter, data_type_filter, title="MSE by Run")
    mae_chart = build_metrics_chart(data, "mae", granularity_filter, data_type_filter, title="MAE by Run")
    multi_chart = build_multi_metric_chart(data, granularity_filter, data_type_filter)
    bt_sharpe_chart = build_metrics_chart(
        data, "bt_sharpe", granularity_filter, data_type_filter, title="Backtest Sharpe Ratio by Run"
    )

    return (
        status,
        summary,
        metrics_df,
        params_df,
        mda_chart,
        mse_chart,
        mae_chart,
        multi_chart,
        bt_sharpe_chart,
    )


def build_experiment_analysis_tab():
    """Build and return the Experiment Analysis tab UI."""
    with gr.TabItem("Experiment Analysis", id="experiment_analysis"):
        gr.Markdown("## Experiment Analysis")
        gr.Markdown(
            "Analyze all runs in an experiment that have **both** inference and backtesting completed. "
            "Compare MDA, MSE, MAE (inference metrics) and backtest performance. Results are grouped by granularity."
        )
        gr.Markdown("*It may take a moment to obtain all the data when you click Analyze.*")

        with gr.Row():
            exp_name_input = gr.Textbox(
                label="Experiment Name",
                placeholder="e.g. 03.09.25_experiment",
                info="MLflow experiment name",
                scale=2,
            )
            analyze_btn = gr.Button("Analyze", variant="primary", scale=1)

        with gr.Row():
            pred_horizon = gr.Number(
                label="Prediction Horizon",
                value=1,
                minimum=1,
                step=1,
                info="Steps ahead for inference metrics (MDA, MSE, MAE)",
            )
            granularity_filter = gr.Dropdown(
                label="Filter by Granularity",
                choices=["All"],
                value="All",
                allow_custom_value=True,
                info="Show only runs with this granularity",
            )
            data_type_filter = gr.Dropdown(
                label="Filter by Data Type",
                choices=["All", "returns", "ohlcv", "volatility"],
                value="All",
                allow_custom_value=True,
                info="Returns = trained on returns; OHLCV = trained on close price",
            )

        status_txt = gr.Textbox(label="Status", interactive=False, lines=2)

        gr.Markdown("---")
        gr.Markdown("### Best Runs Summary")
        summary_txt = gr.Markdown(value="")

        gr.Markdown("---")
        gr.Markdown("### Inference Metrics (MDA, MSE, MAE)")
        gr.Markdown("Lower is better for MSE and MAE; higher is better for MDA.")
        with gr.Row():
            mda_chart = gr.Plot(label="MDA by Run")
            mse_chart = gr.Plot(label="MSE by Run")
        with gr.Row():
            mae_chart = gr.Plot(label="MAE by Run")
            multi_chart = gr.Plot(label="MDA, MSE, MAE Comparison")

        gr.Markdown("### Backtest Metrics")
        bt_sharpe_chart = gr.Plot(label="Backtest Sharpe Ratio by Run")

        gr.Markdown("---")
        gr.Markdown("### Metrics Table")
        metrics_table = gr.Dataframe(
            label="Runs with Inference + Backtest Metrics",
            interactive=False,
            wrap=True,
        )

        gr.Markdown("### Hyperparameters by Run")
        params_table = gr.Dataframe(
            label="Run Parameters (granularity, features, seq_len, pred_len, etc.)",
            interactive=False,
            wrap=True,
        )

        def run_analyze(exp_name, horizon, gran_filter, data_filter):
            result = on_analyze(exp_name, horizon, gran_filter, data_filter)
            # Update granularity and data_type choices from data
            data = fetch_experiment_analysis_data(exp_name, pred_horizon=horizon)
            granules = sorted(set(r.get("granularity", "unknown") for r in data.get("runs", [])))
            data_types = sorted(set(r.get("data_type", "ohlcv") for r in data.get("runs", [])))
            gran_choices = ["All"] + granules
            data_choices = ["All"] + data_types
            return (
                result[0],
                result[1],
                result[2],
                result[3],
                result[4],
                result[5],
                result[6],
                result[7],
                result[8],
                gr.update(choices=gran_choices, value=gran_filter or "All"),
                gr.update(choices=data_choices, value=data_filter or "All"),
            )

        analyze_btn.click(
            fn=run_analyze,
            inputs=[exp_name_input, pred_horizon, granularity_filter, data_type_filter],
            outputs=[
                status_txt,
                summary_txt,
                metrics_table,
                params_table,
                mda_chart,
                mse_chart,
                mae_chart,
                multi_chart,
                bt_sharpe_chart,
                granularity_filter,
                data_type_filter,
            ],
        )

        # When granularity or data_type filter changes, re-render tables and charts
        def on_filter_change(exp_name, horizon, gran_filter, data_filter):
            if not exp_name or not str(exp_name).strip():
                return (
                    "Enter an experiment name and click Analyze first.",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            horizon = int(horizon) if horizon else 1
            data = fetch_experiment_analysis_data(exp_name, pred_horizon=horizon)
            if data.get("error") or not data.get("runs"):
                return (
                    data.get("error", "No data"),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            gf = str(gran_filter).strip() or None
            if gf == "All":
                gf = None
            df = str(data_filter).strip() or None
            if df == "All":
                df = None
            summary = build_analysis_summary(data)
            metrics_df = build_metrics_dataframe(data, gf, df)
            params_df = build_params_dataframe(data, gf, df)
            mda_chart = build_metrics_chart(data, "mda", gf, df, title="MDA by Run")
            mse_chart = build_metrics_chart(data, "mse", gf, df, title="MSE by Run")
            mae_chart = build_metrics_chart(data, "mae", gf, df, title="MAE by Run")
            multi_chart = build_multi_metric_chart(data, gf, df)
            bt_sharpe_chart = build_metrics_chart(
                data, "bt_sharpe", gf, df, title="Backtest Sharpe Ratio by Run"
            )
            return (
                summary,
                metrics_df,
                params_df,
                mda_chart,
                mse_chart,
                mae_chart,
                multi_chart,
                bt_sharpe_chart,
            )

        def on_gran_change(exp_name, horizon, gran_filter, data_filter):
            return on_filter_change(exp_name, horizon, gran_filter, data_filter)

        def on_data_type_change(exp_name, horizon, gran_filter, data_filter):
            return on_filter_change(exp_name, horizon, gran_filter, data_filter)

        granularity_filter.change(
            fn=on_gran_change,
            inputs=[exp_name_input, pred_horizon, granularity_filter, data_type_filter],
            outputs=[
                summary_txt,
                metrics_table,
                params_table,
                mda_chart,
                mse_chart,
                mae_chart,
                multi_chart,
                bt_sharpe_chart,
            ],
        )
        data_type_filter.change(
            fn=on_data_type_change,
            inputs=[exp_name_input, pred_horizon, granularity_filter, data_type_filter],
            outputs=[
                summary_txt,
                metrics_table,
                params_table,
                mda_chart,
                mse_chart,
                mae_chart,
                multi_chart,
                bt_sharpe_chart,
            ],
        )

        # When prediction horizon changes, re-fetch data (MDA/MSE/MAE depend on horizon)
        def on_horizon_change(exp_name, horizon, gran_filter, data_filter):
            result = on_filter_change(exp_name, horizon, gran_filter, data_filter)
            # Update status with new horizon when we have valid data
            if exp_name and str(exp_name).strip() and result[1] is not None:
                h = int(horizon) if horizon else 1
                data = fetch_experiment_analysis_data(exp_name, pred_horizon=h)
                if not data.get("error") and data.get("runs"):
                    status = f"Found {len(data['runs'])} run(s) with inference + backtest. Prediction horizon: {h}."
                    return (status, *result)
            # Early return: result[0] is error msg for summary; use for status too
            return (result[0] if isinstance(result[0], str) else "", *result)

        pred_horizon.change(
            fn=on_horizon_change,
            inputs=[exp_name_input, pred_horizon, granularity_filter, data_type_filter],
            outputs=[
                status_txt,
                summary_txt,
                metrics_table,
                params_table,
                mda_chart,
                mse_chart,
                mae_chart,
                multi_chart,
                bt_sharpe_chart,
            ],
        )
