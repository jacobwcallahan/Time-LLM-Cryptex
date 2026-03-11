"""
Experiment Runs tab for Time-LLM-Cryptex.
Lists all runs in an experiment by ID with inference/backtest status.
Click a run to view inference results and metrics.
"""

import gradio as gr

from utils import (
    list_experiment_runs_with_status,
    check_and_plot_mlflow_inference,
    fetch_summary_table_from_mlflow,
    run_backtest,
    run_simple_inference,
)


def _runs_to_choices(runs):
    """Convert runs list to Gradio dropdown choices: (display_label, run_id)."""
    if not isinstance(runs, list):
        return []
    choices = []
    for r in runs:
        inf_mark = "✓" if r.get("has_inference") else "✗"
        bt_mark = "✓" if r.get("has_backtest") else "✗"
        name = r.get("run_name") or r["run_id"]
        if len(str(name)) > 40:
            name = str(name)[:37] + "..."
        label = f"{name} | Inf: {inf_mark} | Backtest: {bt_mark}"
        choices.append((label, r["run_id"]))
    return choices


def load_runs_ui(experiment_name):
    """
    Load runs for the experiment and return (choices, selected_run_id, status_msg).
    Used to populate the runs dropdown.
    """
    if not experiment_name or not str(experiment_name).strip():
        return [], None, "Enter an experiment name and click Load Runs."
    result = list_experiment_runs_with_status(experiment_name)
    if isinstance(result, str) and result.startswith("Error:"):
        return [], None, result
    choices = _runs_to_choices(result)
    if not choices:
        return [], None, f"No non-failed runs found in experiment '{experiment_name}'."
    return choices, result[0]["run_id"], f"Found {len(result)} run(s) (non-failed only). Select one to view inference and metrics."


def on_run_selected(experiment_name, run_id, pred_horizon):
    """
    When a run is selected, load and display inference plot + metrics + backtest summary.
    """
    if not run_id:
        return "Select a run to view details.", None, None, None, None, None, "No run selected."
    status, fig, mae, mse, mda = check_and_plot_mlflow_inference(experiment_name, run_id, pred_horizon)
    summary_df, err = fetch_summary_table_from_mlflow(experiment_name, run_id)
    if err:
        backtest_status = f"Error loading backtest: {err}"
    elif summary_df is not None and not summary_df.empty:
        backtest_status = "Backtest results loaded from MLflow."
    else:
        backtest_status = "No backtest results. Click **Run Quick Backtest** to run one and save to MLflow."
    return status, fig, mae, mse, mda, summary_df, backtest_status


def run_quick_backtest(experiment_name, run_id):
    """
    Run a quick backtest (all strategies, $10k) and save summary_table to MLflow.
    Returns (summary_df, status_msg).
    """
    if not run_id:
        return None, "Error: No run selected."
    if not experiment_name:
        return None, "Error: Experiment name required."
    fig, summary_str, *_, summary_df = run_backtest(
        experiment_name, run_id,
        strategy=None,
        initial_capital=10000,
        start_date=None,
        end_date=None,
        threshold=None,
        log_to_mlflow=True,
    )
    if summary_df is not None and not summary_df.empty:
        return summary_df, "Backtest complete. Summary saved to MLflow."
    err_msg = summary_str if summary_str and summary_str.startswith("Error:") else "Backtest failed."
    return None, err_msg


def build_experiment_runs_tab():
    """Build and return the Experiment Runs tab UI."""
    with gr.TabItem("Experiment Runs", id="experiment_runs"):
        gr.Markdown("## Experiment Runs")
        gr.Markdown(
            "Enter an experiment name to list all non-failed runs. Each run shows whether inference and backtesting "
            "were run (based on MLflow artifacts). Click a run to view inference results and metrics."
        )

        with gr.Row():
            exp_name_input = gr.Textbox(
                label="Experiment Name",
                placeholder="e.g. 03.09.25_experiment",
                info="MLflow experiment name",
                scale=2,
            )
            load_btn = gr.Button("Load Runs", variant="primary", scale=1)

        runs_status = gr.Textbox(label="Status", interactive=False, lines=2)
        runs_dropdown = gr.Dropdown(
            label="Runs (ID | Inference | Backtest)",
            choices=[],
            value=None,
            allow_custom_value=False,
            info="Select a run to view inference graph and metrics",
        )

        gr.Markdown("---")
        gr.Markdown("### Run Details")
        gr.Markdown("Inference results and metrics for the selected run.")

        gr.Markdown("**Simple Inference**: Run inference from the day after training end to the end of the dataset. Results are logged to MLflow.")
        simple_inf_btn = gr.Button("Simple Inference", variant="secondary")
        simple_inf_status = gr.Textbox(label="Simple Inference Output", interactive=False, lines=20)

        with gr.Row():
            pred_horizon = gr.Number(
                label="Prediction Horizon",
                value=1,
                minimum=1,
                step=1,
                info="Steps ahead for plot and metrics (1 = 1 step, 2 = 2 steps)",
            )

        detail_status = gr.Textbox(label="Detail Status", interactive=False, lines=3)
        inference_plot = gr.Plot(
            label="Inference Results (Candlestick + Prediction)",
            scale=2,
        )
        with gr.Row():
            inf_mae = gr.Number(label="MAE (Mean Absolute Error)", interactive=False)
            inf_mse = gr.Number(label="MSE (Mean Squared Error)", interactive=False)
            inf_mda = gr.Number(label="MDA (Mean Directional Accuracy)", interactive=False)

        gr.Markdown("### Backtest Results")
        gr.Markdown("Summary table from MLflow (if backtest was run). Use **Run Quick Backtest** to run one and save.")
        quick_bt_btn = gr.Button("Run Quick Backtest", variant="secondary")
        backtest_status = gr.Textbox(label="Backtest Status", interactive=False, lines=1)
        backtest_summary = gr.Dataframe(
            label="Backtest Summary Table",
            interactive=False,
            wrap=True,
        )

        def load_runs(exp_name):
            choices, first_id, msg = load_runs_ui(exp_name)
            dropdown_update = gr.update(choices=choices, value=first_id)
            if first_id:
                detail_out = on_run_selected(exp_name, first_id, 1)
                return dropdown_update, msg, *detail_out
            return dropdown_update, msg, "Select a run to view details.", None, None, None, None, None, "No run selected."

        load_btn.click(
            fn=load_runs,
            inputs=[exp_name_input],
            outputs=[runs_dropdown, runs_status, detail_status, inference_plot, inf_mae, inf_mse, inf_mda, backtest_summary, backtest_status],
        )

        def on_select(exp_name, run_id, horizon):
            return on_run_selected(exp_name, run_id, horizon)

        runs_dropdown.change(
            fn=on_select,
            inputs=[exp_name_input, runs_dropdown, pred_horizon],
            outputs=[detail_status, inference_plot, inf_mae, inf_mse, inf_mda, backtest_summary, backtest_status],
        )
        pred_horizon.change(
            fn=on_select,
            inputs=[exp_name_input, runs_dropdown, pred_horizon],
            outputs=[detail_status, inference_plot, inf_mae, inf_mse, inf_mda, backtest_summary, backtest_status],
        )

        def on_quick_backtest(exp_name, run_id):
            summary_df, status = run_quick_backtest(exp_name, run_id)
            return summary_df, status

        quick_bt_btn.click(
            fn=on_quick_backtest,
            inputs=[exp_name_input, runs_dropdown],
            outputs=[backtest_summary, backtest_status],
        )

        simple_inf_btn.click(
            fn=run_simple_inference,
            inputs=[exp_name_input, runs_dropdown],
            outputs=[simple_inf_status],
        )
