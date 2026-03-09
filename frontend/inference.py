"""
Inference tab for Time-LLM-Cryptex.
Provides UI for running inference and visualizing MLflow inference data.
"""

import gradio as gr

from helper_fcns import run_inference_handler, check_and_plot_mlflow_inference


def build_inference_tab():
    """Build and return the Inference tab UI."""
    with gr.TabItem("Inference", id="inference"):
        gr.Markdown("## Run Inference")
        gr.Markdown("Load a trained model and generate predictions on new data.")

        with gr.Row():
            with gr.Column():
                inf_experiment_name = gr.Textbox(
                    label="Experiment Name",
                    placeholder="e.g. my_experiment",
                    info="MLflow experiment name"
                )
                inf_model_name = gr.Textbox(
                    label="Model Name (Run ID)",
                    placeholder="e.g. abc123def456",
                    info="MLflow Run ID of the trained model"
                )
                inf_custom_dataset_path = gr.Textbox(
                    label="Custom Data Path (Optional)",
                    placeholder="path/to/data.csv",
                    info="Path to the custom dataset"
                )
                inf_granularity = gr.Dropdown(
                    label="Granularity",
                    choices=["daily", "hourly", "minute", "weekly"],
                    value="daily",
                    info="Data granularity"
                )

            with gr.Column():
                inf_start_date = gr.DateTime(
                    label="Start Date",
                    value="",
                    info="Inference start date (YYYY-MM-DD)",
                    include_time=False
                )
                inf_end_date = gr.DateTime(
                    label="End Date",
                    value="",
                    info="Inference end date (YYYY-MM-DD)",
                    include_time=False
                )
                inf_aggregate = gr.Number(
                    label="Aggregate",
                    value=1,
                    info="Aggregation factor"
                )

        run_inference_btn = gr.Button("Run Inference", variant="primary")
        inference_output = gr.Textbox(label="Inference Output", lines=25, interactive=False)

        run_inference_btn.click(
            fn=run_inference_handler,
            inputs=[inf_model_name, inf_experiment_name, inf_custom_dataset_path, inf_granularity, inf_aggregate, inf_start_date, inf_end_date],
            outputs=inference_output,
        )

        gr.Markdown("---")
        gr.Markdown("### Load Inference from MLflow")
        gr.Markdown("Visualize inference data and metrics for a specific prediction horizon. The plot and metrics both use the selected horizon.")
        with gr.Row():
            mlflow_experiment_name = gr.Textbox(
                label="Experiment Name",
                placeholder="e.g. my_experiment",
                info="MLflow experiment name"
            )
            mlflow_run_id = gr.Textbox(
                label="MLflow Run ID",
                placeholder="e.g. abc123def456",
                info="Enter the MLflow Run ID to check for inference data"
            )
            mlflow_pred_horizon = gr.Number(
                label="Prediction Horizon",
                value=1,
                minimum=1,
                step=1,
                info="Steps ahead (1 = 1 step, 2 = 2 steps). Plot and metrics use this."
            )
        check_mlflow_btn = gr.Button("Check & Plot Inference", variant="primary")

        mlflow_status = gr.Textbox(label="Status", interactive=False, lines=3)
        mlflow_inference_plot = gr.Plot(
            label="Inference Data Visualization (Candlestick + Prediction)",
            scale=2,  # take more space relative to adjacent components
        )

        gr.Markdown("**Metrics for selected prediction horizon** (from mae_metrics.csv, mse_metrics.csv, mda_metrics.csv)")
        with gr.Row():
            inf_mae = gr.Number(label="MAE (Mean Absolute Error)", interactive=False)
            inf_mse = gr.Number(label="MSE (Mean Squared Error)", interactive=False)
            inf_mda = gr.Number(label="MDA (Mean Directional Accuracy)", interactive=False)

        check_mlflow_btn.click(
            fn=check_and_plot_mlflow_inference,
            inputs=[mlflow_experiment_name, mlflow_run_id, mlflow_pred_horizon],
            outputs=[mlflow_status, mlflow_inference_plot, inf_mae, inf_mse, inf_mda],
        )
