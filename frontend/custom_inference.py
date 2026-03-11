"""
Custom Inference tab for Time-LLM-Cryptex.
Run inference on an uploaded CSV file using a selected model from an experiment.
"""

from pathlib import Path
import gradio as gr
import pandas as pd

from utils import (
    list_experiment_runs_with_status,
    run_custom_inference,
    clean_csv_prices,
    compute_metrics_and_plot_from_csv,
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
    """Load runs for the experiment and return (choices, selected_run_id, status_msg)."""
    if not experiment_name or not str(experiment_name).strip():
        return [], None, "Enter an experiment name and click Load Runs."
    result = list_experiment_runs_with_status(experiment_name)
    if isinstance(result, str) and result.startswith("Error:"):
        return [], None, result
    choices = _runs_to_choices(result)
    if not choices:
        return [], None, f"No non-failed runs found in experiment '{experiment_name}'."
    return choices, result[0]["run_id"], f"Found {len(result)} run(s). Select one to run custom inference."


def guess_ohlcv_columns(cols):
    """Guess OHLCV column names from CSV columns. Returns dict with open, high, low, close, volume."""
    cols_lower = {c: c.lower() for c in cols}
    result = {}
    # open: exact 'open', or contains 'open'
    result["open"] = next((c for c in cols if cols_lower[c] == "open"), None) or next((c for c in cols if "open" in cols_lower[c] and "close" not in cols_lower[c]), None)
    # high
    result["high"] = next((c for c in cols if cols_lower[c] == "high"), None) or next((c for c in cols if "high" in cols_lower[c]), None)
    # low
    result["low"] = next((c for c in cols if cols_lower[c] == "low"), None) or next((c for c in cols if "low" in cols_lower[c]), None)
    # close: exact 'close', or 'closing', 'close_price', 'price'
    result["close"] = next((c for c in cols if cols_lower[c] == "close"), None) or next((c for c in cols if cols_lower[c] in ("closing", "close_price", "price")), None) or next((c for c in cols if "close" in cols_lower[c]), None)
    # volume
    result["volume"] = next((c for c in cols if cols_lower[c] == "volume"), None) or next((c for c in cols if "vol" in cols_lower[c] and "volatility" not in cols_lower[c]), None)
    return result


def on_csv_upload(csv_file):
    """When a CSV is uploaded, return column choices for timestamp, target, OHLCV dropdowns and preview."""
    empty = gr.update(choices=[], value=None)
    if csv_file is None:
        return empty, empty, None, empty, empty, empty, empty, empty
    try:
        path = csv_file if isinstance(csv_file, str) else getattr(csv_file, "name", csv_file)
        df = pd.read_csv(path)
        cols = list(df.columns)
        if not cols:
            return empty, empty, None, empty, empty, empty, empty, empty
        # Timestamp
        ts_default = next((c for c in ["timestamp", "date", "datetime", "time"] if c in cols), cols[0])
        ts_update = gr.update(choices=cols, value=ts_default)
        # Target
        target_default = "close" if "close" in cols else cols[0]
        target_update = gr.update(choices=cols, value=target_default)
        # OHLCV - guess from column names (choices include empty for "not found")
        guessed = guess_ohlcv_columns(cols)
        choices_with_none = [("— Not in file —", "")] + [(c, c) for c in cols]
        def ohlcv_update(col_name):
            v = guessed.get(col_name) or ""
            return gr.update(choices=choices_with_none, value=v)
        open_up = ohlcv_update("open")
        high_up = ohlcv_update("high")
        low_up = ohlcv_update("low")
        close_up = ohlcv_update("close")
        volume_up = ohlcv_update("volume")
        preview = df.head(5)
        return ts_update, target_update, preview, open_up, high_up, low_up, close_up, volume_up
    except Exception:
        return empty, empty, None, empty, empty, empty, empty, empty


def build_custom_inference_tab():
    """Build and return the Custom Inference tab UI."""
    with gr.TabItem("Custom Inference", id="custom_inference"):
        gr.Markdown("## Custom Inference")
        gr.Markdown(
            "Upload a CSV file and run inference using a trained model. "
            "Select the experiment and run (model), then upload your data and choose the target column. "
            "Results are saved as **custom_inference.csv**."
        )

        gr.Markdown("### Model Selection")
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
            label="Model (Run)",
            choices=[],
            value=None,
            allow_custom_value=False,
            info="Select a trained model to run inference",
        )

        gr.Markdown("---")
        gr.Markdown("### Data Upload")
        gr.Markdown("CSV must have OHLCV columns (`open`, `high`, `low`, `close`, `volume`). Select timestamp and target columns below. Use **Clean Prices** if values have $, commas, or other non-numeric characters.")
        csv_upload = gr.File(
            label="Upload CSV",
            file_types=[".csv"],
        )
        with gr.Row():
            timestamp_dropdown = gr.Dropdown(
                label="Timestamp Column",
                choices=[],
                value=None,
                allow_custom_value=False,
                info="Column containing dates/times (will be converted to Unix seconds)",
            )
            timestamp_format = gr.Dropdown(
                label="Timestamp Format",
                choices=[
                    ("Auto-detect", ""),
                    ("YYYY-MM-DD", "%Y-%m-%d"),
                    ("YYYY-MM-DD HH:MM:SS", "%Y-%m-%d %H:%M:%S"),
                    ("YYYY-MM-DDTHH:MM:SS (ISO)", "%Y-%m-%dT%H:%M:%S"),
                    ("MM/DD/YYYY", "%m/%d/%Y"),
                    ("DD/MM/YYYY", "%d/%m/%Y"),
                    ("Unix seconds (numeric)", "unix"),
                ],
                value="",
                allow_custom_value=True,
                info="Select format or type custom. Leave empty for auto-detect.",
            )
        with gr.Row():
            target_dropdown = gr.Dropdown(
                label="Target Column",
                choices=[],
                value=None,
                allow_custom_value=False,
                info="Column to predict (select after uploading CSV)",
            )
        gr.Markdown("**OHLCV column mapping** (auto-guessed from column names; override if needed)")
        with gr.Row():
            open_col = gr.Dropdown(label="Open", choices=[], value=None, allow_custom_value=False)
            high_col = gr.Dropdown(label="High", choices=[], value=None, allow_custom_value=False)
            low_col = gr.Dropdown(label="Low", choices=[], value=None, allow_custom_value=False)
            close_col = gr.Dropdown(label="Close", choices=[], value=None, allow_custom_value=False)
            volume_col = gr.Dropdown(label="Volume", choices=[], value=None, allow_custom_value=False)

        data_preview = gr.Dataframe(
            label="Data Preview (first 5 rows)",
            interactive=False,
        )

        cleaned_path_state = gr.State(value=None)
        clean_btn = gr.Button("Clean Prices", variant="secondary")
        clean_status = gr.Textbox(label="Clean Status", interactive=False, lines=1)

        run_btn = gr.Button("Run Custom Inference", variant="primary")
        output_status = gr.Textbox(label="Output", interactive=False, lines=8)

        gr.Markdown("---")
        gr.Markdown("### Inference Results")
        with gr.Row():
            pred_horizon = gr.Number(
                label="Prediction Horizon",
                value=1,
                minimum=1,
                step=1,
                info="Steps ahead for metrics and plot (1 = 1 step)",
            )
            refresh_btn = gr.Button("Refresh Results", variant="secondary")
        inference_plot = gr.Plot(label="Inference Results (Candlestick + Prediction)")
        with gr.Row():
            inf_mae = gr.Number(label="MAE (Mean Absolute Error)", interactive=False)
            inf_mse = gr.Number(label="MSE (Mean Squared Error)", interactive=False)
            inf_mda = gr.Number(label="MDA (Mean Directional Accuracy)", interactive=False)

        # Load runs when button clicked
        def load_runs(exp_name):
            choices, first_id, msg = load_runs_ui(exp_name)
            return gr.update(choices=choices, value=first_id), msg

        load_btn.click(
            fn=load_runs,
            inputs=[exp_name_input],
            outputs=[runs_dropdown, runs_status],
        )

        # Populate timestamp/target/OHLCV dropdowns and data preview when CSV is uploaded
        def on_upload(csv_file):
            result = on_csv_upload(csv_file)
            return (*result, None)  # clear cleaned_path when new file uploaded

        csv_upload.change(
            fn=on_upload,
            inputs=[csv_upload],
            outputs=[timestamp_dropdown, target_dropdown, data_preview, open_col, high_col, low_col, close_col, volume_col, cleaned_path_state],
        )

        # Clean Prices button: store cleaned path in State (don't output to File - Gradio rejects project paths)
        def on_clean(csv_file, ts_col):
            path, preview, status = clean_csv_prices(csv_file, timestamp_column=ts_col)
            if path:
                return path, preview, status
            return None, gr.update(), status

        clean_btn.click(
            fn=on_clean,
            inputs=[csv_upload, timestamp_dropdown],
            outputs=[cleaned_path_state, data_preview, clean_status],
        )

        # Run: use cleaned file if available, else original upload
        def run_with_cleaned(exp_name, run_id, csv_file, cleaned_path, ts_col, ts_fmt, target_col, o_col, h_col, l_col, c_col, v_col, horizon):
            file_to_use = cleaned_path if (cleaned_path and Path(cleaned_path).exists()) else csv_file
            ohlcv_map = {"open": o_col, "high": h_col, "low": l_col, "close": c_col, "volume": v_col}
            return run_custom_inference(exp_name, run_id, file_to_use, ts_col, ts_fmt, target_col, ohlcv_columns=ohlcv_map, pred_horizon=horizon)

        run_btn.click(
            fn=run_with_cleaned,
            inputs=[exp_name_input, runs_dropdown, csv_upload, cleaned_path_state, timestamp_dropdown, timestamp_format, target_dropdown, open_col, high_col, low_col, close_col, volume_col, pred_horizon],
            outputs=[output_status, inference_plot, inf_mae, inf_mse, inf_mda],
        )

        # Refresh: load metrics/plot from custom_inference.csv with selected horizon
        def on_refresh(horizon):
            project_root = Path(__file__).parent.parent
            csv_path = project_root / "custom_inference.csv"
            status, fig, mae, mse, mda = compute_metrics_and_plot_from_csv(str(csv_path), pred_horizon=horizon)
            msg = f"Loaded from custom_inference.csv\n{status}" if fig else status
            return msg, fig, mae, mse, mda

        refresh_btn.click(
            fn=on_refresh,
            inputs=[pred_horizon],
            outputs=[output_status, inference_plot, inf_mae, inf_mse, inf_mda],
        )
