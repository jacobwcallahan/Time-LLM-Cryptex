"""
Single Run Training tab for Time-LLM-Cryptex.
All options in one tab: HPO args (no inference/backtest), single model params, custom prompt.
Trains one model with fixed hyperparameters.
"""

import gradio as gr
import subprocess
import os
import traceback
import yaml
from pathlib import Path
from datetime import datetime

from frontend_utils import start_before_end, end_after_start

# Fallback prompt when custom_prompt.txt doesn't exist
DEFAULT_PROMPT = """The Binance Bitcoin Hourly Returns (BTC) dataset captures granular financial data from the Binance.us cryptocurrency exchange. It spans nearly four months, from July 2024 to December 2024, with hourly-level resolution. Each record contains updates for returns of hourly closing prices and traded volume in USD. Timestamps are stored in Unix time format. Inactive periods (with no trading activity) are represented with NaN values, while missing timestamps may reflect exchange/API downtime or data collection limitations. The dataset has been carefully deduplicated and validated, and is updated nightly to ensure consistency and completeness."""

DEFAULT_PROMPT_FILENAME = "custom_prompt.txt"
DEFAULT_YAML_FILENAME = "custom_single_config.yaml"


def _load_prompt_from_file(filename: str) -> str:
    """Load prompt from prompt_bank if file exists, else return DEFAULT_PROMPT."""
    project_root = Path(__file__).parent.parent
    prompt_path = project_root / "dataset" / "prompt_bank" / filename
    if prompt_path.exists():
        try:
            return prompt_path.read_text()
        except Exception:
            pass
    return DEFAULT_PROMPT


def _to_date_str(val):
    """Convert Gradio DateTime value (datetime, timestamp, or str) to YYYY-MM-DD."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return datetime.fromtimestamp(val).strftime("%Y-%m-%d")
    if hasattr(val, "strftime"):
        return val.strftime("%Y-%m-%d")
    if isinstance(val, str) and len(val) >= 10:
        return val[:10]
    return str(val) if val else None


def save_prompt(prompt, filename=DEFAULT_PROMPT_FILENAME):
    """Save the prompt to the prompt bank."""
    project_root = Path(__file__).parent.parent
    prompt_path = project_root / "dataset" / "prompt_bank" / filename
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prompt_path, "w") as f:
        f.write(prompt)
    return f"Prompt saved to {prompt_path}"


def generate_single_yaml_config(
    features, seq_len, pred_len, num_tokens, loss, lradj,
    n_heads, d_ff, batch_size, patch_len, stride, epochs,
    llm_layers, d_model, dropout, pct_start, learning_rate,
):
    """Generate YAML config from single values (single-element lists for Optuna format)."""
    config = {
        "categorical": {
            "features": [features],
            "seq_len": [int(seq_len)],
            "pred_len": [int(pred_len)],
            "num_tokens": [int(num_tokens)],
            "loss": [loss],
            "lradj": [lradj],
            "n_heads": [int(n_heads)],
            "d_ff": [int(d_ff)],
            "batch_size": [int(batch_size)],
            "patch_len": [int(patch_len)],
            "stride": [int(stride)],
            "epochs": [int(epochs)],
        },
        "int": {
            "llm_layers": {"low": int(llm_layers), "high": int(llm_layers)},
            "d_model": {"low": int(d_model), "high": int(d_model), "step": 1},
        },
        "float": {
            "dropout": {"low": float(dropout), "high": float(dropout), "step": float(dropout)},
            "pct_start": {"low": float(pct_start), "high": float(pct_start), "step": float(pct_start)},
            "learning_rate": {"low": float(learning_rate), "high": float(learning_rate), "log": False},
        },
    }
    return yaml.dump(config, default_flow_style=False, sort_keys=False)


def save_single_yaml_config(yaml_content, filename=DEFAULT_YAML_FILENAME):
    """Save YAML config to config/yaml_params/."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "yaml_params" / filename
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        f.write(yaml_content)
    return f"Configuration saved to {config_path}"


def build_single_run_command(
    gpu, study_name, granularity, start_date, end_date,
    data_path, returns, volatility,
    experiment_name, aggregate, no_inf_aggregate, log_all_metrics,
    model_id, prompt_filename,
    features, seq_len, pred_len, num_tokens, loss, lradj,
    n_heads, d_ff, batch_size, patch_len, stride, epochs,
    llm_layers, d_model, dropout, pct_start, learning_rate,
):
    """Build the command string for single-run training."""
    cmd_parts = ["python3", "run_single_train.py"]

    if gpu:
        cmd_parts.extend(["--gpu", str(gpu)])
    if study_name:
        cmd_parts.extend(["--study_name", study_name])
    if granularity:
        cmd_parts.extend(["--granularity", granularity])
    if start_date:
        cmd_parts.extend(["--start", str(_to_date_str(start_date))])
    if end_date:
        cmd_parts.extend(["--end", str(_to_date_str(end_date))])
    if data_path:
        cmd_parts.extend(["--data_path", data_path])
    if returns:
        cmd_parts.append("--returns")
    if volatility:
        cmd_parts.append("--volatility")
    if experiment_name:
        cmd_parts.extend(["--experiment_name", experiment_name])
    if aggregate:
        cmd_parts.extend(["--aggregate", str(int(aggregate))])
    if no_inf_aggregate:
        cmd_parts.append("--no_inf_aggregate")
    if log_all_metrics:
        cmd_parts.append("--log_all_metrics")
    if model_id:
        cmd_parts.extend(["--model_id", model_id])
    # Prompt file: strip .txt if present
    pf = (prompt_filename or DEFAULT_PROMPT_FILENAME).strip()
    if pf.endswith(".txt"):
        pf = pf[:-4]
    if pf:
        cmd_parts.extend(["--prompt_file", pf])

    # Model params
    cmd_parts.extend(["--features", str(features)])
    cmd_parts.extend(["--seq_len", str(int(seq_len))])
    cmd_parts.extend(["--pred_len", str(int(pred_len))])
    cmd_parts.extend(["--num_tokens", str(int(num_tokens))])
    cmd_parts.extend(["--loss", str(loss)])
    cmd_parts.extend(["--lradj", str(lradj)])
    cmd_parts.extend(["--n_heads", str(int(n_heads))])
    cmd_parts.extend(["--d_ff", str(int(d_ff))])
    cmd_parts.extend(["--batch_size", str(int(batch_size))])
    cmd_parts.extend(["--patch_len", str(int(patch_len))])
    cmd_parts.extend(["--stride", str(int(stride))])
    cmd_parts.extend(["--epochs", str(int(epochs))])
    cmd_parts.extend(["--llm_layers", str(int(llm_layers))])
    cmd_parts.extend(["--d_model", str(int(d_model))])
    cmd_parts.extend(["--dropout", str(float(dropout))])
    cmd_parts.extend(["--pct_start", str(float(pct_start))])
    cmd_parts.extend(["--learning_rate", str(float(learning_rate))])

    return " ".join(cmd_parts)


def run_single_train(command):
    """Execute the single-run training command with streaming output."""
    if not command or not str(command).strip():
        yield "Error: No command to run. Click 'Generate Command' first, or ensure the command box has content."
        return
    try:
        project_root = Path(__file__).parent.parent
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        output = ""
        for line in iter(process.stdout.readline, ""):
            output += line
            yield output

        process.stdout.close()
        return_code = process.wait()

        if return_code != 0:
            output += f"\n\n--- Process exited with code {return_code} ---"
            yield output

    except Exception as e:
        yield f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"


def build_single_run_training_tab(gpu_dropdown):
    """Build the Single Run Training tab UI."""
    with gr.TabItem("Single Run Training", id="single_run_training"):
        gr.Markdown("## Single Run Training")
        gr.Markdown("Train one model with fixed hyperparameters. All options in one place.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### HPO Arguments")
                study_name = gr.Textbox(
                    label="Study Name",
                    value=f"{datetime.now().strftime('%m.%d.%y')}_single",
                    info="Study name for logging",
                )
                granularity = gr.Dropdown(
                    label="Granularity",
                    choices=["daily", "hourly", "weekly", "minute"],
                    value="daily",
                    info="Data granularity",
                )
                experiment_name = gr.Textbox(
                    label="Experiment Name",
                    value=f"{datetime.now().strftime('%m.%d.%y')}_experiment",
                    info="MLflow experiment name",
                )
                model_id = gr.Textbox(
                    label="Model ID",
                    value="",
                    info="Optional. Auto-generated if empty.",
                )
                aggregate = gr.Number(label="Aggregate", value=1, info="Aggregation period")

                start_date = gr.DateTime(
                    label="Start Date",
                    value="2014-09-17",
                    info="Training start (YYYY-MM-DD)",
                    include_time=False,
                )
                end_date = gr.DateTime(
                    label="End Date",
                    value="2024-02-16",
                    info="Training end (YYYY-MM-DD)",
                    include_time=False,
                )
                data_path = gr.Textbox(label="Data Path", value="", info="Custom data path (optional)")

                start_date.change(start_before_end, inputs=[start_date, end_date], outputs=start_date)
                end_date.change(end_after_start, inputs=[end_date, start_date], outputs=end_date)

                with gr.Row():
                    returns = gr.Checkbox(label="Returns", value=False, info="Train on returns")
                    volatility = gr.Checkbox(label="Volatility", value=False, info="Use volatility target")
                    no_inf_aggregate = gr.Checkbox(
                        label="No Inference Aggregate",
                        value=False,
                        info="Disable inference aggregation",
                    )
                    log_all_metrics = gr.Checkbox(
                        label="Log All Metrics",
                        value=False,
                        info="Log all metrics to MLflow",
                    )

            with gr.Column(scale=1):
                gr.Markdown("### Model Parameters (Single Values)")
                features = gr.Dropdown(
                    label="Features",
                    choices=["S", "MS", "M"],
                    value="S",
                    info="S: Univariate→Univariate, M: Multivariate→Multivariate, MS: Multivariate→Univariate",
                )
                seq_len = gr.Number(label="Sequence Length", value=180, info="Input sequence length")
                pred_len = gr.Number(label="Prediction Length", value=14, info="Prediction horizon")
                num_tokens = gr.Number(label="Number of Tokens", value=500, info="Vocabulary size")
                loss = gr.Dropdown(
                    label="Loss Function",
                    choices=["MSE", "MADL", "GMADL", "MADLSTE", "SHARPE"],
                    value="MSE",
                )
                lradj = gr.Dropdown(
                    label="LR Adjustment",
                    choices=["type1", "type2", "type3", "PEMS", "TST", "COS", "constant"],
                    value="TST",
                )
                n_heads = gr.Number(label="Attention Heads", value=8)
                d_ff = gr.Number(label="Feed-Forward Dimension", value=64)
                batch_size = gr.Number(label="Batch Size", value=16)
                patch_len = gr.Number(label="Patch Length", value=14)
                stride = gr.Number(label="Stride", value=7)
                epochs = gr.Number(label="Epochs", value=20)

                gr.Markdown("### Integer/Float Parameters")
                llm_layers = gr.Number(label="LLM Layers", value=1)
                d_model = gr.Number(label="D Model", value=16)
                dropout = gr.Number(label="Dropout", value=0.1)
                pct_start = gr.Number(label="PCT Start", value=0.2)
                learning_rate = gr.Number(label="Learning Rate", value=1e-3)

        gr.Markdown("### Custom Prompt")
        prompt = gr.Textbox(
            label="Dataset Prompt",
            value=_load_prompt_from_file(DEFAULT_PROMPT_FILENAME),
            lines=6,
            info="Edit the prompt that describes your dataset. Save before training.",
        )
        with gr.Row():
            prompt_filename = gr.Textbox(
                label="Prompt Filename",
                value=DEFAULT_PROMPT_FILENAME,
                info="Saved to dataset/prompt_bank/. Used for training.",
            )
            save_prompt_btn = gr.Button("Save Prompt", variant="secondary")
        prompt_status = gr.Textbox(label="Prompt Status", interactive=False)

        save_prompt_btn.click(
            fn=save_prompt,
            inputs=[prompt, prompt_filename],
            outputs=prompt_status,
        )

        gr.Markdown("### Generate & Run")
        with gr.Row():
            generate_cmd_btn = gr.Button("Generate Command", variant="primary")
            generate_yaml_btn = gr.Button("Generate YAML", variant="secondary")
        yaml_output = gr.Code(label="Generated YAML Configuration", language="yaml", lines=25)
        with gr.Row():
            save_yaml_btn = gr.Button("Save YAML as custom_single_config.yaml", variant="secondary")
        yaml_save_status = gr.Textbox(label="YAML Save Status", interactive=False)
        command_output = gr.Textbox(label="Generated Command", lines=4, interactive=True)
        run_train_btn = gr.Button("Train Model", variant="primary")
        train_output = gr.Textbox(label="Training Output", lines=20, interactive=False)

        generate_cmd_btn.click(
            fn=build_single_run_command,
            inputs=[
                gpu_dropdown,
                study_name,
                granularity,
                start_date,
                end_date,
                data_path,
                returns,
                volatility,
                experiment_name,
                aggregate,
                no_inf_aggregate,
                log_all_metrics,
                model_id,
                prompt_filename,
                features,
                seq_len,
                pred_len,
                num_tokens,
                loss,
                lradj,
                n_heads,
                d_ff,
                batch_size,
                patch_len,
                stride,
                epochs,
                llm_layers,
                d_model,
                dropout,
                pct_start,
                learning_rate,
            ],
            outputs=command_output,
        )

        generate_yaml_btn.click(
            fn=generate_single_yaml_config,
            inputs=[
                features,
                seq_len,
                pred_len,
                num_tokens,
                loss,
                lradj,
                n_heads,
                d_ff,
                batch_size,
                patch_len,
                stride,
                epochs,
                llm_layers,
                d_model,
                dropout,
                pct_start,
                learning_rate,
            ],
            outputs=yaml_output,
        )

        def save_yaml_from_ui(yaml_content):
            return save_single_yaml_config(yaml_content, DEFAULT_YAML_FILENAME)

        save_yaml_btn.click(
            fn=save_yaml_from_ui,
            inputs=yaml_output,
            outputs=yaml_save_status,
        )

        run_train_btn.click(
            fn=run_single_train,
            inputs=command_output,
            outputs=train_output,
        )
