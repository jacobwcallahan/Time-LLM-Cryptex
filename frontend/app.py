"""
Gradio frontend for Time-LLM-Cryptex HPO configuration.
Provides a UI for configuring hyperparameter optimization runs.
"""

import gradio as gr
import yaml
from pathlib import Path
import subprocess
import os
from datetime import datetime
from helper_fcns import check_inf_after_train, start_before_end, end_after_start

# Default prompt from CRYPTEX.txt
DEFAULT_PROMPT = """The Binance Bitcoin Hourly Returns (BTC-H) dataset captures granular financial data from the Binance.us cryptocurrency exchange. It spans nearly four months, from July 2024 to December 2024, with hourly-level resolution. Each record contains updates for returns of hourly closing prices and traded volume in USD. Timestamps are stored in Unix time format. Inactive periods (with no trading activity) are represented with NaN values, while missing timestamps may reflect exchange/API downtime or data collection limitations. The dataset has been carefully deduplicated and validated, and is updated nightly to ensure consistency and completeness."""

# Default YAML configuration values from daily_vol.yaml
DEFAULT_CONFIG = {
    "categorical": {
        "features": ["S"],
        "seq_len": [180],
        "pred_len": [14],
        "num_tokens": [500, 1000],
        "loss": ["MSE"],
        "lradj": ["TST", "type1", "COS", "type3"],
        "n_heads": [2, 4, 8, 16],
        "d_ff": [32, 64, 128, 256],
        "batch_size": [8, 16],
        "patch_len": [7, 14, 21],
        "stride": [7, 20],
        "epochs": [20]
    },
    "int": {
        "llm_layers": {"low": 1, "high": 2},
        "d_model": {"low": 16, "high": 32, "step": 16}
    },
    "float": {
        "dropout": {"low": 0.0, "high": 0.5, "step": 0.1},
        "pct_start": {"low": 0.1, "high": 0.5, "step": 0.1},
        "learning_rate": {"low": 1e-5, "high": 1e-1, "log": True}
    }
}


def build_command(
    # HPO Arguments
    gpu, study_name, granularity, start_date, end_date,
    inf_start, inf_end, data_path, returns, backtest,
    experiment_name, trials, aggregate, no_inf_aggregate,
    log_all_metrics, yaml_file, volatility,
    # Config options - Categorical
    features, seq_len, pred_len, num_tokens, loss, lradj,
    n_heads, d_ff, batch_size, patch_len, stride, epochs,
    # Config options - Int
    llm_layers_low, llm_layers_high,
    d_model_low, d_model_high, d_model_step,
    # Config options - Float
    dropout_low, dropout_high, dropout_step,
    pct_start_low, pct_start_high, pct_start_step,
    lr_low, lr_high, lr_log,
    # Prompt
    prompt
):
    """Build the command string for running HPO."""
    cmd_parts = ["python3", "run_hpo.py"]
    
    # Add HPO arguments
    if gpu:
        cmd_parts.extend(["--gpu", str(gpu)])
    if study_name:
        cmd_parts.extend(["--study_name", study_name])
    if granularity:
        cmd_parts.extend(["--granularity", granularity])
    if start_date:
        cmd_parts.extend(["--start", str(datetime.fromtimestamp(start_date).strftime("%Y-%m-%d"))])
    if end_date:
        cmd_parts.extend(["--end", str(datetime.fromtimestamp(end_date).strftime("%Y-%m-%d"))])
    if inf_start:
        cmd_parts.extend(["--inf_start", str(datetime.fromtimestamp(inf_start).strftime("%Y-%m-%d"))])
    if inf_end:
        cmd_parts.extend(["--inf_end", str(datetime.fromtimestamp(inf_end).strftime("%Y-%m-%d"))])
    if data_path:
        cmd_parts.extend(["--data_path", data_path])
    if returns:
        cmd_parts.append("--returns")
    if backtest:
        cmd_parts.append("--backtest")
    if experiment_name:
        cmd_parts.extend(["--experiment_name", experiment_name])
    if trials:
        cmd_parts.extend(["--trials", str(trials)])
    if aggregate:
        cmd_parts.extend(["--aggregate", str(aggregate)])
    if no_inf_aggregate:
        cmd_parts.append("--no_inf_aggregate")
    if log_all_metrics:
        cmd_parts.append("--log_all_metrics")
    if yaml_file:
        cmd_parts.extend(["--yaml_file", yaml_file])
    if volatility:
        cmd_parts.append("--volatility")
    
    cmd = " ".join(cmd_parts)
    
    return cmd


def parse_custom_values(custom_str, is_numeric=True):
    """Parse comma-separated custom values from a string."""
    if not custom_str or not custom_str.strip():
        return []
    values = [v.strip() for v in custom_str.split(",") if v.strip()]
    if is_numeric:
        parsed = []
        for v in values:
            try:
                # Try int first, then float
                if "." in v:
                    parsed.append(float(v))
                else:
                    parsed.append(int(v))
            except ValueError:
                pass  # Skip invalid values
        return parsed
    return values


def merge_selections(checkbox_vals, custom_str, is_numeric=True, default=None):
    """Merge checkbox selections with custom values."""
    if custom_str is None:
        return checkbox_vals

    result = list(checkbox_vals) if checkbox_vals else []
    custom = parse_custom_values(custom_str, is_numeric)
    
    # Add custom values that aren't already in the list
    for val in custom:
        str_val = str(val)
        if str_val not in result and val not in result:
            result.append(str_val if not is_numeric else val)
    
    if not result and default is not None:
        return default
    return result


def generate_yaml_config(
    # Categorical
    features,
    seq_len, seq_len_custom,
    pred_len, pred_len_custom,
    num_tokens, num_tokens_custom,
    loss,
    lradj,
    n_heads, n_heads_custom,
    d_ff, d_ff_custom,
    batch_size, batch_size_custom,
    patch_len, patch_len_custom,
    stride, stride_custom,
    epochs, epochs_custom,
    # Int
    llm_layers_low, llm_layers_high,
    d_model_low, d_model_high, d_model_step,
    # Float
    dropout_low, dropout_high, dropout_step,
    pct_start_low, pct_start_high, pct_start_step,
    lr_low, lr_high, lr_log
):
    """Generate YAML configuration from UI inputs."""
    # Merge checkbox selections with custom values
    seq_len_merged = merge_selections(seq_len, seq_len_custom, is_numeric=True, default=[180])
    pred_len_merged = merge_selections(pred_len, pred_len_custom, is_numeric=True, default=[14])
    num_tokens_merged = merge_selections(num_tokens, num_tokens_custom, is_numeric=True, default=[100, 500, 1000])
    n_heads_merged = merge_selections(n_heads, n_heads_custom, is_numeric=True, default=[2, 4, 8, 16])
    d_ff_merged = merge_selections(d_ff, d_ff_custom, is_numeric=True, default=[32, 64, 128, 256])
    batch_size_merged = merge_selections(batch_size, batch_size_custom, is_numeric=True, default=[8, 16])
    patch_len_merged = merge_selections(patch_len, patch_len_custom, is_numeric=True, default=[7, 14, 21])
    stride_merged = merge_selections(stride, stride_custom, is_numeric=True, default=[7, 20])
    epochs_merged = merge_selections(epochs, epochs_custom, is_numeric=True, default=[20])
    
    # Convert string values to int for numeric fields
    def to_int_list(lst):
        return [int(x) for x in lst]
    
    config = {
        "categorical": {
            "features": features,
            "seq_len": to_int_list(seq_len_merged),
            "pred_len": to_int_list(pred_len_merged),
            "num_tokens": to_int_list(num_tokens_merged),
            "loss": loss,
            "lradj": lradj,
            "n_heads": to_int_list(n_heads_merged),
            "d_ff": to_int_list(d_ff_merged),
            "batch_size": to_int_list(batch_size_merged),
            "patch_len": to_int_list(patch_len_merged),
            "stride": to_int_list(stride_merged),
            "epochs": to_int_list(epochs_merged)
        },
        "int": {
            "llm_layers": {
                "low": int(llm_layers_low),
                "high": int(llm_layers_high)
            },
            "d_model": {
                "low": int(d_model_low),
                "high": int(d_model_high),
                "step": int(d_model_step)
            }
        },
        "float": {
            "dropout": {
                "low": float(dropout_low),
                "high": float(dropout_high),
                "step": float(dropout_step)
            },
            "pct_start": {
                "low": float(pct_start_low),
                "high": float(pct_start_high),
                "step": float(pct_start_step)
            },
            "learning_rate": {
                "low": float(lr_low),
                "high": float(lr_high),
                "log": lr_log
            }
        }
    }
    return yaml.dump(config, default_flow_style=False, sort_keys=False)


def save_prompt(prompt, filename="CRYPTEX.txt"):
    """Save the prompt to the prompt bank."""
    prompt_path = Path("dataset/prompt_bank") / filename
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prompt_path, "w") as f:
        f.write(prompt)
    return f"Prompt saved to {prompt_path}"


def save_yaml_config(yaml_content, filename):
    """Save YAML configuration to file."""
    config_path = Path("config") / filename
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        f.write(yaml_content)
    return f"Configuration saved to {config_path}"


def run_hpo(command):
    """Execute the HPO command."""
    try:
        # Change to project root directory
        project_root = Path(__file__).parent.parent
        result = subprocess.run(
            command,
            shell=True,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        output = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 1 hour"
    except Exception as e:
        return f"Error: {str(e)}"


# Build the Gradio interface
with gr.Blocks(title="Time-LLM-Cryptex HPO Configuration", theme=gr.themes.Citrus()) as app:
    gr.Markdown("# Time-LLM-Cryptex Hyperparameter Optimization")
    gr.Markdown("Configure and run hyperparameter optimization for Time-LLM models.")
    
    with gr.Tabs():
        # Tab 1: HPO Arguments
        with gr.TabItem("HPO Arguments"):
            gr.Markdown("### Run HPO Configuration")
            with gr.Row():
                with gr.Column():
                    gpu = gr.Textbox(label="GPU", value="1", info="GPU to use")
                    study_name = gr.Textbox(label="Study Name", value=f"{datetime.now().strftime('%m.%d.%y')}_study", info="Optuna study name")
                    granularity = gr.Dropdown(
                        label="Granularity",
                        choices=["daily", "hourly", "weekly", "minute"],
                        value="daily",
                        info="Data granularity"
                    )
                    experiment_name = gr.Textbox(label="Experiment Name", value=f"{datetime.now().strftime('%m.%d.%y')}_experiment", info="MLflow experiment name")
                    trials = gr.Number(label="Trials", value=10, info="Number of optimization trials")
                    aggregate = gr.Number(label="Aggregate", value=1, info="Aggregation period")
                
                with gr.Column():
                    start_date = gr.DateTime(label="Start Date", value="2014-09-17", info="Training start date (YYYY-MM-DD)", include_time=False)
                    end_date = gr.DateTime(label="End Date", value="2024-02-16", info="Training end date (YYYY-MM-DD)", include_time=False)
                    inf_start = gr.DateTime(label="Inference Start", value="", info="Inference start date (YYYY-MM-DD)", include_time=False)
                    inf_end = gr.DateTime(label="Inference End", value="", info="Inference end date (YYYY-MM-DD)", include_time=False)
                    data_path = gr.Textbox(label="Data Path", value="", info="Custom data path (optional)")
                    yaml_file = gr.Textbox(label="YAML Config File", value="optuna_vars.yaml", info="Config file in ./config/")

                    inf_start.change(check_inf_after_train, inputs=[end_date, inf_start], outputs=inf_start)
                    start_date.change(start_before_end, inputs=[start_date, end_date], outputs=start_date)
                    inf_start.change(start_before_end, inputs=[inf_start, inf_end], outputs=inf_start)
                    end_date.change(end_after_start, inputs=[end_date, start_date], outputs=end_date)
                    inf_end.change(end_after_start, inputs=[inf_end, inf_start], outputs=inf_end)
            
            with gr.Row():
                returns = gr.Checkbox(label="Returns", value=False, info="Train on returns")
                backtest = gr.Checkbox(label="Backtest", value=False, info="Run backtest after training")
                volatility = gr.Checkbox(label="Volatility", value=False, info="Use volatility target")
                no_inf_aggregate = gr.Checkbox(label="No Inference Aggregate", value=False, info="Disable inference aggregation")
                log_all_metrics = gr.Checkbox(label="Log All Metrics", value=False, info="Log all metrics to MLflow")
        
        # Tab 2: Model Configuration (from YAML)
        with gr.TabItem("Model Configuration"):
            gr.Markdown("### Categorical Parameters")
            gr.Markdown("*Add custom values as comma-separated list (e.g., `64, 128, 256`)*")
            with gr.Row():
                with gr.Column():
                    features = gr.CheckboxGroup(
                        label="Features",
                        choices=["S", "MS", "M"],
                        value=["S"],
                        info="Feature types - S: Univariate → Univariate, M: Multivariate → Multivariate, MS: Multivariate → Univariate"

                    )
                    
                    seq_len = gr.CheckboxGroup(
                        label="Sequence Length",
                        choices=["72", "96", "120", "168", "180"],
                        value=["180"],
                        info="Input sequence lengths"
                    )
                    seq_len_custom = gr.Textbox(
                        label="Custom Sequence Lengths",
                        placeholder="e.g., 48, 256",
                        info="Add custom sequence lengths"
                    )
                    
                    pred_len = gr.CheckboxGroup(
                        label="Prediction Length",
                        choices=["2", "7", "14", "21"],
                        value=["14"],
                        info="Prediction horizons"
                    )
                    pred_len_custom = gr.Textbox(
                        label="Custom Prediction Lengths",
                        placeholder="e.g., 1, 28",
                        info="Add custom prediction lengths"
                    )
                    
                    num_tokens = gr.CheckboxGroup(
                        label="Number of Tokens",
                        choices=["100", "500", "1000"],
                        value=["500", "1000"],
                        info="Vocabulary sizes"
                    )
                    num_tokens_custom = gr.Textbox(
                        label="Custom Token Counts",
                        placeholder="e.g., 250, 2000",
                        info="Add custom vocabulary sizes"
                    )
                    batch_size = gr.CheckboxGroup(
                        label="Batch Size",
                        choices=["8", "16", "32"],
                        value=["8", "16"],
                        info="Training batch sizes"
                    )
                    batch_size_custom = gr.Textbox(
                        label="Custom Batch Sizes",
                        placeholder="e.g., 4, 64",
                        info="Add custom batch sizes"
                    )
                    
                    patch_len = gr.CheckboxGroup(
                        label="Patch Length",
                        choices=["7", "12", "14", "16", "21", "24"],
                        value=["7", "14", "21"],
                        info="Patch lengths"
                    )
                    patch_len_custom = gr.Textbox(
                        label="Custom Patch Lengths",
                        placeholder="e.g., 4, 32",
                        info="Add custom patch lengths"
                    )
                
                with gr.Column():
                    loss = gr.CheckboxGroup(
                        label="Loss Function",
                        choices=["MSE", "MADL", "GMADL", "MADLSTE"],
                        value=["MSE"],
                        info="Loss functions to try"
                    )
                    
                    lradj = gr.CheckboxGroup(
                        label="LR Adjustment",
                        choices=["type1", "type2", "type3", "PEMS", "TST", "COS", "constant"],
                        value=["TST", "type1", "COS", "type3"],
                        info="Learning rate schedulers"
                    )
                    
                    n_heads = gr.CheckboxGroup(
                        label="Attention Heads",
                        choices=["2", "4", "8", "16"],
                        value=["2", "4", "8", "16"],
                        info="Number of attention heads"
                    )
                    n_heads_custom = gr.Textbox(
                        label="Custom Head Counts",
                        placeholder="e.g., 1, 32",
                        info="Add custom attention head counts"
                    )
                    
                    d_ff = gr.CheckboxGroup(
                        label="Feed-Forward Dimension",
                        choices=["32", "64", "128", "256"],
                        value=["32", "64", "128", "256"],
                        info="FFN dimensions"
                    )
                    d_ff_custom = gr.Textbox(
                        label="Custom FFN Dimensions",
                        placeholder="e.g., 512, 1024",
                        info="Add custom FFN dimensions"
                    )
                    stride = gr.CheckboxGroup(
                        label="Stride",
                        choices=["6", "7", "12", "20"],
                        value=["7", "20"],
                        info="Stride values"
                    )
                    stride_custom = gr.Textbox(
                        label="Custom Strides",
                        placeholder="e.g., 4, 14",
                        info="Add custom stride values"
                    )
                    
                    epochs = gr.CheckboxGroup(
                        label="Epochs",
                        choices=["10", "15", "20", "25", "30"],
                        value=["20"],
                        info="Training epochs"
                    )
                    epochs_custom = gr.Textbox(
                        label="Custom Epochs",
                        placeholder="e.g., 5, 50",
                        info="Add custom epoch counts"
                    )
                                
            gr.Markdown("### Integer Parameters (Range)")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**LLM Layers**")
                    llm_layers_low = gr.Number(label="Low", value=1)
                    llm_layers_high = gr.Number(label="High", value=2)
                
                with gr.Column():
                    gr.Markdown("**D Model**")
                    d_model_low = gr.Number(label="Low", value=16)
                    d_model_high = gr.Number(label="High", value=32)
                    d_model_step = gr.Number(label="Step", value=16)
            
            gr.Markdown("### Float Parameters (Range)")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Dropout**")
                    dropout_low = gr.Number(label="Low", value=0.0)
                    dropout_high = gr.Number(label="High", value=0.5)
                    dropout_step = gr.Number(label="Step", value=0.1)
                
                with gr.Column():
                    gr.Markdown("**PCT Start**")
                    pct_start_low = gr.Number(label="Low", value=0.1)
                    pct_start_high = gr.Number(label="High", value=0.5)
                    pct_start_step = gr.Number(label="Step", value=0.1)
                
                with gr.Column():
                    gr.Markdown("**Learning Rate**")
                    lr_low = gr.Number(label="Low", value=1e-5)
                    lr_high = gr.Number(label="High", value=1e-1)
                    lr_log = gr.Checkbox(label="Log Scale", value=True)
        
        # Tab 3: Prompt Configuration
        with gr.TabItem("Prompt"):
            gr.Markdown("### Dataset Prompt")
            gr.Markdown("This prompt describes the dataset and is used by the model for context.")
            prompt = gr.Textbox(
                label="Prompt",
                value=DEFAULT_PROMPT,
                lines=10,
                info="Edit the prompt that describes your dataset"
            )
            prompt_filename = gr.Textbox(label="Prompt Filename", value="CRYPTEX.txt")
            save_prompt_btn = gr.Button("Save Prompt", variant="secondary")
            prompt_status = gr.Textbox(label="Status", interactive=False)
            
            save_prompt_btn.click(
                fn=save_prompt,
                inputs=[prompt, prompt_filename],
                outputs=prompt_status
            )
        
        # Tab 4: Generate & Run
        with gr.TabItem("Generate & Run"):
            gr.Markdown("### Generate Configuration and Command")
            
            with gr.Row():
                generate_cmd_btn = gr.Button("Generate Command", variant="primary")
                generate_yaml_btn = gr.Button("Generate YAML", variant="secondary")
            
            command_output = gr.Textbox(label="Generated Command", lines=3, interactive=True)
            yaml_output = gr.Code(label="Generated YAML Configuration", language="yaml", lines=30)
            
            with gr.Row():
                save_yaml_filename = gr.Textbox(label="YAML Filename", value="custom_config.yaml")
                save_yaml_btn = gr.Button("Save YAML Config", variant="secondary")
            yaml_save_status = gr.Textbox(label="Save Status", interactive=False)
            
            gr.Markdown("### Execute HPO")
            run_btn = gr.Button("Run HPO", variant="primary")
            run_output = gr.Textbox(label="Execution Output", lines=20, interactive=False)
            
            # Wire up the buttons
            generate_cmd_btn.click(
                fn=build_command,
                inputs=[
                    gpu, study_name, granularity, start_date, end_date,
                    inf_start, inf_end, data_path, returns, backtest,
                    experiment_name, trials, aggregate, no_inf_aggregate,
                    log_all_metrics, yaml_file, volatility,
                    features, seq_len, pred_len, num_tokens, loss, lradj,
                    n_heads, d_ff, batch_size, patch_len, stride, epochs,
                    llm_layers_low, llm_layers_high,
                    d_model_low, d_model_high, d_model_step,
                    dropout_low, dropout_high, dropout_step,
                    pct_start_low, pct_start_high, pct_start_step,
                    lr_low, lr_high, lr_log,
                    prompt
                ],
                outputs=command_output
            )
            
            generate_yaml_btn.click(
                fn=generate_yaml_config,
                inputs=[
                    features, # no custom values
                    seq_len, seq_len_custom,
                    pred_len, pred_len_custom,
                    num_tokens, num_tokens_custom,
                    loss, # no custom values
                    lradj, # no custom values
                    n_heads, n_heads_custom,
                    d_ff, d_ff_custom,
                    batch_size, batch_size_custom,
                    patch_len, patch_len_custom,
                    stride, stride_custom,
                    epochs, epochs_custom,
                    llm_layers_low, llm_layers_high,
                    d_model_low, d_model_high, d_model_step,
                    dropout_low, dropout_high, dropout_step,
                    pct_start_low, pct_start_high, pct_start_step,
                    lr_low, lr_high, lr_log
                ],
                outputs=yaml_output
            )
            
            save_yaml_btn.click(
                fn=save_yaml_config,
                inputs=[yaml_output, save_yaml_filename],
                outputs=yaml_save_status
            )
            
            run_btn.click(
                fn=run_hpo,
                inputs=command_output,
                outputs=run_output
            )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
