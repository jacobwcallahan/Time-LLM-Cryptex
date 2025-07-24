import argparse
import torch
import pandas as pd
import numpy as np
import mlflow
import os
import pickle
from types import SimpleNamespace
from tqdm import tqdm
from models import TimeLLM
import tempfile

def parse_args():
    # Parse command-line arguments for inference
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='MLflow run name/model ID to load model and config from')
    parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM backbone name (should match training)')
    parser.add_argument('--mlflow_tracking_uri', type=str, default=None, help='Optional MLflow tracking URI')
    parser.add_argument('--data_path', type=str, default=None, help='Optional override for input data CSV')
    return parser.parse_args()

def cast_params(params):
    # Convert string params from MLflow to correct types for model config
    int_keys = ['seq_len', 'pred_len', 'enc_in', 'd_model', 'n_heads', 'd_ff', 'patch_len', 'stride', 'llm_layers', 'num_tokens', 'percent']
    float_keys = ['dropout']
    for k in int_keys:
        if k in params: params[k] = int(params[k])
    for k in float_keys:
        if k in params: params[k] = float(params[k])
    return params

def load_mlflow_artifacts_and_args(model_id, llm_model, tracking_uri=None):
    # Load model state dict, scaler, and config params from MLflow
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(llm_model)
    runs = client.search_runs([experiment.experiment_id], f"tags.mlflow.runName = '{model_id}'")
    run = runs[0]
    run_id = run.info.run_id
    # Download model weights and scaler
    model_state_path = client.download_artifacts(run_id, "model_state_dict")
    scaler_path = client.download_artifacts(run_id, "scaler.pkl")
    # Load and cast params
    params = dict(run.data.params)
    params = cast_params(params)
    args = SimpleNamespace(**params)
    args.model_id = model_id
    args.llm_model = llm_model
    return args, model_state_path, scaler_path, run_id

def main():
    cli_args = parse_args()
    # Load config, model, scaler, and MLflow run ID
    args, model_state_path, scaler_path, run_id = load_mlflow_artifacts_and_args(
        cli_args.model_id, cli_args.llm_model, cli_args.mlflow_tracking_uri)
    # Allow CLI override for data_path
    if cli_args.data_path: args.data_path = cli_args.data_path
    # Load the full input data CSV
    df_raw = pd.read_csv(args.data_path)
    # Ensure columns are ordered: timestamp, features..., target
    feature_cols = [col for col in df_raw.columns if col not in ['timestamp', args.target]]
    ordered_cols = ['timestamp'] + feature_cols + [args.target]
    df_raw = df_raw[ordered_cols]

    # Instantiate model and load weights
    model = TimeLLM.Model(args)
    state_dict = torch.load(model_state_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    # Load scaler for input normalization
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    seq_len, pred_len, num_rows = args.seq_len, args.pred_len, len(df_raw)
    results = []
    # Find the index of the target column in the DataFrame
    target_idx = df_raw.columns.get_loc(args.target)

    # Rolling window inference: for each row, predict pred_len steps ahead
    # For the first seq_len rows, predictions will be NaN (not enough history)
    for i in tqdm(range(seq_len, num_rows)):
        # Take the previous seq_len rows as input
        input_window = df_raw.iloc[i - seq_len:i]
        # Drop timestamp, keep only features (shape: [seq_len, num_features])
        input_features = input_window.drop(columns=['timestamp']).values.astype(np.float32)
        # Normalize features using training scaler
        input_scaled = scaler.transform(input_features)
        # Convert to tensor: shape [1, seq_len, num_features]
        input_tensor = torch.tensor(input_scaled).unsqueeze(0).to(device)
        # Model prediction: output shape [1, pred_len, num_features]
        with torch.no_grad():
            output = model(input_tensor)
        output = output.cpu().numpy()[0]  # shape: [pred_len, num_features]
        # Extract predictions for the target feature: shape [pred_len]
        pred_target = output[:, target_idx]
        # Prepare output row: copy last input row's original data
        last_row = input_window.iloc[-1]
        row_dict = {col: last_row[col] for col in df_raw.columns}
        # Add pred_len forecast columns for this row
        for j in range(pred_len):
            # Always output pred_len columns, even if forecasting beyond data
            row_dict[f'{args.target}_predicted_{j+1}'] = pred_target[j] if j < len(pred_target) else np.nan
        row_dict['timestamp'] = int(last_row['timestamp'])
        results.append(row_dict)

    # For the first seq_len rows, fill with NaN predictions (not enough history)
    for i in range(seq_len):
        row_dict = {col: df_raw.iloc[i][col] for col in df_raw.columns}
        for j in range(pred_len):
            row_dict[f'{args.target}_predicted_{j+1}'] = np.nan
        row_dict['timestamp'] = int(df_raw.iloc[i]['timestamp'])
        results.insert(i, row_dict)

    # Save all results to a temporary CSV file, log to MLflow, then delete
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        result_df = pd.DataFrame(results)
        result_df.to_csv(tmp.name, index=False)
        temp_csv_path = tmp.name
    mlflow.set_experiment(args.llm_model)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(temp_csv_path)
        mlflow.set_tag('inference', 'completed')
        print(f"Logged inference results to MLflow run {run_id}.")
    os.remove(temp_csv_path)

if __name__ == '__main__':
    main() 