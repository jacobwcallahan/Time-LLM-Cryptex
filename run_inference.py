import argparse
import torch
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import mlflow
import mlflow.pytorch
from utils.model_io import load_model_and_scaler_from_mlflow
from utils.data_utils import prepare_data_for_prediction
from utils.tools import load_content
from utils.timefeatures import time_features
from models import TimeLLM, Autoformer, DLinear

def run_inference(args, model, device, data_path, scaler=None):
    """Improved inference logic with proper data handling"""
    print(f"Running inference on: {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded data with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Prepare data
    scaled_data, scaler, numeric_cols = prepare_data_for_prediction(df, args, scaler)
    
    # Convert to tensors
    data_tensor = torch.FloatTensor(scaled_data)
    
    # Initialize results DataFrame
    results_df = df.copy()
    for i in range(1, args.pred_len + 1):
        results_df[f'close_predicted_{i}'] = np.nan
    
    model.eval()
    print(f"Starting inference loop for {len(df) - args.seq_len} predictions...")
    
    with torch.no_grad():
        for i in range(args.seq_len, len(df)):
            input_data = data_tensor[i-args.seq_len:i].unsqueeze(0).to(device)
            outputs = model(input_data)
            if args.output_attention:
                outputs = outputs[0]
            
            # Extract predictions
            predictions = outputs[:, -args.pred_len:, :]  # [1, pred_len, features]
            predictions_np = predictions.cpu().numpy().squeeze()
            
            # Inverse transform predictions
            if args.features == 'MS':
                f_dim = -1
                predictions_np = predictions_np[:, f_dim:]
            
            # Create padded array for inverse transform
            padded_preds = np.zeros((predictions_np.shape[0], len(numeric_cols)))
            padded_preds[:, :predictions_np.shape[1]] = predictions_np
            predictions_inv = scaler.inverse_transform(padded_preds)
            
            # Find close column index
            close_col_index = numeric_cols.index('close') if 'close' in numeric_cols else 0
            
            # Store predictions
            for j in range(args.pred_len):
                results_df.loc[i, f'close_predicted_{j+1}'] = predictions_inv[j, close_col_index]
            
            if (i + 1) % 100 == 0:
                print(f"\rProcessed {i+1}/{len(df)} datapoints", end="", flush=True)
    
    print(f"\n✅ Inference complete! Generated predictions for {len(df) - args.seq_len} datapoints")
    return results_df

def main():
    parser = argparse.ArgumentParser(description='TimeLLM Inference')
    parser.add_argument('--run_id', type=str, required=True, help='MLflow Run ID to use for inference.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to CSV file for inference.')
    parser.add_argument('--output_path', type=str, default='inference_results.csv', help='Output path for results.')
    parser.add_argument('--model_type', type=str, default='TimeLLM', choices=['TimeLLM', 'Autoformer', 'DLinear'], 
                       help='Model type used in training.')
    
    cli_args = parser.parse_args()

    # --- MLflow Configuration ---
    MLFLOW_SERVER_IP = "192.168.1.103"
    os.environ["MLFLOW_TRACKING_URI"] = f"http://{MLFLOW_SERVER_IP}:5000"
    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{MLFLOW_SERVER_IP}:9000"

    print(f"🔗 Connecting to MLflow server...")
    client = mlflow.tracking.MlflowClient()
    
    try:
        run = client.get_run(cli_args.run_id)
        print(f"✅ Successfully connected to run: {cli_args.run_id}")
    except Exception as e:
        print(f"❌ Error: Could not fetch run details from MLflow. Details: {e}")
        sys.exit(1)

    try:
        # --- Load Model and Scaler ---
        model_class = TimeLLM.Model if cli_args.model_type == 'TimeLLM' else (
            Autoformer.Model if cli_args.model_type == 'Autoformer' else DLinear.Model)
        model, args, scaler = load_model_and_scaler_from_mlflow(cli_args.run_id, model_class)
        
        # --- Set device ---
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {device}")
        model.to(device)
        
        # --- Run Inference ---
        results_df = run_inference(args, model, device, cli_args.data_path, scaler)
        
        # --- Save Results ---
        results_df.to_csv(cli_args.output_path, index=False)
        print(f"💾 Results saved to: {cli_args.output_path}")
        
        # --- Log to MLflow ---
        print("📊 Logging results to MLflow...")
        with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as tmp:
            results_df.to_csv(tmp.name, index=False)
            mlflow.log_artifact(tmp.name, "inference_results")
        os.remove(tmp.name)
        
        client.set_tag(cli_args.run_id, "inference_status", "complete")
        print("✅ Successfully logged artifacts and tagged the run.")
        
        # Print summary
        print(f"\n📈 Inference Summary:")
        print(f"   - Input data: {cli_args.data_path}")
        print(f"   - Output file: {cli_args.output_path}")
        print(f"   - Model: {cli_args.model_type}")
        print(f"   - Predictions generated: {len(results_df) - args.seq_len}")
        print(f"   - Prediction horizon: {args.pred_len} steps")
        
    except Exception as e:
        print(f"\n❌ An error occurred during inference: {e}")
        client.set_tag(cli_args.run_id, "inference_status", "failed")
        print(f"Tagged run {cli_args.run_id} as failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
