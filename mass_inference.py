import argparse
import pandas as pd
import subprocess
import os
import tempfile
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference multiple times on a date-filtered dataset')
    parser.add_argument('--model_id', type=str, required=True, help='MLflow run name/model ID to load model and config from')
    parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM backbone name (should match training)')
    parser.add_argument('--start_date', type=str, required=True, help='Start date for data filtering (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True, help='End date for data filtering (YYYY-MM-DD)')
    parser.add_argument('--n', type=int, default=1, help='Number of inference runs to perform')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save all inference results')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name for MLflow')
    parser.add_argument('--mlflow_tracking_uri', type=str, default=None, help='Optional MLflow tracking URI')
    return parser.parse_args()


def load_and_filter_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Load daily candlestick data and filter by date range."""
    data_path = '/mnt/nfs/datasets/candlesticks-D.csv'
    df = pd.read_csv(data_path)
    
    # Convert timestamp column to datetime
    # Check if timestamp is in unix format (numeric) or already datetime string
    if pd.api.types.is_numeric_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Parse input dates
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Filter by date range (inclusive)
    mask = (df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)
    df_filtered = df[mask].copy()
    
    # Convert timestamp back to unix for compatibility with run_inference.py
    df_filtered['timestamp'] = df_filtered['timestamp'].astype('int64') // 10**9
    
    print(f"Loaded {len(df_filtered)} rows from {start_date} to {end_date}")
    return df_filtered


def run_inference(data_path: str, save_path: str, model_id: str, llm_model: str, 
                  experiment_name: str = None, mlflow_tracking_uri: str = None):
    """Run run_inference.py with the given parameters."""
    cmd = [
        'python', 'run_inference.py',
        '--model_id', model_id,
        '--llm_model', llm_model,
        '--data_path', data_path,
        '--save_path', save_path,
    ]
    
    if experiment_name:
        cmd.extend(['--experiment_name', experiment_name])
    
    if mlflow_tracking_uri:
        cmd.extend(['--mlflow_tracking_uri', mlflow_tracking_uri])
    
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    
    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Load and filter data by date range
    df_filtered = load_and_filter_data(args.start_date, args.end_date)
    
    if len(df_filtered) == 0:
        print("No data found in the specified date range. Exiting.")
        return
    
    # Save filtered data to a temporary file for inference
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        tmp_data_path = tmp_file.name
        df_filtered.to_csv(tmp_data_path, index=False)
    
    try:
        # Run inference n times
        for i in range(1, args.n + 1):
            print(f"\n{'='*50}")
            print(f"Running inference {i}/{args.n}")
            print(f"{'='*50}\n")
            
            # Create a unique save path for this run
            run_save_path = os.path.join(args.output_folder, f'run_{i}')
            os.makedirs(run_save_path, exist_ok=True)
            
            run_inference(
                data_path=tmp_data_path,
                save_path=run_save_path,
                model_id=args.model_id,
                llm_model=args.llm_model,
                experiment_name=args.experiment_name,
                mlflow_tracking_uri=args.mlflow_tracking_uri
            )
            
            print(f"Inference {i} saved to: {run_save_path}/inference.csv")
    
    finally:
        # Clean up temporary data file
        if os.path.exists(tmp_data_path):
            os.remove(tmp_data_path)
    
    print(f"\n{'='*50}")
    print(f"Completed {args.n} inference runs")
    print(f"Results saved in: {args.output_folder}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
