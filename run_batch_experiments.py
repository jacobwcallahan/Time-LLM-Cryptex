import subprocess
import sys
import mlflow
import uuid
import time
import os
import argparse
from datetime import datetime

# --- Centralized Configuration ---
MLFLOW_SERVER_IP = "192.168.1.103"
# MLflow
os.environ["MLFLOW_TRACKING_URI"] = f"http://{MLFLOW_SERVER_IP}:5000" # Assumes the server is running. Can set to "" to save locally

# MinIO
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{MLFLOW_SERVER_IP}:9000"

# --- Global Parameters ---
GLOBAL_PARAMS = {
    # Model architecture
    'llm_model': 'LLAMA3.1',
    'llm_layers': 6,
    'd_model': 32,
    'n_heads': 8,
    'd_ff': 64,
    'dropout': 0.1,
    'patch_len': 16,
    'stride': 8,
    'num_tokens': 1000,
    
    # Data configuration
    'data': 'CRYPTEX',
    'root_path': './dataset',
    'features': 'MS',
    'target': 'close',
    'enc_in': 7,
    
    # Training configuration
    'train_epochs': 10,  # Run 10 epochs with early stopping
    'batch_size': 16,
    'eval_batch_size': 8,
    'learning_rate': 0.00001,  # 10x smaller to prevent gradient explosion
    'loss': 'MSE',
    'metric': 'MDA',
    'lradj': 'constant',
    'pct_start': 0.2,
    'patience': 10,
    'percent': 100,
    
    # Forecasting task
    'seq_len': 96,
    'pred_len': 24,
    
    # Other
    'num_workers': 10,
    'seed': 2021,
    'enable_mlflow': True,
}

# --- Experiments List ---
EXPERIMENTS = [
    {
        'name': 'hourly_1week_to_1day_25pct',
        'data_path': 'cryptex/hourly/candlesticks-h-clean.csv',  # Use cleaned data without NaN
        'seq_len': 168,
        'pred_len': 24,
        'batch_size': 16,
        'percent': 25,
        'loss': 'MSE',
        'metric': 'MDA',
        'lradj': 'constant',
    },
    {
        'name': 'hourly_2weeks_to_1day_25pct',
        'data_path': 'cryptex/hourly/candlesticks-h-clean.csv',  # Use cleaned data without NaN
        'seq_len': 336,
        'pred_len': 24,
        'batch_size': 16,
        'percent': 25,
        'loss': 'MSE',
        'metric': 'MDA',
        'lradj': 'constant',
    },
    {
        'name': 'hourly_1week_to_1day_50pct',
        'data_path': 'cryptex/hourly/candlesticks-h-clean.csv',  # Use cleaned data without NaN
        'seq_len': 168,
        'pred_len': 24,
        'batch_size': 16,
        'percent': 50,
        'loss': 'MSE',
        'metric': 'MDA',
        'lradj': 'constant',
    },
    {
        'name': 'hourly_2weeks_to_1day_50pct',
        'data_path': 'cryptex/hourly/candlesticks-h-clean.csv',  # Use cleaned data without NaN
        'seq_len': 336,
        'pred_len': 24,
        'batch_size': 16,
        'percent': 50,
        'loss': 'MSE',
        'metric': 'MDA',
        'lradj': 'constant',
    },
    {
        'name': 'hourly_1week_to_1day_75pct',
        'data_path': 'cryptex/hourly/candlesticks-h-clean.csv',  # Use cleaned data without NaN
        'seq_len': 168,
        'pred_len': 24,
        'batch_size': 16,
        'percent': 75,
        'loss': 'MSE',
        'metric': 'MDA',
        'lradj': 'constant',
    },
    {
        'name': 'hourly_2weeks_to_1day_75pct',
        'data_path': 'cryptex/hourly/candlesticks-h-clean.csv',  # Use cleaned data without NaN
        'seq_len': 336,
        'pred_len': 24,
        'batch_size': 16,
        'percent': 75,
        'loss': 'MSE',
        'metric': 'MDA',
        'lradj': 'constant',
    },
    # Daily data experiments - 1e-5 constant LR
    {
        'name': 'daily_1week_to_1day_25pct',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 7,
        'pred_len': 1,
        'batch_size': 16,
        'percent': 25,
        'loss': 'MSE',
        'metric': 'MDA',
        'lradj': 'constant',
    },
    {
        'name': 'daily_2weeks_to_1day_25pct',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 14,
        'pred_len': 1,
        'batch_size': 16,
        'percent': 25,
        'loss': 'MSE',
        'metric': 'MDA',
        'lradj': 'constant',
    },
    {
        'name': 'daily_1week_to_1day_50pct',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 7,
        'pred_len': 1,
        'batch_size': 16,
        'percent': 50,
        'loss': 'MSE',
        'metric': 'MDA',
        'lradj': 'constant',
    },
    {
        'name': 'daily_2weeks_to_1day_50pct',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 14,
        'pred_len': 1,
        'batch_size': 16,
        'percent': 50,
        'loss': 'MSE',
        'metric': 'MDA',
        'lradj': 'constant',
    },
    {
        'name': 'daily_1week_to_1day_75pct',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 7,
        'pred_len': 1,
        'batch_size': 16,
        'percent': 75,
        'loss': 'MSE',
        'metric': 'MDA',
        'lradj': 'constant',
    },
    {
        'name': 'daily_2weeks_to_1day_75pct',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 14,
        'pred_len': 1,
        'batch_size': 16,
        'percent': 75,
        'loss': 'MSE',
        'metric': 'MDA',
        'lradj': 'constant',
    },
    {
        'name': 'daily_1week_to_1day_100pct',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 7,
        'pred_len': 1,
        'batch_size': 16,
        'percent': 100,
        'loss': 'MSE',
        'metric': 'MDA',
        'lradj': 'constant',
    },
    {
        'name': 'daily_2weeks_to_1day_100pct',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'seq_len': 14,
        'pred_len': 1,
        'batch_size': 16,
        'percent': 100,
        'loss': 'MSE',
        'metric': 'MDA',
        'lradj': 'constant',
    },
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='1', help='If not GPU 1, changes OPTUNA_STORAGE_PATH.')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Directory for checkpoints.')
    parser.add_argument('--start-from', type=int, default=0, 
                       help='Start from experiment index (0-based). Default: 0 (run all experiments)')
    return parser.parse_args()

def _find_mlflow_run(client, experiment_name, model_id):
    """Finds an MLflow run based on its name within a given experiment."""
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Error: MLflow experiment '{experiment_name}' not found.")
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{model_id}'"
    )
    
    if runs:
        return runs[0]
    else:
        print(f"Error: MLflow run '{model_id}' not found in experiment '{experiment_name}'.")
        return None

def run_experiment(experiment_config, global_params, args):
    """Run a single experiment with given configuration."""
    
    # Merge global params with experiment-specific overrides
    params = global_params.copy()
    params.update(experiment_config)
    
    # Generate unique model_id
    experiment_name = params['name']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_id = f"{experiment_name}_{timestamp}"
    
    # Build the command
    cmd = [
        'accelerate', 'launch', '--multi_gpu', '--mixed_precision', 'bf16', '--num_processes', '4', '--main_process_port', '29500',
        'run_main.py',
        # All parameters
        '--model_id', model_id,
        '--data', params['data'],
        '--root_path', params['root_path'],
        '--data_path', params['data_path'],
        '--features', params['features'],
        '--target', params['target'],
        '--seq_len', str(params['seq_len']),
        '--pred_len', str(params['pred_len']),
        '--enc_in', str(params['enc_in']),
        '--d_model', str(params['d_model']),
        '--n_heads', str(params['n_heads']),
        '--d_ff', str(params['d_ff']),
        '--dropout', str(params['dropout']),
        '--patch_len', str(params['patch_len']),
        '--stride', str(params['stride']),
        '--llm_model', params['llm_model'],
        '--llm_layers', str(params['llm_layers']),
        '--num_tokens', str(params['num_tokens']),
        '--num_workers', str(params['num_workers']),
        '--train_epochs', str(params['train_epochs']),
        '--batch_size', str(params['batch_size']),
        '--eval_batch_size', str(params['eval_batch_size']),
        '--patience', str(params['patience']),
        '--learning_rate', str(params['learning_rate']),
        '--loss', params['loss'],
        '--metric', params['metric'],
        '--lradj', params['lradj'],
        '--pct_start', str(params['pct_start']),
        '--percent', str(params['percent']),
        '--seed', str(params['seed']),
        '--checkpoints', args.checkpoints,
    ]
    
    if params['enable_mlflow']:
        cmd.append('--enable_mlflow')
    
    print(f"\n--- Starting Experiment: {experiment_name} ---")
    print(f"Model ID: {model_id}")
    
    # Pre-execution validation
    print(f"Data path: {params['data_path']}")
    print(f"Full data path: {params['root_path']}/cryptex/{params['data_path']}")
    print(f"Percent of data: {params['percent']}%")
    print(f"Seq len: {params['seq_len']}, Pred len: {params['pred_len']}")
    print(f"Training epochs: {params['train_epochs']}")
    print(f"Batch size: {params['batch_size']}")
    
    # Check if data file exists
    data_file_path = f"{params['root_path']}/cryptex/{params['data_path']}"
    if not os.path.exists(data_file_path):
        print(f"WARNING: Data path does not exist: {data_file_path}")
        # List what's available
        cryptex_path = f"{params['root_path']}/cryptex/"
        if os.path.exists(cryptex_path):
            print(f"Available in {cryptex_path}:")
            for item in os.listdir(cryptex_path):
                print(f"  - {item}")
    
    # Show full command in readable format
    print(f"\n--- Full Command ---")
    for i in range(0, len(cmd), 4):
        print(' '.join(cmd[i:i+4]))
    print("---\n")
    
    # Run the experiment
    client = mlflow.tracking.MlflowClient()
    
    try:
        # Launch the subprocess (stream output to see real-time progress)
        result = subprocess.run(cmd, check=True, text=True)
        print(f"\n--- Experiment {experiment_name} Completed Successfully ---")
        
        # Give MLflow a moment to log everything
        time.sleep(2)
        
        # Find the MLflow run and get metrics
        run = _find_mlflow_run(client, params['llm_model'], model_id)
        if run:
            latest_metrics = run.data.metrics
            metric_key = f'vali_{params["metric"].lower()}_metric'
            print(f"Final validation metric: {latest_metrics.get(metric_key, 'N/A')}")
        
        return True, model_id, None
        
    except subprocess.CalledProcessError as e:
        print(f"\n--- Experiment {experiment_name} Failed ---")
        print(f"Exit code: {e.returncode}")
        # Note: stdout/stderr not available when capture_output=False
        print("See output above for error details")
        
        time.sleep(2)
        # Log error to MLflow
        run = _find_mlflow_run(client, params['llm_model'], model_id)
        if run:
            failed_run_id = run.info.run_id
            # Since we're not capturing output, log a simple error message
            error_message = f"Experiment {experiment_name} failed with exit code {e.returncode}. See console output for details."
            client.log_text(failed_run_id, error_message, f"failed_experiment_{experiment_name}_error.log")
            print(f"--> Error log saved as an artifact to failed MLflow run ID: {failed_run_id}")
            client.set_terminated(failed_run_id, "FAILED")
        
        return False, model_id, str(e)

def main():
    """Main function to run all experiments."""
    args = parse_args()
    
    # Validate start-from index
    if args.start_from < 0:
        print("Error: --start-from must be >= 0")
        return
    if args.start_from >= len(EXPERIMENTS):
        print(f"Error: --start-from ({args.start_from}) >= number of experiments ({len(EXPERIMENTS)})")
        return
    
    # Get experiments to run
    experiments_to_run = EXPERIMENTS[args.start_from:]
    
    print("=== Batch Experiments Runner ===")
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"Starting from experiment {args.start_from + 1} (skipping first {args.start_from} experiments)")
    print(f"Experiments to run: {len(experiments_to_run)}")
    print(f"Global LLM model: {GLOBAL_PARAMS['llm_model']}")
    print(f"MLflow server: {MLFLOW_SERVER_IP}:5000")
    print("=" * 50)
    
    successful_experiments = []
    failed_experiments = []
    
    for i, experiment_config in enumerate(experiments_to_run):
        actual_index = args.start_from + i
        print(f"\n[{actual_index + 1}/{len(EXPERIMENTS)}] Running experiment: {experiment_config['name']}")
        
        success, model_id, error = run_experiment(experiment_config, GLOBAL_PARAMS, args)
        
        if success:
            successful_experiments.append({
                'name': experiment_config['name'],
                'model_id': model_id
            })
        else:
            failed_experiments.append({
                'name': experiment_config['name'],
                'model_id': model_id,
                'error': error
            })
    
    # Print summary
    print("\n" + "=" * 50)
    print("=== BATCH EXPERIMENTS SUMMARY ===")
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"Experiments run: {len(experiments_to_run)}")
    print(f"Successful: {len(successful_experiments)}")
    print(f"Failed: {len(failed_experiments)}")
    
    if successful_experiments:
        print("\nSuccessful experiments:")
        for exp in successful_experiments:
            print(f"  ✓ {exp['name']} -> {exp['model_id']}")
    
    if failed_experiments:
        print("\nFailed experiments:")
        for exp in failed_experiments:
            print(f"  ✗ {exp['name']} -> {exp['model_id']}")
            print(f"    Error: {exp['error']}")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
