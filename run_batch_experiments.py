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
    'train_epochs': 10,
    'batch_size': 16,
    'eval_batch_size': 8,
    'learning_rate': 0.0001,
    'loss': 'MSE',
    'metric': 'MAE',
    'lradj': 'type1',
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
        'name': 'daily_50pct_test',
        'data_path': 'cryptex/daily/candlesticks-D.csv',
        'percent': 20,  # Use 50% of the data
    },
    # Add more experiments here as needed
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='1', help='If not GPU 1, changes OPTUNA_STORAGE_PATH.')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Directory for checkpoints.')
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
    
    print("=== Batch Experiments Runner ===")
    print(f"Total experiments to run: {len(EXPERIMENTS)}")
    print(f"Global LLM model: {GLOBAL_PARAMS['llm_model']}")
    print(f"MLflow server: {MLFLOW_SERVER_IP}:5000")
    print("=" * 50)
    
    successful_experiments = []
    failed_experiments = []
    
    for i, experiment_config in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] Running experiment: {experiment_config['name']}")
        
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
