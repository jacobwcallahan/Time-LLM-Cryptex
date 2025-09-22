import optuna
import subprocess
import sys
import mlflow
import uuid
import time
import os
import argparse

# --- Centralized Configuration ---
MLFLOW_SERVER_IP = "192.168.1.103"
# MLflow
os.environ["MLFLOW_TRACKING_URI"] = f"http://{MLFLOW_SERVER_IP}:5000" # Assumes the server is running. Can set to "" to save locally

# MinIO
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{MLFLOW_SERVER_IP}:9000"

# Optuna
llm_model = "LLAMA3.1"
N_TRIALS = 50
OPTUNA_STORAGE_PATH = f"sqlite:////mnt/nfs/mlflow/optuna_study.db"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='1', help='If not GPU 1, changes OPTUNA_STORAGE_PATH. Also assigns MLFLOW_SERVER_IP')
    return parser.parse_args()

def stream_and_capture(cmd):
    """
    Streams and captures the output of a command. 
    This is used to print the output of the run_main.py file to the console in real-time.
    As well as capture the output of the command in a list of lines.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    lines = []
    for line in iter(proc.stdout.readline, ''):
        print(line, end='', flush=True)
        lines.append(line)
    proc.stdout.close()
    rc = proc.wait()
    return rc, ''.join(lines)


# Helper function
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
    
    if not runs:
        print(f"Warning: Could not find MLflow run with model_id {model_id}")
        return None
    
    return runs[0]

# --- 1. Define the Objective Function ---
# This function defines a single experiment run. Optuna will call it multiple times.
def objective(trial):
    """
    Defines one trial in the Optuna study.
    Optuna will suggest hyperparameter values, which we use to launch run_main.py.
    The function returns the metric we want to optimize (e.g., validation loss).
    """
    # --- 2. Define the Hyperparameter Search Space ---
    # Optuna will intelligently pick values from these ranges/choices.
    
    # Categorical parameters: Optuna will choose from the list.
    features = trial.suggest_categorical("features", ["M", "MS", "S"])
    seq_len = trial.suggest_categorical("seq_len", [14, 49, 91, 168])
    pred_len = trial.suggest_categorical("pred_len", [2, 7, 14])
    num_tokens = trial.suggest_categorical("num_tokens", [100, 500, 1000])
    loss = trial.suggest_categorical("loss", ["MSE", "MADL", "GMADL"])
    lradj = trial.suggest_categorical("lradj", ["type1", "type2", "type3", "PEMS", "TST", "constant"])

    # Integer parameters: Optuna will choose an integer within the range.
    llm_layers = trial.suggest_int("llm_layers", 4, 12)
    d_model = trial.suggest_int("d_model", 16, 64, step=16) # Suggests 16, 32, 48, 64
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8, 16])
    d_ff = trial.suggest_categorical("d_ff", [32, 64, 128, 256])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 24, 32, 64])
    patch_len = trial.suggest_categorical("patch_len", [4, 8, 16, 32])
    stride = trial.suggest_categorical("stride", [1, 4, 8])

    # Float parameters
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    pct_start = trial.suggest_float("pct_start", 0.1, 0.5, step=0.1)

    # Logarithmic uniform parameters: Good for searching learning rates.
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    # --- Static Parameters (won't be tuned in this study) ---
    # llm_model = "LLAMA" # Defined outside the function to be used in Optuna study name
    granularity = "weekly-full"
    metric = "MDA"
    
    # --- Dynamic/Conditional Parameters ---
    # Generate a unique model_id for each trial
    trial_id = str(uuid.uuid4())[:8]
    model_id = f"{llm_model}_L{llm_layers}_{features}_seq{seq_len}_trial_{trial_id}"
    
    # Set data path based on granularity
    data_path_map = {
        'hourly': 'candlesticks-h.csv',
        'minute': 'candlesticks-Min.csv',
        'daily': 'candlesticks-D.csv', 
        'weekly': 'candlesticks-W.csv',
        'weekly-full': 'candlesticks-W-2024-2025.csv'
    }
    data_path = data_path_map[granularity]

    # --- 3. Build and Launch the Experiment Command ---
    # This assembles the command to run your main training script.
    cmd = [
        'accelerate', 'launch', '--multi_gpu', '--mixed_precision', 'bf16', '--num_processes', '4', '--main_process_port', '29500',
        'run_main.py',
        # Tuned Parameters
        '--model_id', model_id,
        '--features', features,
        '--seq_len', str(seq_len),
        '--pred_len', str(pred_len),
        '--llm_layers', str(llm_layers),
        '--d_model', str(d_model),
        '--n_heads', str(n_heads),
        '--d_ff', str(d_ff),
        '--dropout', str(dropout),
        '--patch_len', str(patch_len),
        '--stride', str(stride),
        '--batch_size', str(batch_size),
        '--learning_rate', str(learning_rate),
        '--num_tokens', str(num_tokens),
        '--loss', loss,
        '--lradj', lradj,
        '--pct_start', str(pct_start),
        '--metric', metric,
        # Static Parameters
        '--llm_model', llm_model,
        '--data', 'CRYPTEX',
        '--root_path', './dataset/cryptex/',
        '--data_path', data_path,
        '--target', 'close',
        '--train_epochs', '10',
    ]
    
    print(f"\n--- Starting Trial {trial.number} ---\n{' '.join(cmd)}\n")

    # --- 4. Run the Trial and Get the Result ---
    # We use MLflow to get the result of the trial.
    # This is more robust than parsing stdout.
    client = mlflow.tracking.MlflowClient()
    
    # We need to find the MLflow run associated with this trial.
    # We'll use the model_id (which includes trial_id) as a unique tag.
    
    try:
        # Launch the subprocess
        rc, lines = stream_and_capture(cmd)
        
        # After the run completes, find it in MLflow
        time.sleep(2) # Give MLflow a moment to log everything
        run = _find_mlflow_run(client, llm_model, model_id)
        
        if not run:
            raise optuna.exceptions.TrialPruned("Could not find MLflow run post-execution.")

        # Get the validation metric from the last recorded step
        latest_metrics = run.data.metrics
        # The key should match what you log in run_main.py
        validation_metric_key = f"vali_{metric.lower()}_metric" 
        
        if validation_metric_key not in latest_metrics:
            raise optuna.exceptions.TrialPruned(f"Metric '{validation_metric_key}' not found.")
            
        final_metric = latest_metrics[validation_metric_key]
        
        print(f"--- Trial {trial.number} Finished ---")
        print(f"Validation Metric ({validation_metric_key}): {final_metric}")
        
        return final_metric

    except subprocess.CalledProcessError as e:
        print(f"Trial {trial.number} failed with error.")
        print(e.stderr)
        
        time.sleep(2)
        # --- Error Logging to MLflow ---
        run = _find_mlflow_run(client, llm_model, model_id)
        if run:
            failed_run_id = run.info.run_id
            full_output = f"--- STDOUT ---\n{e.stdout}\n\n--- STDERR ---\n{e.stderr}"
            client.log_text(failed_run_id, full_output, f"failed_trial_{trial.number}_error.log")
            print(f"--> Error log saved as an artifact to failed MLflow run ID: {failed_run_id}")
            # Finally, set the run status to FAILED
            client.set_terminated(failed_run_id, "FAILED")

        # Tell Optuna this trial failed and should be pruned.
        raise optuna.exceptions.TrialPruned()


if __name__ == "__main__":
    # --- 5. Create and Run the Optuna Study ---
    # The 'study_name' will group your runs. If you restart the script, it will resume.
    # 'storage' tells Optuna to save results to a local SQLite database.
    args = parse_args()
    if args.gpu != '1':
        OPTUNA_STORAGE_PATH = f"sqlite:////mnt/nfs/mlflow/optuna_study_2.db"

    study = optuna.create_study(
        study_name=f"{llm_model.lower()}_study_weekly_full",
        direction="minimize",  # We want to minimize validation loss/metric
        storage=OPTUNA_STORAGE_PATH,
        load_if_exists=True # Resume study if it already exists
    )
    
    # 'n_trials' is the total number of experiments you want to run.
    # Optuna will intelligently choose the parameters for these runs.
    study.optimize(objective, n_trials=N_TRIALS)

    # --- 6. Print the Results ---
    print("\n--- Hyperparameter Optimization Finished ---")
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Value (min validation metric): {trial.value}")
    
    print("  Best Parameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

