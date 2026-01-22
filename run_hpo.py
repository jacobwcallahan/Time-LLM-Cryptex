"""
Runs the hyperparameter optimization for a given model and experiment.
Logs the metrics to the MLflow server.

It uses the Optuna library to run the hyperparameter optimization.
It uses the MLflow server to log the metrics.

The optuna configuration is stored in the ./config/optuna_vars.yaml file.
This can be changed to use a different configuration file in the config folder.

Arguments:
    --gpu: GPU to use. Default is 1
    --study_name: Study Name default is empty
    --granularity: Granularity default is daily
    --start: Start Date default is None
    --end: End Date default is None
    --inf_start: Start Date for inference default is None
    --inf_end: End Date for inference default is None
    --data_path: Data Path default is None
    --returns: If True, converts the data to returns. Default is False
    --backtest: If True, runs the backtest. Default is False
    --experiment_name: Experiment Name default is None
    --trials: Number of trials to run. Default is 10
    --aggregate: Aggregate default is 1
    --no_inf_aggregate: If True, does not aggregate the inference data. Default is False
    --log_all_metrics: If True, logs all metrics to MLflow. Default is False
    --yaml_file: YAML file to use for the study. Default is optuna_vars.yaml. Contained in ./config/
"""

from typing import Any
import optuna
import pandas as pd
import subprocess
import sys
import mlflow
import uuid
import time
import os
import argparse
from datetime import datetime
import yaml
from pathlib import Path
from utils.pipeline import run_inference, perform_backtest, convert_to_returns, metrics_to_db, create_metrics_json, aggregate_data, get_mse_vals, get_mda_vals
import pathlib
import warnings
import sqlite3
import shutil

# --- Centralized Configuration ---
MLFLOW_SERVER_IP = "192.168.1.103"
# MLflow
os.environ["MLFLOW_TRACKING_URI"] = f"http://{MLFLOW_SERVER_IP}:5000" # Assumes the server is running. Can set to "" to save locally

# MinIO
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{MLFLOW_SERVER_IP}:9000"

llm_model = "LLAMA3.1"
OPTUNA_STORAGE_PATH = "sqlite:////data-fast/nfs/mlflow/optuna_study.db" # Optuna storage path
METRICS_DB_PATH = "/data-fast/nfs/mlflow/metrics.db" # Metrics database path
DATASET_PATH = Path("/data-fast/nfs/dataset/") # Dataset path (without specific dataset)
DATA_PATH = Path("temp/data.csv") # Data path in temp folder
INF_PATH = Path("temp/inf_data.csv") # Inference path in temp folder
INFERENCE = False # Bool to determine whether to run inference

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='1', help='If not GPU 1, changes OPTUNA_STORAGE_PATH.')
    parser.add_argument('--study_name', type=str, default='', help='If not empty, uses the study name. Model name is added to the beginning of the study name.')
    parser.add_argument('--granularity', type=str, default='daily', help='Granularity to use. daily, hourly, weekly, minute')
    parser.add_argument('--start', type=str, default=None, help='Start date to use. Format: YYYY-MM-DD')
    parser.add_argument('--end', type=str, default=None, help='End date to use. Format: YYYY-MM-DD')
    parser.add_argument('--inf_start', type=str, default=None, help='Start date to use for inference. Format: YYYY-MM-DD')
    parser.add_argument('--inf_end', type=str, default=None, help='End date to use for inference. Format: YYYY-MM-DD')
    parser.add_argument('--data_path', type=str, default=None, help='Data path to use.(Optional, if not provided, uses the full daily dataset)')
    parser.add_argument("--returns", action='store_true', help='If True, converts the data to returns.')
    parser.add_argument('--backtest', action='store_true', help='If set, run backtest after training')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name to use. Default is None.')
    parser.add_argument('--trials', type=int, default=10, help='Number of trials to run.')
    parser.add_argument('--aggregate', type=int, default=1, help='If set, aggregates from the original granularity to the specified granularity.')
    parser.add_argument('--no_inf_aggregate', action='store_true', help='By default, aggregates inference data. Set this flag to disable aggregation.')
    parser.add_argument('--log_all_metrics', action='store_true', help='By default, logs only the best metric to MLflow. Set this flag to log all metrics (still logs as artifacts).')
    parser.add_argument('--yaml_file', type=str, default='optuna_vars.yaml', help='YAML file to use for the study. Default is optuna_vars.yaml. Contained in ./config/')
    parser.add_argument('--volatility', action='store_true', help='If True, uses the volatility target.')
    return parser.parse_args()
  

# Helper function
def _find_mlflow_run(client, experiment_name, model_id):
    """Finds an MLflow run based on its name within a given experiment.
    Args:
        client: MLflow client
        experiment_name: Experiment name
        model_id: Model id

    Returns:
        run: MLflow run object
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Error: MLflow experiment '{experiment_name}' not found.")
        return None # pruned

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{model_id}'"
    )
    
    if not runs:
        print(f"Warning: Could not find MLflow run with model_id {model_id}")
        return None # pruned
    
    return runs[0]


def create_train_cmd(trial_dict, model_id, data_path):
    """
    Creates the command to train the model and returns it as a list.

    args:
        trial_dict: dictionary of trial parameters
        model_id: model id
        data_path: path to the data
    
    returns:
        cmd (list): command to train the model
    """
    cmd = [
        'accelerate', 'launch', '--multi_gpu', '--mixed_precision', 'bf16', '--num_processes', '4', '--main_process_port', '29500',
        'run_main.py',
        # Tuned Parameters
        '--model_id', model_id,
        '--features', trial_dict['features'],
        '--seq_len', str(trial_dict['seq_len']),
        '--pred_len', str(trial_dict['pred_len']),
        '--llm_layers', str(trial_dict['llm_layers']),
        '--d_model', str(trial_dict['d_model']),
        '--n_heads', str(trial_dict['n_heads']),
        '--d_ff', str(trial_dict['d_ff']),
        '--dropout', str(trial_dict['dropout']),
        '--patch_len', str(trial_dict['patch_len']),
        '--stride', str(trial_dict['stride']),
        '--batch_size', str(trial_dict['batch_size']),
        '--learning_rate', str(trial_dict['learning_rate']),
        '--num_tokens', str(trial_dict['num_tokens']),
        '--loss', trial_dict['loss'],
        '--lradj', trial_dict['lradj'],
        '--pct_start', str(trial_dict['pct_start']),
        '--metric', trial_dict['metric'],
        # Static Parameters
        '--llm_model', llm_model,
        '--data', 'CRYPTEX',
        '--root_path', ".",
        '--data_path', str(data_path),
        '--target', str(trial_dict['target']),
        '--train_epochs', str(trial_dict['epochs']),
        '--experiment_name', trial_dict['experiment_name'],
    ]
    return cmd


def set_optuna_vars(trial, data_path, yaml_file):
    """Sets the optuna variables for the trial into a dictionary.
    The dictionary is then used as arguments for the run_main.py script.
    The values are pulled from the optuna_vars.yaml (or given) 
    file.

    Args:
        trial: Optuna trial object
        data_path: Path to the data
        args: Arguments

    Returns:
        params: Dictionary of trial parameters
    """

    with open(Path("config") / args.yaml_file, "r") as f:
        config = yaml.safe_load(f)

    params = {}

    # Categorical parameters
    for name, values in config.get("categorical", {}).items():
        # If there is only one value, use it twice
        # This is because Optuna requires two values for categorical parameters
        if len(values) == 1:

            params[name] = trial.suggest_categorical(name, values * 2)
        else:
            params[name] = trial.suggest_categorical(name, values)

    # Int parameters
    for name, cfg in config.get("int", {}).items():
        # If step is provided, use it to suggest the int parameter
        if "step" in cfg:
            params[name] = trial.suggest_int(
            name,
            int(cfg["low"]),
            int(cfg["high"]),
            step=int(cfg.get("step", 1))
        )
        else:
            params[name] = trial.suggest_int(
                name,
                int(cfg["low"]),
                int(cfg["high"])
            )

    # Float parameters
    for name, cfg in config.get("float", {}).items():
        # If step is provided, use it to suggest the float parameter
        if "step" in cfg:
            params[name] = trial.suggest_float(
                name,
                float(cfg["low"]),
                float(cfg["high"]),
                step=float(cfg.get("step", 1))
            )
        else:
            # If step is not provided, use the log flag to suggest the float parameter
            params[name] = trial.suggest_float(
                name,
                float(cfg["low"]),
                float(cfg["high"]),
                log=cfg.get("log", False)
            )

    for name, cfg in config.get("log_float", {}).items():
        params[name] = trial.suggest_float(
            name,
            float(cfg["low"]),
            float(cfg["high"])
            )


    params["target"] = "returns" if args.returns else "close"
    params['target'] = "volatility" if args.volatility else params["target"]
    params["metric"] = "SHARPE"
    #params["dates"] = f"{args.start}_{args.end}"
    params["experiment_name"] = args.experiment_name or llm_model

    #trial.set_user_attr("dates", f"{args.start}_{args.end}")
    trial.set_user_attr("granularity", args.granularity)
    trial.set_user_attr("aggregate", args.aggregate)
    trial.set_user_attr("target", params["target"])
    trial.set_user_attr("data_type", "returns" if args.returns else "ohlcv")
    trial.set_user_attr("metric", "SHARPE")
    
    print("--------------------------------\n")
    print("Trial Parameters:")
    for key, value in params.items():
        print(f"{key}: {value}", end=" | ")
    
    print("\n\n--------------------------------")

    return params


def run_pipeline(run, mlflow_client, metrics_db_path, model_id, llm_model, args, trial_dict, experiment_name):
    """
    Runs the pipeline for the model if the inference path is provided.
    It logs the MDA metric for the first candle, the parameters, and the summary table to the metrics database.
    Also logs the summary table to the MLflow run.

    Args:   
        run: MLflow run object
        metrics_db_path: path to the metrics database
        model_id: model id
        llm_model: llm model
        args: arguments
        inf_path: path to the inference data
        trial_dict: dictionary of trial parameters
        experiment_name: experiment name
    """

    inf_save_path = Path("temp")   # Folder name for the inference data
    inf_output_path = Path("temp") / "inference.csv"      # Path to the inference data


    # Checks to run inference if the inference path is provided
    # As well checks if the returns flag is set and converts the data back to candlesticks
    if not INFERENCE:
        return

    try:
        inf_output_path = run_inference(
            model_id = model_id, 
            mlflow_client = mlflow_client,
            experiment_name = experiment_name,
            dataset_path = DATASET_PATH.parent, 
            granularity = args.granularity, 
            aggregate = args.aggregate, 
            start_date = args.inf_start, 
            end_date = args.inf_end, 
            save_path = inf_save_path)

    except Exception as e:
        print(f"\nInference failed - Stopping Pipeline: {e}\n")
        return


    mda_vals = get_mda_vals(inf_output_path)
    
    if args.returns:
        inf_output_path = Path("temp") / "inference_ret.csv"
    else:
        inf_output_path = Path("temp") / "inference.csv"

    mse_vals = get_mse_vals(inf_output_path, pred_len = trial_dict['pred_len'], target = trial_dict['target'])
    rmse_vals = {f"RMSE_{key.split('_')[2]}": round((value) ** 0.5, 6) for key, value in mse_vals.items()}

    # Turns the metrics into dataframes and saves them to the temp folder so they can be logged to the MLflow run as artifacts.
    pd.DataFrame(list[tuple](mse_vals.items()), columns=['metric', 'value']).to_csv(Path("temp") / "mse_metrics.csv", index=False)
    pd.DataFrame(list[tuple](rmse_vals.items()), columns=['metric', 'value']).to_csv(Path("temp") / "rmse_metrics.csv", index=False)

    if args.log_all_metrics:
        mlflow.log_metrics(mda_vals, step = 1, run_id = run.info.run_id)
        mlflow.log_metrics(mse_vals, step = 1, run_id = run.info.run_id)
        mlflow.log_metrics(rmse_vals, step = 1, run_id = run.info.run_id)
    else:
        max_mda = max(mda_vals.values())
        for key, value in mda_vals.items():
            if value == max_mda:
                mlflow.log_metric(key = f"Best Inf MDA", value = value, step = 1, run_id = run.info.run_id)
                mlflow.log_metric(key = f"Best Inf MDA Candle", value = int(key.split("_")[2]), step = 1, run_id = run.info.run_id)
                break

        min_mse = min(mse_vals.values())
        for key, value in mse_vals.items():
            if value == min_mse:
                mlflow.log_metric(key = f"Min Inf MSE", value = round(value, 6), step = 1, run_id = run.info.run_id)
                mlflow.log_metric(key = f"Min Inf MSE Candle", value = int(key.split("_")[2]), step = 1, run_id = run.info.run_id)
                break
        
        min_rmse = {"Min Inf RMSE": round((min_mse) ** 0.5, 6)}
        mlflow.log_metrics(min_rmse, step = 1, run_id = run.info.run_id)



    try:    
        # Saves the MDA metrics to the MLflow as an artifact then removes the file
        mda_path = Path("temp") / "mda_metrics.csv"
        pd.DataFrame(list[tuple](mda_vals.items()), columns=['metric', 'value']).to_csv(mda_path, index=False)
        mlflow.log_artifact(mda_path, run_id = run.info.run_id)

        mse_path = Path("temp") / "mse_metrics.csv"
        pd.DataFrame(list[tuple](mse_vals.items()), columns=['metric', 'value']).to_csv(mse_path, index=False)
        mlflow.log_artifact(mse_path, run_id = run.info.run_id)

        rmse_path = Path("temp") / "rmse_metrics.csv"
        pd.DataFrame(list[tuple](rmse_vals.items()), columns=['metric', 'value']).to_csv(rmse_path, index=False)
        mlflow.log_artifact(rmse_path, run_id = run.info.run_id)

    
    except Exception as e:
        print(f"\nMDA metrics save failed: {e}\n")
    

    # Performs the backtest if the backtest flag is set
    if args.backtest:   
        try:
            perform_backtest(inf_output_path) # Performs backtest
        except Exception as e:
            print(f"\nBacktest failed: \n\n{e}\n")  
        
        summary_table = pd.read_csv(Path("temp") / "summary_table.csv")

        # creates the metrics json
        metrics_json = create_metrics_json(run.info.run_id,llm_model, experiment_name, summary_table, mda_vals, trial_dict)
        # saves the metrics to the database
        try:
            metrics_to_db(metrics_db_path, model_id, metrics_json)
        except sqlite3.Error as e:
            print(f"\nSQLite error: \n\n{e}\n")
        except Exception as e:
            print(f"\nMetrics to database failed: \n\n{e}\n")

        # Logs the summary table to the MLflow run
        mlflow.log_artifact(Path("temp") / "summary_table.csv", run_id = run.info.run_id)


# --- 1. Define the Objective Function ---
# This function defines a single experiment run. Optuna will call it multiple times.
def objective(trial):
    """
    Defines one trial in the Optuna study.
    Optuna will suggest hyperparameter values, which we use to launch run_main.py.
    The function returns the metric we want to optimize (e.g., validation loss).

    It also runs the pipeline for the model if the inference path is provided.
    It logs the MDA metric for the first candle, the parameters, and the summary table to the metrics database.
    Also logs the summary table to the MLflow run.

    Args:
        trial: Optuna trial object

    Returns:
        final_metric: Final metric value
    """

    # Sets the optuna variables
    trial_dict = set_optuna_vars(trial, args.data_path, args)

    # Saves the original data to the DATA_PATH and INF_PATH
    # This is done to avoid using data from previous trials
    org_data_path = Path("temp/org_data.csv")
    pd.read_csv(org_data_path).to_csv(DATA_PATH, index=False)
    if INFERENCE:
        # Saves the original inference data to the INF_PATH
        org_inf_path = Path("temp") / "org_inf_data.csv"
        pd.read_csv(org_inf_path).to_csv(INF_PATH, index=False)  


    # Checks if the returns flag is set
    if args.returns:
        train_path = convert_to_returns(DATA_PATH)

        if INFERENCE:
            convert_to_returns(INF_PATH)
    else:
        train_path = DATA_PATH

    
    # --- Dynamic/Conditional Parameters ---
    # Generate a unique model_id for each trial
    trial_id = str(uuid.uuid4())[:8]
    model_id = f"trial_{trial_id}_{args.granularity}_{args.data_path if args.data_path is not None else 'full'}_dates_{args.start}_{args.end}_features_{trial_dict['features']}_seq_{trial_dict['seq_len']}"

    # Set the experiment name
    experiment_name = trial_dict['experiment_name']

    # --- 4. Run the Trial and Get the Result ---
    # We use MLflow to get the result of the trial.
    # This is more robust than parsing stdout.
    client = mlflow.tracking.MlflowClient()

    
    # We need to find the MLflow run associated with this trial.
    # We'll use the model_id (which includes trial_id) as a unique tag.
    
    try:
        if args.backtest and not INFERENCE:
            warnings.warn("Backtest flag is set but no inference date is provided. - Will not perform backtest.")

        # Creates the command to train the model
        cmd = create_train_cmd(trial_dict, model_id, train_path)
        print(f"\n--- Starting Trial {trial.number} ---\n{' '.join(cmd)}\n")

        # Launch the subprocess
        subprocess.run(cmd, check=True, text=True, capture_output=True)
        # After the run completes, find it in MLflow
        time.sleep(4) # Give MLflow a moment to log everything

        run = _find_mlflow_run(client, experiment_name, model_id)

        client.log_param(run_id = run.info.run_id, key = "granularity", value = args.granularity)
        client.log_param(run_id = run.info.run_id, key = "start date", value = args.start)
        client.log_param(run_id = run.info.run_id, key = "end date", value = args.end)
        client.log_param(run_id = run.info.run_id, key = "aggregate", value = args.aggregate)
        
        
        if not run:
            raise optuna.exceptions.TrialPruned("Could not find MLflow run post-execution.")

        # Get the validation metric from the last recorded step
        latest_metrics = run.data.metrics

        # The key should match what you log in run_main.py
        validation_metric_key = f"vali_{trial_dict['metric'].lower()}_metric" 
        
        if validation_metric_key not in latest_metrics:
            raise optuna.exceptions.TrialPruned(f"Metric '{validation_metric_key}' not found.")
            
        final_metric = latest_metrics[validation_metric_key]
        
        print(f"--- Trial {trial.number} Finished ---")
        
        # This section checks to run inference if the inference path is provided
        # As well checks if the returns flag is set and converts the data back to candlesticks
        run_pipeline(run, client, METRICS_DB_PATH, model_id, llm_model, args, trial_dict, experiment_name)

        # Checks if the validation metric is 0
        if final_metric == 0:
            raise optuna.exceptions.TrialPruned("Validation metric is 0.")
        
        return final_metric


    # Checks if the trial failed due to an error
    except subprocess.CalledProcessError as e:
        print(f"\nTrial {trial.number} failed with error.\n")
        print(e.stderr)
        
        time.sleep(2)
        # --- Error Logging to MLflow ---
        run = _find_mlflow_run(client, experiment_name, model_id)

        if run:
            failed_run_id = run.info.run_id
            full_output = f"--- STDOUT ---\n{e.stdout}\n\n--- STDERR ---\n{e.stderr}"
            client.log_text(failed_run_id, full_output, f"failed_trial_{trial.number}_error.log")
            print(f"--> Error log saved as an artifact to failed MLflow run ID: {failed_run_id}")
            # Finally, set the run status to FAILED
            client.set_terminated(failed_run_id, "FAILED")

        # Tell Optuna this trial failed and should be pruned.
        raise optuna.exceptions.TrialPruned()

def main(   
        gpu = 1, 
        study_name = '', 
        granularity = 'daily', 
        start = None, 
        end = None, 
        inf_start = None, 
        inf_end = None, 
        data_path = None, 
        returns = False, 
        backtest = False, 
        experiment_name = None, 
        trials = 10, 
        aggregate = 1, 
        no_inf_aggregate = False, 
        log_all_metrics = False, 
        yaml_file = 'optuna_vars.yaml', 
        volatility = False):
    # --- 5. Create and Run the Optuna Study ---
    # The 'study_name' will group your runs. If you restart the script, it will resume.
    # 'storage' tells Optuna to save results to a local SQLite database.

    os.makedirs("temp", exist_ok=True)
    org_data_path = Path("temp/org_data.csv")

    if args.gpu != '1': # If the GPU is not 1, uses the NFS server for the storage path
        OPTUNA_STORAGE_PATH = f"sqlite:////mnt/nfs/mlflow/optuna_study.db"
        DATASET_PATH = Path("/mnt/nfs/datasets/")
        METRICS_DB_PATH = f"/mnt/nfs/mlflow/metrics.db"

    INFERENCE = args.inf_start is not None or args.inf_end is not None
        
    # Sets the dataset path based on the granularity argument
    if args.granularity.lower() in ['daily', 'd']:
        DATASET_PATH = DATASET_PATH / "candlesticks-D.csv"
    elif args.granularity.lower() in ['hourly', 'h']:
        DATASET_PATH = DATASET_PATH / "candlesticks-h.csv"
    elif args.granularity.lower() in ['weekly', 'w']:
        DATASET_PATH = DATASET_PATH / "candlesticks-W.csv"
    elif args.granularity.lower() in ['minute', 'min']:
        DATASET_PATH = DATASET_PATH / "candlesticks-Min.csv"

    if args.data_path is None and args.start is None:
        warnings.warn("Data path and start date are not provided - Will start from the beginning of the dataset. If no end date is provided, it will use the entire dataset.")

    print(f"Prepping Data...")

    if args.data_path is not None: # If the data path is provided, uses the data path
        full_data = pd.read_csv(args.data_path)
        inf_data = pd.read_csv(args.data_path)
    else:
        full_data = pd.read_csv(DATASET_PATH)
        inf_data = full_data.copy()

    if args.start: # If the start date is provided, uses the start date
        full_data = full_data[full_data['timestamp'] >= datetime.strptime(args.start, '%Y-%m-%d').timestamp()]

    if args.end: # If the end date is provided, uses the end date
        full_data = full_data[full_data['timestamp'] <= datetime.strptime(args.end, '%Y-%m-%d').timestamp()]

    if args.aggregate: # If the aggregate period is provided, aggregates the data
        full_data = aggregate_data(full_data, args.aggregate)

    full_data.to_csv(org_data_path, index=False)
    
    # If inference is enabled, we need to filter the inference data based on the inference start and end dates
    if INFERENCE:

        if args.inf_start:
            inf_data = inf_data[inf_data['timestamp'] >= datetime.strptime(args.inf_start, '%Y-%m-%d').timestamp()]

        if args.inf_end:
            inf_data = inf_data[inf_data['timestamp'] <= datetime.strptime(args.inf_end, '%Y-%m-%d').timestamp()]
            
        if args.aggregate and not args.no_inf_aggregate:
            inf_data = aggregate_data(inf_data, args.aggregate)

        inf_data.to_csv(Path('temp') / "org_inf_data.csv", index=False)


    if args.study_name == '': # Uses the default study name
        study_name = f"{llm_model.lower()}_study"
    else: # Uses the given study name
        study_name = f"{args.study_name}"

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize", 
        storage=OPTUNA_STORAGE_PATH,
        load_if_exists=True # Resume study if it already exists
    )
    
    # 'n_trials' is the total number of experiments you want to run.
    # Optuna will intelligently choose the parameters for these runs.
    study.optimize(objective, n_trials=args.trials)

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

    if os.path.exists("temp"):
        shutil.rmtree("temp")



if __name__ == "__main__":
    args = parse_args()
    main(args.gpu, args.study_name, args.granularity, args.start, args.end, args.inf_start, args.inf_end, args.data_path, args.returns, args.backtest, args.experiment_name, args.trials, args.aggregate, args.no_inf_aggregate, args.log_all_metrics, args.yaml_file, args.volatility)

    