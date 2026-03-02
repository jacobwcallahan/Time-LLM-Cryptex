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

import optuna

import subprocess
import mlflow
import uuid
import time
import os

from pathlib import Path

import shutil

from hpo_core.DataManager import DataManager
from hpo_core.WorkDir import WorkDir
from hpo_core.HpoArgs import HpoArgs
from hpo_core.OptunaParams import OptunaParams
from hpo_core.PipelineRunner import PipelineRunner
from hpo_core.CalcMetrics import CalcMetrics
from hpo_core.MLFlowArtifacts import MLFlowArtifacts

# --- Centralized Configuration ---
MLFLOW_SERVER_IP = "192.168.1.103"
# MLflow
os.environ["MLFLOW_TRACKING_URI"] = f"http://{MLFLOW_SERVER_IP}:5000" # Assumes the server is running. Can set to "" to save locally

# MinIO
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{MLFLOW_SERVER_IP}:9000"

llm_model = "LLAMA3.1"
DATASET_PATH = Path("./dataset/candles/") # Dataset path for gpu1 (without specific dataset)
DATA_PATH = Path("temp/data.csv") # Data path in temp folder
INF_PATH = Path("temp/inf_data.csv") # Inference path in temp folder
ARGS = {"gpu": 1, 
        "study_name": '', 
        "granularity": 'daily', 
        "start": None, 
        "end": None, 
        "inf_start": None, 
        "inf_end": None, 
        "data_path": None, 
        "returns": False, 
        "backtest": False, 
        "experiment_name": None, 
        "trials": 10, 
        "aggregate": 1, 
        "no_inf_aggregate": False, 
        "log_all_metrics": False, 
        "yaml_file": 'optuna_vars.yaml', 
        "model_id_name": None,
        "volatility": False}


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


# --- 1. Define the Objective Function ---
# This function defines a single experiment run. Optuna will call it multiple times.
def objective(trial, args: HpoArgs, data_manager: DataManager, work_dir: WorkDir):
    """
    Defines one trial in the Optuna study.
    Optuna will suggest hyperparameter values, which we use to launch run_main.py.
    The function returns the metric we want to optimize (e.g., validation loss).

    It also runs the pipeline for the model if the inference path is provided.
    Also logs the summary table to the MLflow run.

    Args:
        trial: Optuna trial object

    Returns:
        final_metric: Final metric value
    """

    # Sets the optuna variables
    try:
        optuna_params = OptunaParams(trial, args, work_dir)
        trial_dict = optuna_params.get_params()
    except ValueError as e:
        if "CategoricalDistribution does not support dynamic value space." in str(e):
            print(f"Error: {e}")
            print("Have you run new parameters for the same Optuna trial? \n\nEvery Optuna trial requires the same set of parameters\n")
        else:
            print(f"{e}")
        raise exit(1)

    
    # --- Dynamic/Conditional Parameters ---
    # Generate a unique model_id for each trial
    trial_id = str(uuid.uuid4())[:8]
    if args.model_id_name is not None and args.model_id_name != "None":
        print(f"Using provided model id name: {args.model_id_name}")
        model_id = args.model_id_name + f"_trial_{trial_id}"
    else:
        model_id = f"trial_{trial_id}_{args.granularity}_{args.data_path if args.data_path is not None else 'full'}_dates_{args.start}_{args.end}_features_{trial_dict['features']}_seq_{trial_dict['seq_len']}"

    print(f"Trial dictionary: {trial_dict}\n\n")
    
    # Set the experiment name
    trial_dict["experiment_name"] = args.experiment_name
    trial_dict["model_id"] = model_id
    trial_dict["data_path"] = work_dir.get_train_data_path()

    # --- 4. Run the Trial and Get the Result ---
    # We use MLflow to get the result of the trial.
    # This is more robust than parsing stdout.
    client = mlflow.tracking.MlflowClient()
    
    # We need to find the MLflow run associated with this trial.
    # We'll use the model_id (which includes trial_id) as a unique tag.
    
    try:
        # Creates the command to train the model
        pipeline_runner = PipelineRunner(work_dir)
        pipeline_runner.run_training(trial_dict)
        #cmd = create_train_cmd(trial_dict, model_id, work_dir.get_train_data_path())

        
        #print(f"\n--- Starting Trial {trial.number} ---\n{' '.join(cmd)}\n")

        # Launch the subprocess
        #subprocess.run(cmd, check=True, text=True, capture_output=True)
        # After the run completes, find it in MLflow
        time.sleep(4) # Give MLflow a moment to log everything

        print(f"Finding MLflow run for trial {trial.number}")
        run = _find_mlflow_run(client, trial_dict["experiment_name"], trial_dict["model_id"])

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
        pipeline_runner = PipelineRunner(work_dir)
        if args.INFERENCE:
            pipeline_runner.run_inference(data_manager = data_manager, experiment_name = trial_dict["experiment_name"], run_id = model_id)

            mlflow_artifacts = MLFlowArtifacts(run.info.run_id, client, work_dir)

            if args.returns:
                ohlcv_inf_data = data_manager.convert_back_to_candlesticks(optuna_params.params['pred_len'], work_dir.get_org_ohlcv_inf_data(), work_dir.get_inferenced_data())
                work_dir.write_ohlcv_inferenced_data(ohlcv_inf_data)
                work_dir.rename_ret_inferenced_data()
                mlflow_artifacts.log_inference_data(work_dir.get_ret_inferenced_path())
            else:
                work_dir.rename_ohlcv_inferenced_data()
            
            mlflow_artifacts.log_inference_data(work_dir.get_ohlcv_inferenced_path())

            calc_metrics = CalcMetrics(args, data_manager, work_dir, optuna_params)

            metrics_dict = calc_metrics.calc_metrics()
            mlflow_artifacts.log_all_metrics(metrics_dict)

            for metric, data in metrics_dict.items():
                print(metric)
                print(data)
                print("--------------------------------")

            if args.backtest:
                pipeline_runner.run_backtest(pipeline = True)
                mlflow_artifacts.log_summary_table(work_dir.summary_table_path())


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
        run = _find_mlflow_run(client, trial_dict["experiment_name"], trial_dict["model_id"])

        if run:
            failed_run_id = run.info.run_id
            full_output = f"--- STDOUT ---\n{e.stdout}\n\n--- STDERR ---\n{e.stderr}"
            client.log_text(failed_run_id, full_output, f"failed_trial_{trial.number}_error.log")
            print(f"--> Error log saved as an artifact to failed MLflow run ID: {failed_run_id}")
            # Finally, set the run status to FAILED
            client.set_terminated(failed_run_id, "FAILED")

        # Tell Optuna this trial failed and should be pruned.
        raise optuna.exceptions.TrialPruned()

def main(args: HpoArgs):
   
    # --- 5. Create and Run the Optuna Study ---
    # The 'study_name' will group your runs. If you restart the script, it will resume.
    # 'storage' tells Optuna to save results to a local SQLite database.

    work_dir = WorkDir(args)
    work_dir.create_work_dir()

    if args.gpu != '1': # If the GPU is not 1, uses the NFS server for the storage path
        OPTUNA_STORAGE_PATH = f"sqlite:////mnt/nfs/mlflow/optuna_study.db"
        #** DATASET_PATH = Path("/mnt/nfs/datasets/")
        #** ignored while the /mnt nfs is not mounted
    else:
        OPTUNA_STORAGE_PATH = "sqlite:////data-fast/nfs/mlflow/optuna_study.db"
    

    print(f"Prepping Data...")
    data_manager = DataManager(args, work_dir=work_dir)
    data_manager.prepare_train_data()
    data_manager.prepare_inf_data()

    if args.study_name == '': # Uses the default study name
        study_name = f"{llm_model.lower()}_study"
    else: # Uses the given study name
        study_name = f"{args.study_name}"

    print(f"Study name: {study_name}")

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize", 
        storage=OPTUNA_STORAGE_PATH,
        load_if_exists=True # Resume study if it already exists
    )
    
    print(f"Data is Prepped... Starting HPO...")
    
    # 'n_trials' is the total number of experiments you want to run.
    # Optuna will intelligently choose the parameters for these runs.
    study.optimize(
        lambda trial: objective(trial, args, data_manager, work_dir),
        n_trials=args.trials,
    )

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
    args = HpoArgs(parse_cli=True)
    main(args)