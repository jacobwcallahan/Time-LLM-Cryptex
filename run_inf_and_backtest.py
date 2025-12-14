"""
Runs the inference and backtest for a given model and experiment. Logs the metrics to the MLflow server.
Arguments:
    --model_name: Model Name
    --experiment_name: Experiment Name
    --granularity: Granularity default is daily
    --aggregate: Aggregate default is 1
    --start_date: Start Date default is None
    --end_date: End Date default is None
    --dataset_path: Dataset Path default is /mnt/nfs/datasets/
    --save_path: Save Path default is temp folder
"""

from utils.pipeline import run_inference, perform_backtest, get_mda_vals
import argparse
import mlflow
import os
import pandas as pd
from pathlib import Path

os.environ["MLFLOW_TRACKING_URI"] = f"http://192.168.1.103:5000" # Assumes the server is running. Can set to "" to save locally

# MinIO
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://192.168.1.103:9000"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Model Name')
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment Name')
    parser.add_argument('--granularity', type=str, required=False, default='daily', help='Granularity default is daily')
    parser.add_argument('--aggregate', type=int, required=False, default=1, help='Aggregate default is 1')
    parser.add_argument('--start_date', type=str, required=False, default=None, help='Start Date default is None')
    parser.add_argument('--end_date', type=str, required=False, default=None, help='End Date default is None')
    parser.add_argument('--dataset_path', type=str, required=False, default='/mnt/nfs/datasets/', help='Dataset Path default is /mnt/nfs/datasets/')
    parser.add_argument('--save_path', type=str, required=False, default='temp', help='Save Path default is temp folder')
    return parser.parse_args()

def run_inference_and_backtest(model_name, experiment_name, granularity, aggregate, start_date, end_date, dataset_path,custom_dataset_path, save_path, asset = None):
    client = mlflow.tracking.MlflowClient()
    inf_output_path = run_inference(model_id = model_name, 
                                    mlflow_client = client, 
                                    experiment_name = experiment_name, 
                                    dataset_path = dataset_path, 
                                    custom_dataset_path = custom_dataset_path,
                                    granularity = granularity, 
                                    aggregate = aggregate, 
                                    start_date = start_date, 
                                    end_date = end_date, 
                                    save_path = save_path)
    
    mlflow.log_artifact(inf_output_path, artifact_path="inference")

    mda_vals = get_mda_vals(inf_output_path)
    for metric, value in mda_vals.items():
        print(f"{metric}: {value}")
    try:
        if asset is not None:
            mda_vals_path = Path(save_path) / f"{asset}_mda_vals.csv"
        else:
            mda_vals_path = Path(save_path) / "mda_vals.csv"
        pd.DataFrame(list[tuple](mda_vals.items()), columns=['metric', 'value']).to_csv(mda_vals_path, index=False)
        mlflow.log_artifact(mda_vals_path, artifact_path=f"{asset}_mda_metrics" if asset is not None else "mda_metrics")
    except Exception as e:
        print(f"\nMDA metrics save failed: {e}\n")

    perform_backtest(inf_output_path, save_path = save_path)
    if asset is not None:
        summary_table_path = Path(save_path) / f"summary_table.csv"
    else:
        summary_table_path = Path(save_path) / "summary_table.csv"
    summary_table = pd.read_csv(summary_table_path)
    print(summary_table)
    summary_table.to_csv(summary_table_path, index=False)
    mlflow.log_artifact(summary_table_path, 
                        artifact_path=f"{asset}_summary_table" if asset is not None else "summary_table",
                        run_id = model_name)

if __name__ == "__main__":
    args = parse_args()
    run_inference_and_backtest(
        args.model_name, 
        args.experiment_name, 
        args.granularity, 
        args.aggregate, 
        args.start_date, 
        args.end_date, 
        args.dataset_path, 
        args.custom_dataset_path, 
        args.save_path, 
        args.asset)