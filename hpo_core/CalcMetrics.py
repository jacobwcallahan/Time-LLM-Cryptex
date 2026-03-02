import mlflow
from hpo_core.DataManager import DataManager
from hpo_core.WorkDir import WorkDir
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Optional
from hpo_core.HpoArgs import HpoArgs
from hpo_core.OptunaParams import OptunaParams


class CalcMetrics:  
    """
    This class is responsible for calculating the metrics for the pipeline.

    This must be done after the inference and backtest are completed.
    """

    def __init__(
        self,
        args: HpoArgs,
        data_manager: DataManager,
        work_dir: WorkDir,
        optuna_params: Optional[OptunaParams] = None,
        params: Optional[dict] = None,
    ):
        self.args = args
        self.data_manager = data_manager
        self.work_dir = work_dir
        self._params = params if params is not None else (optuna_params.params if optuna_params else {})

    def calc_metrics(self) -> dict:
        """
        Calculates the metrics for the inference data.

        returns:
            dict: A dictionary containing the MDA, MSE, and MAE values.
        """
        if self.args.INFERENCE and self._params:
            self.mda_vals = self.get_mda_vals(self.work_dir.get_ohlcv_inferenced_path())
            pred_len = self._params.get("pred_len")
            target = self._params.get("target", "close")
            self.mse_vals = self.get_mse_vals(self.work_dir.get_inferenced_path(), pred_len, target=target)
            self.mae_vals = self.get_mae_vals(self.work_dir.get_inferenced_path(), pred_len, target=target)
            return {"mda": self.mda_vals, "mse": self.mse_vals, "mae": self.mae_vals}
        else:
            return {}

    def get_mda_vals(self, inf_path) -> pd.DataFrame:
        """
        Perform analysis on the inference data.
        It can only be used on the OHLCV data.

        args:
            client: mlflow client
            new_data_path: path to the new data
        """
        target = 'close'

        data = pd.read_csv(inf_path)
        pred_len = data.columns.str.contains('predicted').sum()

        mda_vals = {}   
        
        for pred in range(1, pred_len+1):
            pred_col = f'{target}_predicted_{pred}'
            
            if pred_col not in data.columns:
                print(f"Column {pred_col} not found in data.")
                continue
            
            # Find rows where this prediction column has values (not NaN)
            valid_pred_mask = data[pred_col].notna()
            valid_pred_indices = data.index[valid_pred_mask].values
            
            if len(valid_pred_indices) == 0:
                print(f"No valid predictions found in {pred_col}")
                continue
            
            # For each valid prediction, we need to check if we have the future actual value
            # The prediction at index i predicts the value at index i+pred
            valid_comparisons = []
            
            for idx in valid_pred_indices:
                future_idx = idx + pred
                
                # Check if future actual value exists in the dataframe
                if future_idx < len(data) and pd.notna(data[target].iloc[future_idx]):
                    current_actual = data[target].iloc[idx]
                    predicted_value = data[pred_col].iloc[idx]
                    future_actual = data[target].iloc[future_idx]
                    
                    # Calculate directions
                    predicted_direction = predicted_value - current_actual
                    actual_direction = future_actual - current_actual
                    
                    # Check if signs match
                    correct = (predicted_direction * actual_direction) > 0
                    valid_comparisons.append(correct)
            
            if len(valid_comparisons) == 0:
                print(f"No valid comparisons for prediction horizon {pred}")
                continue
            
            # Calculate directional accuracy
            mda = np.mean(valid_comparisons)
            mda_vals[f'inf_pred_{pred}_mda'] = mda

        if len(mda_vals) == 0:
            print(inf_path)
            raise ValueError("No valid MDA values found.")
        
        mda_vals = pd.DataFrame(list[tuple](mda_vals.items()), columns=['prediction', 'value'])
        return mda_vals

    def get_mse_vals(self, inf_path, pred_len, target = 'close') -> pd.DataFrame:
        """
        Get the MSE values for the inference data.

        args:
            inf_path: path to the inference data
            pred_len: number of predictions to get MSE values for
            target: target column
        """
        data = pd.read_csv(inf_path)
        pred_len = data.columns.str.contains('predicted').sum()
        mse_vals = {}

        try:
            errors = {f'pred_{pred}': [] for pred in range(1, pred_len+1)}
            for i in range(len(data) - pred_len):
                row = data.iloc[i]
                if pd.isna(row[f'{target}_predicted_1']):
                    continue
                for pred in range(1, pred_len+1):
                    next_row = data.iloc[i+pred]
                    if pd.notna(row[f'{target}_predicted_{pred}']):
                        error = row[f'{target}_predicted_{pred}'] - next_row[target]
                        sq_error = error ** 2
                        errors[f'pred_{pred}'].append(sq_error)

            for pred in range(1, pred_len+1):
                mse_vals[f'inf_pred_{pred}_mse'] = np.mean(errors[f'pred_{pred}'])
        except Exception as e:
            print(f"Error getting MSE values: {e}")
            raise ValueError(f"Error getting MSE values: {e}")

        mse_vals = pd.DataFrame(list[tuple](mse_vals.items()), columns=['prediction', 'value'])
        return mse_vals

    def get_mae_vals(self, inf_path, pred_len, target = 'close') -> pd.DataFrame:
        """
        Get the MAE values for the inference data.
        """
        data = pd.read_csv(inf_path)
        pred_len = data.columns.str.contains('predicted').sum()
        mae_vals = {}
        
        try:
            errors = {f'pred_{pred}': [] for pred in range(1, pred_len+1)}
            for i in range(len(data) - pred_len):
                row = data.iloc[i]
                if pd.isna(row[f'{target}_predicted_1']):
                    continue
                for pred in range(1, pred_len+1):
                    next_row = data.iloc[i+pred]
                    if pd.notna(row[f'{target}_predicted_{pred}']):
                        error = row[f'{target}_predicted_{pred}'] - next_row[target]
                        abs_error = abs(error)
                        errors[f'pred_{pred}'].append(abs_error)

            for pred in range(1, pred_len+1):
                mae_vals[f'inf_pred_{pred}_mae'] = np.mean(errors[f'pred_{pred}'])
        except Exception as e:
            print(f"Error getting MAE values: {e}")
            raise ValueError(f"Error getting MAE values: {e}")

        mae_vals = pd.DataFrame(list[tuple](mae_vals.items()), columns=['prediction', 'value'])
        return mae_vals

    def calc_and_log_to_mlflow(
        self,
        client: mlflow.tracking.MlflowClient,
        run_id: str,
        log_returns_inference: bool = False,
        summary_table_path: Optional[Path] = "temp/summary_table.csv",
        verbose: bool = True,
    ) -> dict:
        """
        Compute metrics and log everything to MLflow.

        Args:
            client: MLflow client
            run_id: MLflow run UUID to log artifacts to
            log_returns_inference: If True, also log ret_inference.csv (when model uses returns)
            summary_table_path: If provided, log the backtest summary table
            verbose: If True, print metrics to stdout

        Returns:
            dict: Computed metrics (mda, mse, mae)
        """
        metrics_dict = self.calc_metrics()

        # Write metrics to work_dir and log as artifacts
        if metrics_dict:
            for metric_name, data in metrics_dict.items():
                self.work_dir.write_metrics(metric_name, data)
                client.log_artifact(run_id=run_id, local_path=str(self.work_dir.metrics_path(metric_name)))
                if verbose:
                    print(metric_name)
                    print(data)
                    print("--------------------------------")

        # Log inference data (always)
        client.log_artifact(run_id=run_id, local_path=str(self.work_dir.get_ohlcv_inferenced_path()))
        if log_returns_inference and self.work_dir.get_ret_inferenced_path().exists():
            client.log_artifact(run_id=run_id, local_path=str(self.work_dir.get_ret_inferenced_path()))

        # Log summary table if backtest ran
        if summary_table_path is not None and summary_table_path.exists():
            client.log_artifact(run_id=run_id, local_path=str(summary_table_path))

        return metrics_dict


