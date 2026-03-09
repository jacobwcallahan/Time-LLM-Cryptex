from hpo_core.WorkDir import WorkDir
from run_inference import main as run_inference_main
import argparse
from backtesting.backtest import main as backtest_main
from run_main import main as run_training_main
from infra.TrainConfig import TrainConfig
from hpo_core.DataManager import DataManager
import os
import mlflow

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]

class PipelineRunner:
    """
    This class is responsible for running the pipeline.
    """

    def __init__(self, work_dir: WorkDir):
        self.work_dir = work_dir

    def run_training(self, config: dict):
        work_dir_path = str(self.work_dir.get_work_dir_path())
        defaults = {
            "model_id": "run",
            "seed": 2021,
            "data": "CRYPTEX",
            "root_path": ".",
            "data_path": self.work_dir.get_train_data_path(),
            "features": "MS",
            "target": "close",
            "checkpoints": work_dir_path,
            "seq_len": config['seq_len'],
            "pred_len": config['pred_len'],
            "enc_in": 7,
            "d_model": config['d_model'],
            "n_heads": config['n_heads'],
            "d_ff": config['d_ff'],
            "dropout": config['dropout'],
            "patch_len": config['patch_len'],
            "stride": config['stride'],
            "llm_model": config['llm_model'],
            "batch_size": config['batch_size'],
            "learning_rate": config['learning_rate'],
            "loss": config['loss'],
            "metric": config['metric'],
            "lradj": config['lradj'],
            "pct_start": config['pct_start'],
            "train_epochs": config['epochs'],
            "use_amp": False,
            "llm_layers": config['llm_layers'],
            "num_tokens": config['num_tokens'],
            "enable_mlflow": True,
            "experiment_name": config['experiment_name'],
        }
        training_args = TrainConfig.from_dict({**defaults, **config})
        
        run_training_main(training_args)

    def run_inference(self, experiment_name: str, run_id: str):
        """
        Runs the inference pipeline. Requires DataManager.current() to be set.

        Args:
            experiment_name (str): The MLflow experiment name.
            run_id (str): The MLflow run name / model ID to load.
        """
        # Align data format with model: fetch target from MLflow and set returns/volatility
        target = self.get_target_from_mlflow(
            run_id, experiment_name=experiment_name, tracking_uri=MLFLOW_TRACKING_URI
        )
        self.work_dir.args.returns = target == "returns"
        self.work_dir.args.volatility = target == "volatility"

        data_manager = DataManager.current()
        data_manager.prepare_inf_data(inf_start_date=self.work_dir.args.inf_start,
                                    inf_end_date=self.work_dir.args.inf_end, 
                                    aggregate=self.work_dir.args.aggregate, 
                                    returns=self.work_dir.args.returns)

        inf_args = argparse.Namespace()
        inf_args.model_id = run_id
        inf_args.llm_model = "LLAMA3.1"
        inf_args.data_path = str(self.work_dir.get_inf_data_path())
        inf_args.mlflow_tracking_uri = MLFLOW_TRACKING_URI
        inf_args.save_path = self.work_dir.get_work_dir_path()
        inf_args.experiment_name = experiment_name

        run_inference_main(inf_args)

        # Post-process: convert returns to OHLCV for backtest, or copy inference to ohlcv path
        # (volatility target: no convert_back; backtest may need separate handling)
        if self.work_dir.args.returns:
            inferenced = data_manager.work_dir.get_inferenced_data()
            pred_cols = [c for c in inferenced.columns if "_predicted_" in c and c.split("_")[-1].isdigit()]
            pred_len = len(pred_cols)
            ohlcv_inf = data_manager.convert_back_to_candlesticks(
                pred_len,
                self.work_dir.get_org_ohlcv_inf_data(),
                inferenced,
            )
            self.work_dir.write_ohlcv_inferenced_data(ohlcv_inf)
            self.work_dir.rename_ret_inferenced_data()
        else:
            self.work_dir.rename_ohlcv_inferenced_data()

        # Log inference artifacts to MLflow so Check & Plot Inference and Backtest can use them
        run_uuid, _ = self.get_mlflow_run_info(run_id, experiment_name=experiment_name, tracking_uri=MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        client.log_artifact(
            run_id=run_uuid,
            local_path=str(self.work_dir.get_ohlcv_inferenced_path()),
        )
        if self.work_dir.args.returns:
            ret_path = self.work_dir.get_ret_inferenced_path()
            if ret_path.exists():
                client.log_artifact(run_id=run_uuid, local_path=str(ret_path))

    def run_backtest(
        self,
        pipeline: bool = False,
        data_path: str = None,
        strategy: str = None,
        cash: float = None,
        run_id: str = None,
        experiment_name: str = None,
    ):
        """
        Run backtest on inference results.

        Args:
            pipeline: If True, suppress console output.
            data_path: Override path to inference CSV. Default: work_dir ohlcv_inference.
            strategy: Strategy name to run (e.g. 'SimpleAI'). None = run all.
            cash: Initial capital. Default: 100000.
            run_id: MLflow run name. If provided with experiment_name, log summary_table to MLflow.
            experiment_name: MLflow experiment name. If provided with run_id, log summary_table to MLflow.

        Returns:
            Summary DataFrame from backtest.
        """
        data = data_path if data_path else str(self.work_dir.get_ohlcv_inferenced_path())
        backtest_args = {
            "data": data,
            "cash": cash if cash is not None else 100000,
            "commission": 0.001,
            "pipeline": pipeline,
            "optimize": False,
            "strategy": strategy,
            "walk_forward": None,
        }
        result = backtest_main(
            backtest_args,
            summary_table_path=str(self.work_dir.summary_table_path()),
        )
        if run_id and experiment_name:
            run_uuid, _ = self.get_mlflow_run_info(run_id, experiment_name=experiment_name, tracking_uri=MLFLOW_TRACKING_URI)
            summary_path = self.work_dir.summary_table_path()
            if summary_path.exists():
                client = mlflow.tracking.MlflowClient()
                client.log_artifact(run_id=run_uuid, local_path=str(summary_path))
        return result

    def get_target_from_mlflow(self, model_id: str, experiment_name: str = None, llm_model: str = "LLAMA3.1", tracking_uri: str = None) -> str:
        """Fetch the target param from an MLflow run. Used by pipeline to align inference data format with model."""
        _, params = self.get_mlflow_run_info(model_id, experiment_name, llm_model, tracking_uri)
        return params.get("target", "close")

    def get_mlflow_run_info(self, model_id: str, experiment_name: str = None, llm_model: str = "LLAMA3.1", tracking_uri: str = None):
        """Fetch MLflow run UUID and params. Returns (run_id_uuid, params_dict)."""
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(experiment_name) if experiment_name else client.get_experiment_by_name(llm_model)
        runs = client.search_runs([exp.experiment_id], f"tags.mlflow.runName = '{model_id}'")
        if not runs:
            raise ValueError(f"No MLflow run found with name '{model_id}' in experiment '{experiment_name or llm_model}'")
        run = runs[0]
        params = dict(run.data.params)
        # Cast pred_len to int for CalcMetrics
        if "pred_len" in params:
            params["pred_len"] = int(params["pred_len"])
        return run.info.run_id, params


