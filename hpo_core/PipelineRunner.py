from hpo_core.WorkDir import WorkDir
from run_inference import main as run_inference_main
import argparse
from backtesting.backtest import main as backtest_main
from run_main import main as run_training_main
from infra.TrainConfig import TrainConfig
from hpo_core.DataManager import DataManager

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

    def run_inference(self, data_manager: DataManager, experiment_name: str, run_id: str):
        """
        Runs the inference pipeline.

        Args:
            data_manager (DataManager): The data manager object.
            experiment_name (str): The experiment name.
            run_id (str): The run ID.
        """

        # TODO Make this change the dates of the data. As well make it check if the data is already prepared.

        data_manager.prepare_inf_data()
        # TODO: Make this a InferenceConfig object
        inf_args = argparse.ArgumentParser()
        inf_args.model_id = run_id
        inf_args.llm_model = "LLAMA3.1"
        inf_args.data_path = self.work_dir.get_inf_data_path()
        inf_args.mlflow_tracking_uri = None
        inf_args.save_path = self.work_dir.get_work_dir_path()
        inf_args.experiment_name = experiment_name

        run_inference_main(inf_args)

    def run_backtest(self, pipeline: bool = False):
        backtest_args = {
            "data": self.work_dir.get_ohlcv_inferenced_path(),
            "cash": 100000,
            "commission": 0.001,
            "pipeline": pipeline,
            "optimize": False,
            "strategy": None,
            "walk_forward": None,
            }
        backtest_main(backtest_args, summary_table_path = self.work_dir.summary_table_path())
