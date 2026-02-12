from hpo_core.WorkDir import WorkDir
from run_inference import main as run_inference_main
import argparse
from backtesting.backtest import main as backtest_main
from run_main import main as run_training_main

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
            "root_path": work_dir_path,
            "data_path": "train_data.csv",
            "features": "MS",
            "target": "close",
            "checkpoints": work_dir_path,
            "seq_len": 96,
            "pred_len": 96,
            "enc_in": 7,
            "d_model": 16,
            "n_heads": 8,
            "d_ff": 32,
            "dropout": 0.1,
            "patch_len": 16,
            "stride": 8,
            "llm_model": "LLAMA3.1",
            "num_workers": 10,
            "train_epochs": 10,
            "batch_size": 32,
            "eval_batch_size": 8,
            "patience": 10,
            "learning_rate": 0.0001,
            "loss": "MSE",
            "metric": "MAE",
            "lradj": "type1",
            "pct_start": 0.2,
            "use_amp": False,
            "llm_layers": 6,
            "percent": 100,
            "num_tokens": 1000,
            "enable_mlflow": True,
            "experiment_name": None,
        }
        training_args = argparse.Namespace(**{**defaults, **config})
        
        run_training_main(training_args)

    def run_inference(self, experiment_name: str, run_id: str):
        inf_args = argparse.ArgumentParser()
        inf_args.model_id = run_id
        inf_args.llm_model = "LLAMA3.1"
        inf_args.data_path = self.work_dir.get_inf_data_path()
        inf_args.mlflow_tracking_uri = None
        inf_args.save_path = self.work_dir.get_work_dir_path()
        inf_args.experiment_name = experiment_name

        run_inference_main(inf_args)

    def run_backtest(self):
        backtest_args = {
            "data": self.work_dir.get_inf_data_path(),
            "cash": 100000,
            "commission": 0.001,
            "pipeline": True,
        }
        backtest_main(backtest_args)
