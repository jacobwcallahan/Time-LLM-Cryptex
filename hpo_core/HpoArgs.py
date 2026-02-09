import argparse
from pathlib import Path

class HpoArgs:
    """
    This class is responsible for storing the hpo arguments.
    """

    llm_model = "LLAMA3.1"
    OPTUNA_STORAGE_PATH = "sqlite:////data-fast/nfs/mlflow/optuna_study.db" # Optuna storage path
    DATASET_PATH = Path("./dataset/candles/") # Dataset path for gpu1 (without specific dataset)

    

    def __init__(self):
        self.args = self.get_parser()
        
    def get_parser(self):
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
        parser.add_argument('--model_id_name', type=str, default=None, help='Name to use for the model id, trail number is added to the end. Default is set by a series of parameters.')
        parser.add_argument('--volatility', action='store_true', help='If True, uses the volatility target.')
        return parser.parse_args()

    @property
    def args(self):
        return self.args

    
  