import argparse
from pathlib import Path




class HpoArgs:
    """
    Stores HPO arguments from CLI and/or programmatic overrides.
    All attributes are accessible directly: args.gpu, args.start, etc.
    """

    llm_model = "LLAMA3.1"
    OPTUNA_STORAGE_PATH = "sqlite:////data-fast/nfs/mlflow/optuna_study.db"
    DATASET_PATH = Path("./dataset/candles/")

    def __init__(self, parse_cli: bool = True, **overrides):
        """
        Args:
            parse_cli: If True, parse sys.argv. If False, start with defaults (useful for programmatic use).
            **overrides: Any CLI arg name as keyword. Overrides CLI values. Use for programmatic args.
        """
        if parse_cli:
            self._args = self._build_parser().parse_args()
        else:
            self._args = self._build_parser().parse_args([])
        for key, value in overrides.items():
            setattr(self._args, key, value) 
        self.INFERENCE = (
            self._args.inf_start is not None and self._args.inf_end is not None
        )
        setattr(self._args, "INFERENCE", self.INFERENCE)

    def __getattr__(self, name):
        """Delegate to underlying args for direct access: args.gpu, args.start, etc."""
        try:
            return getattr(self._args, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """Delegate attribute assignment to underlying args (allows kwargs-style mutation)."""
        if name in ("_args", "INFERENCE"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._args, name, value)

    @property
    def args(self):
        """The underlying argparse Namespace (for backward compatibility)."""
        return self._args

    def _build_parser():
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
        return parser

