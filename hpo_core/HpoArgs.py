import argparse
from pathlib import Path
from types import SimpleNamespace


# Declarative CLI arg definitions. Add new args here; they become both CLI flags and assignable attributes.
# Format: (dest, type_or_action, default, help)
# For action='store_true', use (dest, 'store_true', False, help)
ARG_DEFINITIONS = [
    ('gpu', str, '1', 'If not GPU 1, changes OPTUNA_STORAGE_PATH.'),
    ('study_name', str, '', 'If not empty, uses the study name. Model name is added to the beginning of the study name.'),
    ('granularity', str, 'daily', 'Granularity to use. daily, hourly, weekly, minute'),
    ('start', str, None, 'Start date to use. Format: YYYY-MM-DD'),
    ('end', str, None, 'End date to use. Format: YYYY-MM-DD'),
    ('inf_start', str, None, 'Start date to use for inference. Format: YYYY-MM-DD'),
    ('inf_end', str, None, 'End date to use for inference. Format: YYYY-MM-DD'),
    ('data_path', str, None, 'Data path to use. (Optional, if not provided, uses the full daily dataset)'),
    ('returns', 'store_true', False, 'If True, converts the data to returns.'),
    ('backtest', 'store_true', False, 'If set, run backtest after training'),
    ('experiment_name', str, None, 'Experiment name to use. Default is None.'),
    ('trials', int, 10, 'Number of trials to run.'),
    ('aggregate', int, 1, 'If set, aggregates from the original granularity to the specified granularity.'),
    ('no_inf_aggregate', 'store_true', False, 'By default, aggregates inference data. Set this flag to disable aggregation.'),
    ('log_all_metrics', 'store_true', False, 'By default, logs only the best metric to MLflow. Set this flag to log all metrics (still logs as artifacts).'),
    ('yaml_file', str, 'optuna_vars.yaml', 'YAML file to use for the study. Default is optuna_vars.yaml. Contained in ./config/'),
    ('model_id_name', str, None, 'Name to use for the model id, trail number is added to the end. Default is set by a series of parameters.'),
    ('volatility', 'store_true', False, 'If True, uses the volatility target.'),
]


class HpoArgs:
    """
    Stores HPO arguments from CLI and/or programmatic overrides.
    All attributes are accessible directly: args.gpu, args.start, etc.
    Missing attributes return None. You can assign freely: args.inf_start = "2024-01-01"
    """

    llm_model = "LLAMA3.1"
    OPTUNA_STORAGE_PATH = "sqlite:////data-fast/nfs/mlflow/optuna_study.db"
    DATASET_PATH = Path("./dataset/candles/")

    def __init__(self, parse_cli: bool = True, **overrides):
        """
        Args:
            parse_cli: If True, parse sys.argv. If False, start with defaults (useful for programmatic use).
            **overrides: Any arg name as keyword. Overrides CLI values. Use for programmatic args.
        """
        if parse_cli:
            self._store = self._build_parser().parse_args()
        else:
            self._store = SimpleNamespace(**{dest: default for dest, _, default, _ in ARG_DEFINITIONS})
        for key, value in overrides.items():
            setattr(self._store, key, value)

    def __getattr__(self, name):
        """Delegate to underlying store. Missing attributes return None (no errors)."""
        if name == "_store":
            raise AttributeError  # Let Python raise for _store before init
        return getattr(self._store, name, None)

    def __setattr__(self, name, value):
        """Delegate assignment to underlying store. Supports arbitrary attributes."""
        if name == "_store":
            object.__setattr__(self, name, value)
        else:
            setattr(self._store, name, value)

    @property
    def INFERENCE(self) -> bool:
        """True if inference date range is set (inf_start or inf_end)."""
        return (
            getattr(self._store, "inf_start", None) is not None
            or getattr(self._store, "inf_end", None) is not None
        )

    @property
    def args(self):
        """The underlying namespace (for backward compatibility)."""
        return self._store

    def _build_parser(self):
        parser = argparse.ArgumentParser()
        for dest, typ, default, help_text in ARG_DEFINITIONS:
            if typ == "store_true":
                parser.add_argument(f"--{dest}", dest=dest, action="store_true", default=default, help=help_text)
            else:
                parser.add_argument(f"--{dest}", dest=dest, type=typ, default=default, help=help_text)
        return parser

