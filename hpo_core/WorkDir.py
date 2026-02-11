from pathlib import Path
import os
import pandas as pd
import hpo_core.HpoArgs as HpoArgs

class WorkDir:
    """
    Manages the work directory and dataset paths. Configurable via __init__.
    """

    WORK_DIR = Path("temp")
    DATASET_PATH = Path("dataset/candles/")
    OPTUNA_STORAGE_PATH = "sqlite:////data-fast/nfs/mlflow/optuna_study.db" # Optuna storage path

    granularity_map = {
        "daily": "D",
        "d": "D",
        "hourly": "h",
        "h": "h",
        "weekly": "W",
        "w": "W",
        "minute": "Min",
        "min": "Min",
        "day": "D",
        "hour": "h",
        "week": "W",
    }

    def __init__(
        self,
        args: HpoArgs,
        work_dir: Path = Path("temp"),
        dataset_path: Path = Path("dataset/candles/"),
        optuna_storage_path: str = "sqlite:////data-fast/nfs/mlflow/optuna_study.db",
    ):
    
        self.args = args

        self.work_dir = work_dir
        self.dataset_path = dataset_path
        self.optuna_storage_path = optuna_storage_path

        self.full_data_path = None
        self.set_full_data_path()

        
    def create_work_dir(self):
        os.makedirs(self.work_dir, exist_ok=True)

    def ohlcv_train_data_path(self) -> Path:
        """Path to the original train data sliced by given dates."""
        return self.work_dir / "ohlcv_train_data.csv"

    def org_ohlcv_inf_data_path(self) -> Path:
        """Path to the original inference data sliced by given dates."""
        return self.work_dir / "org_ohlcv_inf_data.csv"

    def write_ohlcv_train_data(self, data: pd.DataFrame):
        self.create_work_dir()
        data.to_csv(self.ohlcv_train_data_path(), index=False)

    def write_org_ohlcv_inf_data(self, data: pd.DataFrame):
        self.create_work_dir()
        data.to_csv(self.org_ohlcv_inf_data_path(), index=False)

    def ret_inf_data_path(self) -> Path:
        return self.work_dir / "ret_inf_data.csv"

    def write_ret_inf_data(self, data: pd.DataFrame):
        self.create_work_dir()
        data.to_csv(self.ret_inf_data_path(), index=False)

    def inferenced_path(self) -> Path:
        """Path of the data that has already been inferenced."""
        return self.work_dir / "inference.csv"

    def train_data_path(self) -> Path:
        """Path to the train data. This could be either the OHLCV or the returns data."""
        return self.work_dir / "train_data.csv"

    def write_train_data(self, data: pd.DataFrame):
        self.create_work_dir()
        data.to_csv(self.train_data_path(), index=False)
    
    def get_full_data_path(self) -> Path:
        if self.full_data_path is None:
            self.set_full_data_path()
        return self.full_data_path

    def set_full_data_path(self):
        if self.args.data_path is not None:
            self.full_data_path = Path(self.args.data_path)
        else:
            self.full_data_path = self.dataset_path / f"candlesticks-{self.granularity_map[self.args.granularity.lower()]}.csv"

    def get_full_data_df(self) -> pd.DataFrame:
        if self.full_data_path is None:
            self.set_full_data_path()
        return pd.read_csv(self.full_data_path)

    def metrics_path(self, metric: str) -> Path:
        return self.work_dir / f"{metric}_metrics.csv"

    def write_metrics(self, metric: str, data: pd.DataFrame):
        data.to_csv(self.metrics_path(metric), index=False)

    def summary_table_path(self) -> Path:
        return self.work_dir / "summary_table.csv"

