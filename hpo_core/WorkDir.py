from pathlib import Path
import os
import pandas as pd
import hpo_core.HpoArgs as HpoArgs
import yaml
import shutil

class WorkDir:
    """
    Manages the work directory and dataset paths. Configurable via __init__.
    """

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

    # ------------------------ Work Directory ------------------------
    def create_work_dir(self):
        os.makedirs(self.work_dir, exist_ok=True)

    def get_work_dir_path(self) -> Path:
        return self.work_dir

    # ------------------------ YAML File ------------------------
    def get_yaml_file_path(self) -> Path:
        if self.args.yaml_file.endswith(".yaml"):
            yaml = self.args.yaml_file
        else:
            yaml = f"{self.args.yaml_file}.yaml"

        return Path("config") / "yaml_params" / yaml

    def get_yaml_file(self) -> dict:
        with open(self.get_yaml_file_path(), "r") as f:
            return yaml.safe_load(f)

    # ------------------------ Train Data ------------------------
    def get_train_data_path(self) -> Path:
        """Path to the train data. This could be either the OHLCV or the returns data."""
        return self.work_dir / "train_data.csv"

    def write_train_data(self, data: pd.DataFrame):
        data.to_csv(self.get_train_data_path(), index=False)

    # ------------------------ Inference Data ------------------------
    def get_inf_data_path(self) -> Path:
        return self.work_dir / "inf_data.csv"

    def write_inf_data(self, data: pd.DataFrame):
        data.to_csv(self.get_inf_data_path(), index=False)

    # ------------------------ OHLCV Train Data ------------------------
    def get_ohlcv_train_data_path(self) -> Path:
        """Path to the original train data sliced by given dates."""
        return self.work_dir / "ohlcv_train_data.csv"

    def write_ohlcv_train_data(self, data: pd.DataFrame):
        self.create_work_dir()
        data.to_csv(self.get_ohlcv_train_data_path(), index=False)

    # ------------------------ OHLCV Inference Data ------------------------
    def get_org_ohlcv_inf_data_path(self) -> Path:
        """Path to the original inference data sliced by given dates."""
        return self.work_dir / "org_ohlcv_inf_data.csv"

    def write_org_ohlcv_inf_data(self, data: pd.DataFrame):
        data.to_csv(self.get_org_ohlcv_inf_data_path(), index=False)

    def get_org_ohlcv_inf_data(self) -> pd.DataFrame:
        return pd.read_csv(self.get_org_ohlcv_inf_data_path())

    # ------------------------ Returns Train Data ------------------------
    def get_ret_train_data_path(self) -> Path:
        return self.work_dir / "ret_train_data.csv"

    def write_ret_train_data(self, data: pd.DataFrame):
        data.to_csv(self.get_ret_train_data_path(), index=False)

    # ------------------------ Returns Inference Data ------------------------
    def get_ret_inf_data_path(self) -> Path:
        return self.work_dir / "ret_inf_data.csv"

    def write_ret_inf_data(self, data: pd.DataFrame):
        data.to_csv(self.get_ret_inf_data_path(), index=False)

    # ------------------------ Inferenced Data ------------------------
    def get_inferenced_path(self) -> Path:
        """Path of the data that has already been inferenced."""
        return self.work_dir / "inference.csv"

    def get_inferenced_data(self) -> pd.DataFrame:
        return pd.read_csv(self.get_inferenced_path())

    # ------------------------ OHLCV Inferenced Data ------------------------
    def get_ohlcv_inferenced_path(self) -> Path:
        """Path of the data that has already been inferenced in OHLCV format."""
        return self.work_dir / "ohlcv_inference.csv"

    def write_ohlcv_inferenced_data(self, data: pd.DataFrame):
        data.to_csv(self.get_ohlcv_inferenced_path(), index=False)

    def get_ohlcv_inf_data(self) -> pd.DataFrame:
        return pd.read_csv(self.get_ohlcv_inferenced_path())

    # ------------------------ Returns Inferenced Data ------------------------
    def get_ret_inferenced_path(self) -> Path:
        """Path of the data that has already been inferenced in returns format."""
        return self.work_dir / "ret_inference.csv"

    def write_ret_inferenced_data(self, data: pd.DataFrame):
        data.to_csv(self.get_ret_inferenced_path(), index=False)
    
    def get_ret_inf_data(self) -> pd.DataFrame:
        return pd.read_csv(self.get_ret_inferenced_path())

    def rename_ret_inferenced_data(self):
        """ Copies the inferenced data to the returns inferenced data path. 
            This is to log the returns inferenced data to the MLflow run."""
        shutil.copy(self.get_inferenced_path(), self.get_ret_inferenced_path())

    def rename_ohlcv_inferenced_data(self):
        """ Copies the inferenced data to the OHLCV inferenced data path. 
            This is to log the OHLCV inferenced data to the MLflow run."""
        shutil.copy(self.get_inferenced_path(), self.get_ohlcv_inferenced_path())

    # ------------------------ Full Data ------------------------
    def set_full_data_path(self):
        if self.args.data_path is not None:
            self.full_data_path = Path(self.args.data_path)
        else:
            self.full_data_path = self.dataset_path / f"candlesticks-{self.granularity_map[self.args.granularity.lower()]}.csv"

    def get_full_data_path(self) -> Path:
        if self.full_data_path is None:
            self.set_full_data_path()
        return self.full_data_path

    def get_full_data(self) -> pd.DataFrame:
        if self.full_data_path is None:
            self.set_full_data_path()
        return pd.read_csv(self.full_data_path)

    # ------------------------ Metrics ------------------------
    def metrics_path(self, metric: str) -> Path:
        return self.work_dir / f"{metric}_metrics.csv"

    def write_metrics(self, metric: str, data: pd.DataFrame | dict):
        if not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data)
            except Exception as e:
                print(f"Data is not a dictionary: {data}")
                print(f"Error writing metrics: {e}")
                raise ValueError(f"Error writing metrics: {e}")
        data.to_csv(self.metrics_path(metric), index=False)

    # ------------------------ Summary Table ------------------------
    def summary_table_path(self) -> Path:
        return self.work_dir / "summary_table.csv"
    

