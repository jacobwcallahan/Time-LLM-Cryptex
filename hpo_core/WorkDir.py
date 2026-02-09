from pathlib import Path
import os
import pandas as pd


class WorkDir:
    """
    This class is responsible for managing the work directory.
    """

    WORK_DIR = Path("temp")
    DATASET_PATH = Path("dataset/candles/")
    OPTUNA_STORAGE_PATH = "sqlite:////data-fast/nfs/mlflow/optuna_study.db" # Optuna storage path

    granularity_map = {
            'daily': 'D',
            'd': 'D',
            'hourly': 'h',
            'h': 'h',
            'weekly': 'W',
            'w': 'W',
            'minute': 'Min',
            'min': 'Min',
            "day": "D",
            "hour": "h",
            "week": "W",
        }


    @staticmethod
    def create_work_dir():
        os.makedirs(WorkDir.WORK_DIR, exist_ok=True)

    @staticmethod
    def ohlcv_train_data_path():
        """
        Returns the path to the original train data sliced by given dates.
        """
        return WorkDir.WORK_DIR / "ohlcv_train_data.csv"

    @staticmethod
    def org_ohlcv_inf_data_path():
        """
        Returns the path to the original inference data sliced by given dates.
        """
        return WorkDir.WORK_DIR / "org_ohlcv_inf_data.csv"

    @staticmethod
    def write_ohlcv_train_data(data: pd.DataFrame):
        WorkDir.create_work_dir()
        data.to_csv(WorkDir.ohlcv_train_data_path(), index=False)

    @staticmethod
    def write_org_ohlcv_inf_data(data: pd.DataFrame):
        WorkDir.create_work_dir()
        data.to_csv(WorkDir.org_ohlcv_inf_data_path(), index=False)

    @staticmethod
    def ret_train_data_path():
        return WorkDir.WORK_DIR / "ret_train_data.csv"

    @staticmethod
    def ret_inf_data_path():
        return WorkDir.WORK_DIR / "ret_inf_data.csv"

    @staticmethod
    def write_ret_train_data(data: pd.DataFrame):
        WorkDir.create_work_dir()
        data.to_csv(WorkDir.ret_train_data_path(), index=False)

    @staticmethod
    def write_ret_inf_data(data: pd.DataFrame):
        WorkDir.create_work_dir()
        data.to_csv(WorkDir.ret_inf_data_path(), index=False)

    @staticmethod
    def inferenced_path():
        """This is the path of the data that has already been inferenced.
        """
        return WorkDir.WORK_DIR / "inference.csv"

    @staticmethod
    def get_full_data_path(granularity: str = "daily"):
        return WorkDir.DATASET_PATH / f"candlesticks-{WorkDir.granularity_map[granularity.lower()]}.csv"

    @staticmethod
    def get_full_data(granularity: str = "daily"):
        return pd.read_csv(WorkDir.get_full_data_path(granularity))
