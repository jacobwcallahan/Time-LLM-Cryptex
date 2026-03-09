"""
This class is responsible for managing the train and inference data. 
"""

from pathlib import Path
from typing import Optional
import pandas as pd
from datetime import datetime, timezone
import warnings
from hpo_core.WorkDir import WorkDir
import numpy as np
from hpo_core.HpoArgs import HpoArgs
import os

class DataManager:
    _current = None

    def __init__(self, work_dir: WorkDir):
        self.work_dir = work_dir

        self.full_data = work_dir.get_full_data()

        self.first_time = datetime.fromtimestamp(int(self.full_data.iloc[0]['timestamp'])).timestamp()
        self.last_time = datetime.fromtimestamp(int(self.full_data.iloc[-1]['timestamp'])).timestamp()
        
        self.data = None
        self.inf_data = None

        DataManager._current = self

    
    @classmethod
    def current(cls) -> "DataManager":
        if cls._current is None:
            raise RuntimeError("DataManager not initialized. Call DataManager(work_dir) first.")
        return cls._current

        
    def prepare_train_data(self, start: str, end: str, aggregate: int, returns: bool):
        """
        Prepares the data for training and inference.
        """
        start_ts, end_ts = self.check_date_validity(start, end)
        self.data = self.full_data[(self.full_data['timestamp'] >= start_ts) & (self.full_data['timestamp'] <= end_ts)]

        self.data = self.aggregate_data(self.data, aggregate)

        # Write the OHLCV train data to the work directory
        self.work_dir.write_ohlcv_train_data(self.data)

        # If the data is returns, convert it to returns and save it
        if returns:
            self.data = self.convert_to_returns(self.data)
            self.work_dir.write_ret_train_data(self.data)

        self.work_dir.write_train_data(self.data)

    def prepare_inf_data(self, 
                        inf_start_date: Optional[str] = None, 
                        inf_end_date: Optional[str] = None, 
                        aggregate: Optional[int] = None, 
                        returns: Optional[bool] = None):
        """
        Prepare inference data. When all params are None, reads from work_dir.args.
        """
        args = self.work_dir.args
        inf_start_date = inf_start_date if inf_start_date is not None else args.inf_start
        inf_end_date = inf_end_date if inf_end_date is not None else args.inf_end
        aggregate = aggregate if aggregate is not None else (1 if args.no_inf_aggregate else args.aggregate)
        returns = returns if returns is not None else args.returns

        # Use full dataset range when dates not set (standalone inference without training)
        if inf_start_date is None and inf_end_date is None:
            # When both are None, use actual min/max from data to avoid any date conversion issues
            ts_min = self.full_data['timestamp'].min()
            ts_max = self.full_data['timestamp'].max()
            self.inf_start_date = float(ts_min)
            self.inf_end_date = float(ts_max)
            self.inf_data = self.full_data[(self.full_data['timestamp'] >= self.inf_start_date) & (self.full_data['timestamp'] <= self.inf_end_date)]
        else:
            if inf_start_date is None:
                inf_start_date = datetime.fromtimestamp(self.first_time, tz=timezone.utc).strftime('%Y-%m-%d')
                warnings.warn(f"No inf_start provided. Using first date of dataset: {inf_start_date}")
            if inf_end_date is None:
                inf_end_date = datetime.fromtimestamp(self.last_time, tz=timezone.utc).strftime('%Y-%m-%d')
                warnings.warn(f"No inf_end provided. Using last date of dataset: {inf_end_date}")

            # Use UTC for timestamp conversion so Unix timestamps in data match the filter range
            self.inf_start_date = datetime.strptime(inf_start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()
            self.inf_end_date = datetime.strptime(inf_end_date + ' 23:59:59', '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
            self.inf_data = self.full_data[(self.full_data['timestamp'] >= self.inf_start_date) & (self.full_data['timestamp'] <= self.inf_end_date)]
        self.inf_data = self.aggregate_data(self.inf_data, aggregate)

        self.work_dir.write_org_ohlcv_inf_data(self.inf_data)
        if returns:
            self.inf_data = self.convert_to_returns(self.inf_data)
            self.work_dir.write_ret_inf_data(self.inf_data)
        elif getattr(args, "volatility", False):
            self.inf_data = self.convert_to_volatility(self.inf_data)
        self.work_dir.write_inf_data(self.inf_data) 

    
    def check_date_validity(self, start: Optional[str], end: Optional[str]):
        """
        Checks the validity of the start and end dates for either the training data or the inference data.
        """
        if start is None:
            start = self.first_time
            warnings.warn(f"No start date provided. Using the first date of the dataset: {datetime.fromtimestamp(self.first_time).date()}")
        else:
            start = datetime.strptime(start, '%Y-%m-%d').timestamp()

        if end is None:
            end = self.last_time
            warnings.warn(f"No end date provided. Using the last date of the dataset: {datetime.fromtimestamp(self.last_time).date()}\n")
        else:
            end = datetime.strptime(end, '%Y-%m-%d').timestamp()
        
        if start > end:
            warnings.warn(f"The start date given ({start}) is after the end date given ({end}). Using the last date of the dataset: {datetime.fromtimestamp(self.last_time).date()} as the end date.\n")
            end = self.last_time

        if start < self.first_time:
            warnings.warn(f"The start date given ({start}) is before the first date of the dataset. Using the first date of the dataset: {datetime.fromtimestamp(self.first_time).date()}\n")
            start = self.first_time
        elif start > self.last_time:
            warnings.warn(f"The start date given ({start}) is after the last date of the dataset. Using the first date of the dataset: {datetime.fromtimestamp(self.first_time).date()}\n")
            start = self.first_time

        if end > self.last_time:
            warnings.warn(f"The end date given ({end}) is after the last date of the dataset. Using the last date of the dataset: {datetime.fromtimestamp(self.last_time).date()}\n")
            end = self.last_time    
        elif end < self.first_time:
            warnings.warn(f"The end date given ({end}) is before the first date of the dataset. Using the last date of the dataset: {datetime.fromtimestamp(self.last_time).date()}\n")
            end = self.last_time

        return start, end

    def check_date_compatibility(self, start: Optional[str], end: Optional[str], inf_start: Optional[str] = None, inf_end: Optional[str] = None):
        """
        Checks the compatibility of the start and end dates for the training data and the inference data.
        Returns (start_ts, end_ts, inf_start_ts, inf_end_ts) as unix timestamps.
        """
        def _to_ts(s):
            if s is None:
                return None
            return s if isinstance(s, (int, float)) else datetime.strptime(s, '%Y-%m-%d').timestamp()

        start_ts = _to_ts(start)
        end_ts = _to_ts(end)
        inf_start_ts = _to_ts(inf_start)
        inf_end_ts = _to_ts(inf_end)

        if start_ts is not None and end_ts is not None:
            if start_ts > end_ts:
                warnings.warn(f"The start date given ({start}) is after the end date given ({end}). Using the last date of the dataset: {datetime.fromtimestamp(self.last_time).date()} as the end date.\n")
                end_ts = self.last_time

        if inf_start_ts is not None and inf_end_ts is not None:
            if inf_start_ts > inf_end_ts:
                warnings.warn(f"The inference start date given ({inf_start}) is after the inference end date given ({inf_end}). Using the last date of the dataset: {datetime.fromtimestamp(self.last_time).date()} as the inference end date.\n")
                inf_end_ts = self.last_time

        if start_ts is not None and end_ts is not None and inf_start_ts is not None and inf_end_ts is not None:
            if end_ts > inf_start_ts:
                warnings.warn(f"The end date given ({end}) is after the inference start date given ({inf_start}). Using the end date as the inference start date.\n")
                inf_start_ts = end_ts

        if end_ts is not None and inf_start_ts is None:
            warnings.warn(f"No inference start date provided. Using the end date as the inference start date: {datetime.fromtimestamp(end_ts).date()}\n")
            inf_start_ts = end_ts

        if inf_end_ts is not None and inf_start_ts is None:
            warnings.warn(f"No inference end date provided. Using the last date of the dataset as the inference end date: {datetime.fromtimestamp(self.last_time).date()}\n")
            inf_end_ts = self.last_time

        if end_ts is None and inf_start_ts is None:
            raise ValueError("No start or end date provided for the training data and no start date provided for the inference data.")

        return start_ts, end_ts, inf_start_ts, inf_end_ts


    def aggregate_data(self, data: pd.DataFrame, aggregate: int):
        """
        Aggregate OHLCV data from the original granularity to the specified granularity.
        Saves to save_path

        args:
            data_path: path to the data
            aggregate: aggregate period (e.g., '5' for 5 minutes from 1 minute, '60' for 1 hour from 1 minute, '1440' for 1 day from 1 minute)
        returns:
            Path to the aggregated data file (save_path)
        """
        if aggregate <= 1:
            warnings.warn("Aggregate period is 1 or less. Returning original data.")
            return data

        if 'timestamp' not in data:
            raise ValueError("Missing 'timestamp' column")

        unix = data['timestamp'].dtype in ('int64', 'float64')
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s', utc=True) if unix else pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)

        # Convert numeric aggregate to pandas frequency string (e.g., 5 -> '5T' for 5 minutes)
        freq = f'{aggregate}T' if isinstance(aggregate, (int, float)) else aggregate

        agg = {k: v for k, v in {
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }.items() if k in data}
        agg.update({c: 'mean' for c in data if c not in agg})

        out = data.resample(freq).agg(agg).dropna().reset_index()
        if unix:
            out['timestamp'] = out['timestamp'].astype('int64') // 10**9

        return out

    def convert_to_returns(self, 
                           data: pd.DataFrame, 
                           keep_high_low: bool = False, 
                           keep_volume: bool = True, 
                           log_returns: bool = False):
        """
        Convert data to returns.

        args:
            data: DataFrame containing the data
            log_returns: bool, if True, the data is converted to log returns
            keep_high_low: bool, if True, the high and low prices are kept
            keep_volume: bool, if True, the volume column is kept
        returns:
                DataFrame containing the returns data
        """
        # Checks if the data path is a file or a directory and saves the output path accordingly
        try:
            data = pd.DataFrame({"close": data["close"], "volume": data["volume"], "timestamp": data["timestamp"]})
        except Exception as e:
            print("-------------ERROR-------------------")
            print(e)
            print("-------------ERROR-------------------")
            raise ValueError("Missing 'close', 'volume', or 'timestamp' columns")
        if log_returns:
            data["returns"] = np.log(data["close"] / data["close"].shift(1))
        else:
            data["returns"] = data["close"] / data["close"].shift(1) - 1
        
        data = data.dropna().reset_index(drop=True)

        final_data = pd.DataFrame()
        final_data['returns'] = data['returns']

        if keep_high_low:
            final_data["high"] = data["high"]
            final_data["low"] = data["low"]

        if keep_volume:
            final_data["volume"] = data["volume"]

        final_data["timestamp"] = data["timestamp"]

        return final_data


    def convert_back_to_candlesticks(self, num_predictions: int, org_inf_data: pd.DataFrame, processed_inf_data: pd.DataFrame, custom_save_path=None):
        """
        Convert inferenced returns data back to OHLCV (close_predicted_*) for backtesting.
        Backtest requires OHLCV format; it cannot use returns directly.

        For horizon 1: pred_close[i] = close[i-1] * (1 + returns_predicted_1[i])
        """
        result = org_inf_data.copy()
        predicted_returns = processed_inf_data.copy()

        # Align: ensure integer index for .shift
        result = result.reset_index(drop=True)
        predicted_returns = predicted_returns.reset_index(drop=True)

        if 'close' not in result.columns:
            raise ValueError("org_inf_data must have 'close' column for OHLCV conversion")

        for i in range(1, num_predictions + 1):
            col = f'returns_predicted_{i}'
            if col not in predicted_returns.columns:
                continue
            # Horizon 1: pred_close = prev_close * (1 + return). Use per-row prev close, not scalar.
            prev_close = result['close'].shift(1)
            pred_close = prev_close * (1 + predicted_returns[col])
            result[f'close_predicted_{i}'] = pred_close

        # Convert unix timestamp to UTC datetime for backtest
        if 'timestamp' in result.columns:
            result["timestamp"] = pd.to_datetime(result["timestamp"], unit='s', utc=True)

        if custom_save_path is not None:
            result.to_csv(custom_save_path, index=False)

        return result


    def check_data_paths(self, inference: bool = False, returns: bool = False) -> bool:
        """
        Checks if the data paths exist.

        args:
            inference: If True, also checks inference data paths.
            returns: If True and inference is True, also checks returns inference data path.
        returns:
            True if the data paths exist, False otherwise
        """
        if not os.path.exists(self.work_dir.get_ohlcv_train_data_path()):
            raise ValueError(f"OHLCV train data path {self.work_dir.get_ohlcv_train_data_path()} does not exist.")

        if not os.path.exists(self.work_dir.get_train_data_path()):
            raise ValueError(f"Train data path {self.work_dir.get_train_data_path()} does not exist.")

        if inference:
            if not os.path.exists(self.work_dir.get_org_ohlcv_inf_data_path()):
                raise ValueError(f"Original inference data path {self.work_dir.get_org_ohlcv_inf_data_path()} does not exist.")
            if returns:
                if not os.path.exists(self.work_dir.get_ret_inf_data_path()):
                    raise ValueError(f"Returns inference data path {self.work_dir.get_ret_inf_data_path()} does not exist.")
        return True