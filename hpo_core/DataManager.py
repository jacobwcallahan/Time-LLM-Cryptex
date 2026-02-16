"""
This class is responsible for managing the train and inference data. 
"""

from pathlib import Path
import pandas as pd
from datetime import datetime
import warnings
from hpo_core.WorkDir import WorkDir
import numpy as np
from hpo_core.HpoArgs import HpoArgs
import os


class DataManager:
    """
    This class is responsible for managing the train and inference data.

    functions: 
        prepare_data: Prepares the data for training and inference.
        check_date_validity: Checks the validity of the start and end dates.
        aggregate_data: Aggregates the data from the original granularity to the specified granularity.
        convert_to_returns: Converts the data to returns.
        convert_back_to_candlesticks: Converts the returns data back to candlesticks.

    args:
        args: HpoArgs object

    raises:
        ValueError: If the data path is not a valid file.
        ValueError: If the start date is before the first date of the dataset.
        ValueError: If the end date is after the last date of the dataset.

    attributes:
        data: DataFrame containing the training data
        inf_data: DataFrame containing the inference data
        data_path: Path to the data file
        full_data: DataFrame containing the full data
        first_time: Timestamp of the first date of the dataset
        last_time: Timestamp of the last date of the dataset
        granularity: Granularity of the data
        start_date: Timestamp of the start date
        end_date: Timestamp of the end date
        inf_start_date: Timestamp of the inference start date
        inf_end_date: Timestamp of the inference end date
        returns: Boolean indicating if the data is returns
        aggregate: Integer indicating the aggregate period
        INFERENCE: Boolean indicating if inference is enabled
        DATASET_PATH: Path to the dataset


    """

    
    def __init__(self, args: HpoArgs, work_dir: WorkDir):
        self.args = args
        self.work_dir = work_dir


        self.start_date = None
        self.end_date = None

        self.inf_start_date = None
        self.inf_end_date = None

        self.aggregate = args.aggregate

        self.args.returns = args.returns

        self.full_data = work_dir.get_full_data()

        self.first_time = datetime.fromtimestamp(int(self.full_data.iloc[0]['timestamp'])).timestamp()
        self.last_time = datetime.fromtimestamp(int(self.full_data.iloc[-1]['timestamp'])).timestamp()
        
        self.data = None
        self.inf_data = None
        self.check_date_validity()

    def prepare_train_data(self):
        """
        Prepares the data for training and inference.
        """
        self.data = self.full_data[(self.full_data['timestamp'] >= self.start_date) & (self.full_data['timestamp'] <= self.end_date)]

        self.data = self.aggregate_data(self.data, self.aggregate)

        # Write the OHLCV train data to the work directory
        self.work_dir.write_ohlcv_train_data(self.data)

        # If the data is returns, convert it to returns and save it
        if self.args.returns:
            self.data = self.convert_to_returns(self.data)
            self.work_dir.write_ret_train_data(self.data)

        self.work_dir.write_train_data(self.data)

    def prepare_inf_data(self):
        if not self.args.INFERENCE:
            warnings.warn("Inference data is not prepared. Please run the prepare_data function first.")
            return

        self.inf_start_date = datetime.strptime(self.args.inf_start, '%Y-%m-%d').timestamp()
        self.inf_end_date = datetime.strptime(self.args.inf_end, '%Y-%m-%d').timestamp()
            
        self.inf_data = self.full_data[(self.full_data['timestamp'] >= self.inf_start_date) & (self.full_data['timestamp'] <= self.inf_end_date)]
        self.inf_data = self.aggregate_data(self.inf_data, self.aggregate)

        self.work_dir.write_org_ohlcv_inf_data(self.inf_data)
        if self.args.returns:
            self.inf_data = self.convert_to_returns(self.inf_data)
            self.work_dir.write_ret_inf_data(self.inf_data)

        self.work_dir.write_inf_data(self.inf_data) 

    def check_date_validity(self):
        """
        Checks the validity of the start and end dates.
        """

        if self.args.start is None:
            self.start_date = self.first_time
            warnings.warn(f"No start date provided. Using the first date of the dataset: {datetime.fromtimestamp(self.first_time).date()}")
        else:
            self.start_date = datetime.strptime(self.args.start, '%Y-%m-%d').timestamp()
            if self.start_date < self.first_time:
                warnings.warn(f"The start date given ({self.start_date}) is before the first date of the dataset. Using the first date of the dataset: {datetime.fromtimestamp(self.first_time).date()}\n")
                self.start_date = self.first_time

        if self.args.end is None:
            self.end_date = self.last_time
            warnings.warn(f"No end date provided. Using the last date of the dataset: {datetime.fromtimestamp(self.last_time).date()}\n")
        else:
            self.end_date = datetime.strptime(self.args.end, '%Y-%m-%d').timestamp()
            if self.end_date > self.last_time:
                warnings.warn(f"The end date given ({self.end_date}) is after the last date of the dataset. Using the last date of the dataset: {datetime.fromtimestamp(self.last_time).date()}\n")
                self.end_date = self.last_time

        if self.start_date is not None and self.end_date is not None:
            if self.start_date > self.end_date:
                warnings.warn(f"The start date given ({self.start_date}) is after the end date given ({self.end_date}). Using the last date of the dataset: {datetime.fromtimestamp(self.last_time).date()} as the end date.\n")
                self.end_date = self.last_time

        if self.args.INFERENCE:
            if self.args.inf_start is not None and self.args.inf_end is not None:
                self.inf_start_date = datetime.strptime(self.args.inf_start, '%Y-%m-%d').timestamp()
                self.inf_end_date = datetime.strptime(self.args.inf_end, '%Y-%m-%d').timestamp()
                if self.inf_start_date > self.inf_end_date:
                    warnings.warn(f"The inference start date given ({self.inf_start_date}) is after the inference end date given ({self.inf_end_date}). Using the last date of the dataset: {datetime.fromtimestamp(self.last_time).date()} as the inference end date.\n")
                    self.inf_end_date = self.last_time
            else:
                if self.args.inf_start is not None:
                    self.inf_start_date = datetime.strptime(self.args.inf_start, '%Y-%m-%d').timestamp()
                    if self.inf_start_date < self.first_time:
                        warnings.warn(f"The inference start date given ({self.inf_start_date}) is before the first date of the dataset. Using the first date of the dataset: {datetime.fromtimestamp(self.first_time).date()}\n")
                        self.inf_start_date = self.first_time
                if self.args.inf_end is not None:
                    self.inf_end_date = datetime.strptime(self.args.inf_end, '%Y-%m-%d').timestamp()
                    if self.inf_end_date > self.last_time:
                        warnings.warn(f"The inference end date given ({self.inf_end_date}) is after the last date of the dataset. Using the last date of the dataset: {datetime.fromtimestamp(self.last_time).date()}\n")
                        self.inf_end_date = self.last_time

            if self.start_date is not None and self.end_date is not None and self.inf_start_date is not None and self.inf_end_date is not None:
                if self.end_date > self.inf_start_date:
                    warnings.warn(f"The end date given ({self.end_date}) is after the inference start date given ({self.inf_start_date}). Using the end date as the inference start date.\n")
                    self.inf_start_date = self.end_date

            if self.end_date is not None and self.inf_start_date is None:
                warnings.warn(f"No inference start date provided. Using the end date as the inference start date: {datetime.fromtimestamp(self.end_date).date()}\n")
                self.inf_start_date = self.end_date

            if self.inf_end_date is not None and self.inf_start_date is None:
                warnings.warn(f"No inference end date provided. Using the last date of the dataset as the inference end date: {datetime.fromtimestamp(self.last_time).date()}\n")
                self.inf_end_date = self.last_time

            if self.end_date is None and self.inf_start_date is None:
                raise ValueError("No start or end date provided for the training data and no start date provided for the inference data.")


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

    def convert_to_returns(self, data: pd.DataFrame, keep_high_low=False, keep_volume=True, log_returns=False):
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
        Convert inferenced returns data back to candlesticks. This is used to backtest the model.
        Writes the inferenced data to the inferenced data path.

        args:
            num_predictions: number of predictions to convert back to candlesticks
            custom_save_path: optional path to save the result
        """

        result = org_inf_data
        predicted_returns = processed_inf_data.copy()

        # Get the last known close price before predictions start
        try:
            last_close = result.loc[result.index[predicted_returns['returns_predicted_1'].first_valid_index()-1], 'close']
        except Exception as e:
            print(f"Error getting last close price: {e}\n")
            raise ValueError(f"Error getting last close price: {e}")

        for i in range(1, num_predictions+1):  
            col = f'returns_predicted_{i}'
            if col in predicted_returns.columns:
                # Calculate cumulative returns 
                pred_close = last_close * (1 + predicted_returns[col])
                # Rename column
                result[f'close_predicted_{i}'] = pred_close

        # Convert unix timestamp to UTC datetime
        result["timestamp"] = pd.to_datetime(result["timestamp"], unit='s', utc=True)

        if custom_save_path is not None:
            result.to_csv(custom_save_path, index=False)

        return result


    def check_data_paths(self) -> bool:
        """
        Checks if the data paths exist.

        args:
            self: DataManager object
        returns:
            True if the data paths exist, False otherwise
        """
        if not os.path.exists(self.work_dir.get_ohlcv_train_data_path()):
            raise ValueError(f"OHLCV train data path {self.work_dir.get_ohlcv_train_data_path()} does not exist.")

        if not os.path.exists(self.work_dir.get_train_data_path()):
            raise ValueError(f"Train data path {self.work_dir.get_train_data_path()} does not exist.")
        
        if self.args.INFERENCE:
            if not os.path.exists(self.work_dir.get_org_ohlcv_inf_data_path()):
                raise ValueError(f"Original inference data path {self.work_dir.get_org_ohlcv_inf_data_path()} does not exist.")
            if self.args.returns:
                if not os.path.exists(self.work_dir.get_ret_inf_data_path()):
                    raise ValueError(f"Returns inference data path {self.work_dir.get_ret_inf_data_path()} does not exist.")
        return True