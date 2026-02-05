import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class Dataset_Crypto(Dataset):
    """
    Custom dataset for CRYPTEX OHLCV data. Returns input_data and target_data.
    size: [seq_len, pred_len] must be provided (defaults handled by argument parser)
    """
    def __init__(self, root_path, data_path='candlesticks-D.csv', flag='train', 
                 size=None, features='MS', target='close', percent=100, train_val_test_ratio=[0.8, 0.1, 0.1]):
        assert size is not None, 'size (seq_len, pred_len) must be provided.'
        self.seq_len = size[0]  # Length of input sequence
        self.pred_len = size[1] # Length of prediction sequence
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]  # 0: train, 1: val, 2: test

        self.features = features  # Feature selection mode: 'M', 'S', or 'MS'
        self.target = target      # Target column name
        self.percent = percent    # Percentage of training data to use
        self.flag = flag          # Store flag for debugging

        self.root_path = root_path  # Root directory for data
        self.data_path = data_path  # CSV file name
        self.__read_data__(train_val_test_ratio)        # Load and preprocess data

        self.num_features = self.data_x.shape[-1]  # Number of input features
        # Number of possible starting points for a sequence in the data
        self.tot_len = len(self.data_x) - (self.seq_len + self.pred_len - 1)

    def __read_data__(self, split_ratio):
        try:
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        except Exception as e:
            if df_raw is not None:
                raise ValueError(f"Error reading data from {os.path.join(self.root_path, self.data_path)}: {e}")
            else:
                raise ValueError(f"Dataset is None. Please check if the data path is correct. {os.path.join(self.root_path, self.data_path)}")


        # Always put target as last column (except timestamp)
        feature_cols = [col for col in df_raw.columns if col not in ['timestamp', self.target]]
        ordered_cols = ['timestamp'] + feature_cols + [self.target]
        df_raw = df_raw[ordered_cols]

        # Split train/val/test by fixed proportions (80/10/10)
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1

        total_len = len(df_raw)
        num_train = int(total_len * train_ratio)
        num_vali = int(total_len * val_ratio)
        num_test = total_len - num_train - num_vali
        # Calculate borders for each split
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # For training, optionally use only a percentage of the data
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        # Feature selection: use all features or just the target
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]  # Exclude timestamp
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        data = df_data.values

        # Store the processed data for this split
        self.data_x = data[border1:border2]
        
        # Data validation - check for NaN/Inf values (using numpy for numpy arrays)
        if np.isnan(self.data_x).any():
            print(f"WARNING: NaN values found in {self.flag} data!")
            print(f"NaN count: {np.isnan(self.data_x).sum()}")
        if np.isinf(self.data_x).any():
            print(f"WARNING: Inf values found in {self.flag} data!")
            print(f"Inf count: {np.isinf(self.data_x).sum()}")
        
        print(f"{self.flag} data shape: {self.data_x.shape}")
        print(f"{self.flag} data range: {self.data_x.min():.6f} to {self.data_x.max():.6f}")

    def __getitem__(self, index):
        # Calculate which feature and which time window this index refers to
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        # Slice input and target data for this sample and feature
        input_data = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        target_data = self.data_x[r_begin:r_end, feat_id:feat_id + 1]
        return input_data, target_data

    def __len__(self):
        # Total number of samples = number of possible windows * number of features
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.num_features


def data_provider(args, flag):
    """
    Returns a Dataset_Crypto and DataLoader for the CRYPTEX dataset.
    """
    dataset = Dataset_Crypto(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.pred_len],
        features=args.features,
        target=args.target,
        percent=args.percent,
    )

    shuffle_flag = flag != 'test' # Shuffle for train/val, no shuffle for test
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=True # Ensures full batches only
    )
    return dataset, data_loader 