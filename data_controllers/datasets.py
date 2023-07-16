import torch
import pandas as pd

from torch.utils.data import Dataset
from typing import Tuple


# Just plain PyTorch dataset
class TimeSeriesDataset(Dataset):

    def __init__(self, series: pd.Series, seq_len: int = 1):
        sequences_X = [series.iloc[i: i + seq_len].to_list() for i in range(series.shape[0] - seq_len)]
        labels_y = [series.iloc[i + seq_len] for i in range(series.shape[0] - seq_len)]

        print(series.shape)
        self.X = torch.Tensor(sequences_X).unsqueeze(-1).float()
        self.y = torch.Tensor(labels_y).unsqueeze(-1).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        return (self.X[idx, :], self.y[idx])
