from array import array
import os
import pytorch_lightning as pl
import pandas as pd
import numpy as np

from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm



# Constants
UNIX_TIME_COL = "unix"
SERIES_LENGTH = 90

BATCH_SIZE = 64
TARGET_COL = ""

BASE_DIR = r"C:\Users\Antek\Desktop\inzynierka\ModelBuilding"
BASE_DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILENAME = "gemini_eth_usd_1h_headless.csv"

FILENAME_TRAIN = os.path.join(*[BASE_DIR, "data", "train_set_dummy.csv"])
FILENAME_VAL = os.path.join(*[BASE_DIR, "data", "val_set_dummy.csv"])
FILENAME_TEST = os.path.join(*[BASE_DIR, "data", "test_set_dummy.csv"])


class BaseDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=BATCH_SIZE):
        self.batch_size = batch_size

    def setup(self):
        self.train_dataset = TimeSeriesDataset(FILENAME_TRAIN, TARGET_COL)
        self.val_dataset = TimeSeriesDataset(FILENAME_VAL, TARGET_COL)
        self.test_dataset = TimeSeriesDataset(FILENAME_TEST, TARGET_COL)

    def train_dataloader(self):
        train_time_series = DataLoader(self.train_dataset, batch_size=self.batch_size)
        return train_time_series

    def val_dataloader(self):
        val_time_series = DataLoader(self.val_dataset, batch_size=self.batch_size)
        return val_time_series

    def test_dataloader(self):
        test_time_series = DataLoader(self.val_dataset, batch_size=self.batch_size)
        return test_time_series


class TimeSeriesDataset(Dataset):

    def __init__(self, file_path: str = None, target_col: str = None):
        dataframe = pd.read_csv(file_path)
        self.features = dataframe.drop(target_col, axis=1)
        self.labels = dataframe[target_col]

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[pd.Series, object]:
        return self.features.iloc[idx], self.labels.iloc[idx]


def make_dummy_series(dataset: pd.DataFrame):
    dataset_series = dataset['close']
    train_X, train_y = [], []

    for i in tqdm(range(len(dataset_series) - SERIES_LENGTH)):
        train_X.append(dataset_series.iloc[i: i + SERIES_LENGTH].apply(np.float16))
        train_y.append(np.float16(dataset_series.iloc[i + SERIES_LENGTH]))

    breakpoint()
    train_X_df, train_y_df = pd.DataFrame(train_X), pd.DataFrame(train_y)

    train_y_df = train_y_df.reindex(train_X_df.index)

    # TODO: SOMEWHERE HERE IS AN ERROR
    breakpoint()

    result = pd.concat([train_X_df, train_y_df], axis=1)
    print(result.shape)

    return result


def data_preparer(full_path: str, THR_TIMESTAMP: int, THR_VAL_TIMESTAMP: int) -> None:
    data = pd.read_csv(full_path)
    train_set_full = data[data[UNIX_TIME_COL] < THR_TIMESTAMP]

    train_set = train_set_full[train_set_full[UNIX_TIME_COL] < THR_VAL_TIMESTAMP]
    val_set = train_set_full[train_set_full[UNIX_TIME_COL] >= THR_VAL_TIMESTAMP]
    test_set = data[data[UNIX_TIME_COL] >= THR_TIMESTAMP]

    train_set = make_dummy_series(train_set)
    val_set = make_dummy_series(val_set)
    test_set = make_dummy_series(test_set)

    train_set.to_csv(FILENAME_TRAIN, index=False)
    val_set.to_csv(FILENAME_VAL, index=False)
    test_set.to_csv(FILENAME_TEST, index=False)


if __name__ == "__main__":
    THR_TIMESTAMP = 1640991600 * (10 ** 9)
    THR_VAL_TIMESTAMP = 1638226800 * (10 ** 9)
    data_preparer(os.path.join(BASE_DATA_DIR, DATA_FILENAME), THR_TIMESTAMP, THR_VAL_TIMESTAMP)

