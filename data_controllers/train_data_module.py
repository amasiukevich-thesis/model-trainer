import pandas as pd

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

from data_controllers.datasets import TimeSeriesDataset

from config import TRAIN_FILE_PATH, VAL_FILE_PATH


# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s [%(levelname)s]: %(message)s",
#     datefmt='%Y-%m-%d %H:%M:%S'
# )

@dataclass
class Globals:

    # DAILY_FILE_PATH = FULL_DATA_PATH #r"C:\Users\Antek\Desktop\inzynierka\ModelBuilding\data\gemini_eth_usd_1d_headless.csv"

    # DATE_THR_TRAIN = pd.to_datetime('2017-01-01')
    # DATE_THR_TEST = pd.to_datetime('2021-01-01')
    # DATE_THR_VAL = pd.to_datetime('2020-12-01')
    OPEN_COL = "price_open"
    DAY_COL = "rate_date"


class CryptoDataModule(pl.LightningDataModule):

    def __init__(self, seq_len=1, batch_size=128, num_workers=0):

        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_series = None
        self.val_series = None
        self.test_series = None

        self.columns = None
        self.preprocessing = None
        self.scaler = StandardScaler()  # FunctionTransformer(func=np.log1p, inverse_func=np.expm1)

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        if stage == "fit" and self.train_series is not None:
            return

        if stage == 'test' and self.test_series is not None:
            print("Maybe I return here???")
            return

        if stage is None and not (self.train_series is None or self.test_series is None):
            return

        train_series = pd.read_csv(TRAIN_FILE_PATH)
        train_series = train_series[Globals.OPEN_COL]

        val_series = pd.read_csv(VAL_FILE_PATH)
        val_series = val_series[Globals.OPEN_COL]

        # data_daily = pd.read_csv(Globals.DAILY_FILE_PATH)
        #
        # data_series = data_daily[[Globals.OPEN_COL, Globals.DAY_COL]]
        # data_series[Globals.DAY_COL] = data_series[Globals.DAY_COL].apply(pd.to_datetime)
        #
        # train_gen_series = data_series[data_series[Globals.DAY_COL] < Globals.DATE_THR_TEST]

        # train_series = train_gen_series[
        #     (train_gen_series[Globals.DAY_COL] >= Globals.DATE_THR_TRAIN) & \
        #     (train_gen_series[Globals.DAY_COL] < Globals.DATE_THR_VAL)
        #     ][Globals.OPEN_COL]

        # val_series = train_gen_series[
        #     (train_gen_series[Globals.DAY_COL] >= Globals.DATE_THR_VAL) & \
        #     (train_gen_series[Globals.DAY_COL] < Globals.DATE_THR_TEST)
        #     ][Globals.OPEN_COL]

        # test_series = data_series[
        #     data_series[Globals.DAY_COL] >= Globals.DATE_THR_TEST
        #     ][Globals.OPEN_COL]

        # TODO: SOOO TEMPORARY FIX
        column_name = Globals.OPEN_COL
        self.scaler.fit(train_series.values.reshape(-1, 1))  # TODO: temporary fix while not using

        # TODO: SOOO TEMPORARY FIX
        train_column_name = column_name
        self.train_series = train_series
        self.train_series = pd.Series(self.scaler.transform(self.train_series.values.reshape(-1, 1))[:, 0],
                                      name=train_column_name)

        print("Train series created")

        val_column_name = column_name
        self.val_series = val_series
        self.val_series = pd.Series(self.scaler.transform(self.val_series.values.reshape(-1, 1))[:, 0],
                                    name=val_column_name)

        print("Val series created")
        #
        # test_column_name = column_name
        # self.test_series = test_series
        # self.test_series = pd.Series(self.scaler.transform(self.test_series.values.reshape(-1, 1))[:, 0],
        #                              name=test_column_name)

        # print("Test series created")

        # TODO: use there different stages for the trainer

    def train_dataloader(self) -> DataLoader:
        self.train_dataset = TimeSeriesDataset(self.train_series, seq_len=self.seq_len)
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.num_workers, drop_last=True)
        return train_dataloader

    def val_dataloader(self) -> DataLoader:

        self.val_dataset = TimeSeriesDataset(self.val_series, seq_len=self.seq_len)
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                                    num_workers=self.num_workers, drop_last=False)
        return val_dataloader

    # def test_dataloader(self) -> DataLoader:
    #     self.test_dataset = TimeSeriesDataset(self.test_series, seq_len=self.seq_len)
    #     test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
    #                                  num_workers=self.num_workers, drop_last=False)
    #     return test_dataloader

