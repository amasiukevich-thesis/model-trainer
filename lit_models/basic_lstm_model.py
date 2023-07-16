from typing import Tuple
import torchmetrics
import torch
import torch.nn as nn
import pytorch_lightning as pl


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        epsilon = 1e-6
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y) + epsilon)
        return loss


class LSTMRegressor(pl.LightningModule):
    """
    Standard Pytorch Lightning module
    credits: https://www.kaggle.com/code/tartakovsky/pytorch-lightning-lstm-timeseries-clean-code
    """

    def __init__(self,
                 n_features: int,
                 hidden_size: int,
                 seq_len: int,
                 batch_size: int,
                 num_layers: int,
                 dropout_prob: float,
                 learning_rate: float,
                 criterion: torch.nn = RMSELoss()):
        super(LSTMRegressor, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.criterion = criterion
        self.learning_rate = learning_rate

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_prob,
            batch_first=True
        )

        self.linear = nn.Linear(hidden_size, 1)

        # torchmetrics RMSE
        self.train_metric = torchmetrics.MeanSquaredError(squared=False)
        self.val_metric = torchmetrics.MeanSquaredError(squared=False)
        self.test_metric = torchmetrics.MeanSquaredError(squared=False)

        self.train_mape = torchmetrics.MeanAbsolutePercentageError()
        self.val_mape = torchmetrics.MeanAbsolutePercentageError()
        self.test_mape = torchmetrics.MeanAbsolutePercentageError()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        print(lstm_out[:, -1].shape)

        y_pred = self.linear(lstm_out[:, -1])
        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int):
        batch_X, batch_y = batch
        print(f"Training batch shape: {batch_X.shape}")
        preds = self(batch_X)

        loss = self.criterion(preds, batch_y)

        batch_metric_train = self.train_metric(preds, batch_y)
        self.log('train_metric_step', batch_metric_train)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor], batch_idx: int):
        batch_X, batch_y = batch
        preds = self(batch_X)

        loss = self.criterion(preds, batch_y)

        batch_metric_val = self.val_metric(preds, batch_y)
        self.log('val_metric_step', batch_metric_val)

        return loss

    def test_step(self, batch: Tuple[torch.Tensor], batch_idx: int):
        batch_X, batch_y = batch
        preds = self(batch_X)

        batch_y = self.trainer.datamodule.scaler.inverse_transform(batch_y)
        preds = self.trainer.datamodule.scaler.inverse_transform(preds.detach().numpy())

        batch_y, preds = torch.tensor(batch_y), torch.tensor(preds)

        loss = self.criterion(preds, batch_y)

        batch_metric_test = self.test_metric(preds, batch_y)

        batch_mape_test = self.test_mape(preds, batch_y)
        self.log('test_metric_step', batch_metric_test)
        self.log('test_mape_step', batch_mape_test)

        return loss

    def predict_step(self, batch, batch_idx):
        batch = batch.unsqueeze(0)
        preds = self(batch)

        scaler = self.trainer.datamodule.scaler
        preds = torch.tensor(scaler.inverse_transform(preds))

        return preds

    # def training_epoch_end(self, outs):
    #     self.log('train_metric_epoch', self.train_metric.compute(), prog_bar=True)
    #     self.train_metric.reset()
    #
    # def validation_epoch_end(self, outs):
    #     self.log('val_metric_epoch', self.val_metric.compute(), prog_bar=True)
    #     self.val_metric.reset()
    #
    # def testing_epoch_end(self, outs):
    #     self.log('test_metric_epoch', self.test_metric.compute(), prog_bar=True)
    #     self.loc('test_mape_epoch', self.test_mape.computecle(), prog_bar=True)
    #     self.val_metric.reset()

