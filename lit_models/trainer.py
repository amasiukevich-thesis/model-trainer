import torch
import pickle
import pytorch_lightning as pl
from pytorch_lightning.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import TQDMProgressBar

import torch.onnx as onnx


import os
from datetime import datetime

from config import (
    MODEL_CONFIG,
    DATA_MODULE_PARAMS,
    TRAINER_PARAMS,
    TRAIN_LOGS_DIR,
    MODEL_CLASS_NAME,
    MODEL_SAVE_DIR
)

from lit_models.basic_lstm_model import LSTMRegressor
from data_controllers.train_data_module import CryptoDataModule

# TODO: move to helpers later

if __name__ == "__main__":
    # logger = CSVLogger(TRAIN_LOGS_DIR, name=MODEL_CLASS_NAME, version='0')

    model = LSTMRegressor(**MODEL_CONFIG)
    data_module = CryptoDataModule(**DATA_MODULE_PARAMS)

    trainer = pl.Trainer(**TRAINER_PARAMS)

    trainer.fit(model, data_module)
    trainer.validate(model, datamodule=data_module)
    # trainer.test(model, datamodule=data_module)

    # TODO: save the model checkpoint here
    now_time = datetime.now()
    VERSION = now_time.strftime("%Y-%m-%d_%H-%M-%S")

    if not os.path.isdir(os.path.join(MODEL_SAVE_DIR, str(VERSION))):
        os.mkdir(os.path.join(MODEL_SAVE_DIR, str(VERSION)))

    # TODO: refactor this folder name into a global variable
    torch.save(
        model.state_dict(),
        os.path.join(MODEL_SAVE_DIR, str(VERSION), "lstm_model.ckpt")
    )

    with open(os.path.join(MODEL_SAVE_DIR, str(VERSION), "scaler.pkl"), "wb") as f:
        pickle.dump(data_module.scaler, f)

    # TODO: refactor this later for the helper function
    dummy_input = torch.randn(1, MODEL_CONFIG['seq_len'], MODEL_CONFIG['n_features'])
    onnx.export(
        model,
        dummy_input,
        os.path.join(MODEL_SAVE_DIR, str(VERSION), "lstm_model.onnx")
    )
