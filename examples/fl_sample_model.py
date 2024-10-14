from torch.nn import MSELoss
from darts.models import RNNModel
from darts.utils.callbacks import TFMProgressBar

from pytorch_lightning.callbacks import Callback, EarlyStopping

from torchmetrics import MetricCollection, MeanAbsoluteError
from p2pfl.model import TSForecastingModel
from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers.scaler import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
import pandas as pd

class SampleTSForecastingModel(TSForecastingModel):
    def train(self, data: TimeSeries):
        train_set, val_set = data.split_after(0.8)
        transformer = Scaler()

        train_transformed = transformer.fit_transform(train_set)
        val_transformed = transformer.transform(val_set)

        year_series = datetime_attribute_timeseries(pd.date_range(start=data.start_time(), freq=data.freq_str, periods=30000), attribute='year', one_hot=False)
        year_series = Scaler().fit_transform(year_series)

        month_series = datetime_attribute_timeseries(year_series, attribute='month', one_hot=True)
        day_series = datetime_attribute_timeseries(month_series, attribute='day', one_hot=True)
        hour_series = datetime_attribute_timeseries(day_series, attribute='hour', one_hot=True)

        covariates = year_series.stack(month_series).stack(day_series).stack(hour_series)

        self.forecasting_model.fit(train_transformed, future_covariates=covariates, val_series=val_transformed, val_future_covariates=covariates)
        TSForecastingModel.train(self, data)

class LossRecorder(Callback):
    def __init__(self):
        self.train_loss_history = []
        self.val_loss_history = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_loss_history.append(trainer.callback_metrics["train_loss"].item())
        self.val_loss_history.append(trainer.callback_metrics["val_loss"].item())

loss_recorder = LossRecorder()
early_stopper = EarlyStopping(monitor="val_loss", patience=5, min_delta=0.0005, mode='min')

def recreate_early_stopper():
    global early_stopper
    early_stopper = EarlyStopping(monitor="val_loss", patience=5, min_delta=0.0005, mode='min')

def generate_torch_kwargs():
    return {
        "pl_trainer_kwargs": {
            "accelerator": "cpu",
            #"accelerator": "gpu" # uncomment this line to enable gpu training
            #"devices": [0], # uncomment this line to enable gpu training
            "callbacks": [TFMProgressBar(enable_train_bar_only=False), loss_recorder, early_stopper]
        }
    }

def create_new_model():
    return SampleTSForecastingModel(RNNModel(model = 'LSTM', hidden_dim=196, n_rnn_layers=2, dropout=0.22222, batch_size=128, n_epochs=100, optimizer_kwargs={"lr": 1e-3}, random_state=42, training_length=32, input_chunk_length=32, loss_fn=MSELoss(), force_reset=True, **generate_torch_kwargs()))

def train_model(model_args, callbacks):
    callbacks.append(early_stopper)
    torch_metrics = MetricCollection([MeanAbsoluteError()])
    return RNNModel(model = 'LSTM', n_epochs=300, optimizer_kwargs={"lr": 1e-3}, random_state=42, force_reset=True, torch_metrics=torch_metrics, pl_trainer_kwargs={"callbacks":callbacks, "enable_progress_bar": False}, **model_args)
