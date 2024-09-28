from torch.nn import L1Loss, MSELoss
from darts.models import RNNModel
from darts.utils.callbacks import TFMProgressBar

from pytorch_lightning.callbacks import Callback, EarlyStopping
import matplotlib.pyplot as plt

from torchmetrics import MetricCollection, MeanAbsolutePercentageError, MeanAbsoluteError

class LossRecorder(Callback):
    def __init__(self):
        self.train_loss_history = []
        self.val_loss_history = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_loss_history.append(trainer.callback_metrics["train_loss"].item())
        self.val_loss_history.append(trainer.callback_metrics["val_loss"].item())

loss_recorder = LossRecorder()
early_stopper = EarlyStopping(monitor="val_loss", patience=30, min_delta=0.05, mode='min')

def generate_torch_kwargs():
    return {
        "pl_trainer_kwargs": {
            "accelerator": "gpu",
            "devices": [0],
            "callbacks": [TFMProgressBar(enable_train_bar_only=False), loss_recorder]
        }
    }

# Best hyperparameters found were:  {'batch_size': 16, 'n_rnn_layers': 2, 'dropout': 0.14587531358360983, 'training_length': 32, 'input_chunk_length': 32, 'hidden_dim': 18, 'loss_fn': MSELoss()}
model = RNNModel(model = 'LSTM', hidden_dim=18, n_rnn_layers=2, dropout=0.14587, batch_size=16, n_epochs=1000, optimizer_kwargs={"lr": 1e-3}, random_state=42, training_length=32, input_chunk_length=32, loss_fn=MSELoss(), force_reset=True, **generate_torch_kwargs())

def train_model(model_args, callbacks):
    callbacks.append(early_stopper)
    torch_metrics = MetricCollection([MeanAbsoluteError()])
    return RNNModel(model = 'LSTM', n_epochs=300, optimizer_kwargs={"lr": 1e-3}, random_state=42, force_reset=True, torch_metrics=torch_metrics, pl_trainer_kwargs={"callbacks":callbacks, "enable_progress_bar": False}, **model_args)
