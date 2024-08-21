from torch.nn import L1Loss
from darts.models import RNNModel
from darts.utils.callbacks import TFMProgressBar

def generate_torch_kwargs():
    return {
        "pl_trainer_kwargs": {
            "accelerator": "gpu",
            "devices": [0],
            "callbacks": [TFMProgressBar(enable_train_bar_only=False)]
        }
    }

model = RNNModel(model = 'LSTM', hidden_dim=128, n_rnn_layers=2, dropout=0.5, batch_size=64, n_epochs=300, optimizer_kwargs={"lr": 0.001}, random_state=42, training_length=60, input_chunk_length=32, loss_fn=L1Loss(), force_reset=True, **generate_torch_kwargs())