from torch.nn import L1Loss
from darts.models import RNNModel
from darts.utils.callbacks import TFMProgressBar

def generate_torch_kwargs():
    return {
        "pl_trainer_kwargs": {
            "accelerator": "cpu",
            "callbacks": [TFMProgressBar(enable_train_bar_only=False)]
        }
    }

model = RNNModel(model = 'LSTM', hidden_dim=20, n_rnn_layers=1, dropout=0, batch_size=16, n_epochs=100, optimizer_kwargs={"lr": 0.01}, random_state=42, training_length=16, input_chunk_length=10, loss_fn=L1Loss(), force_reset=True, **generate_torch_kwargs())