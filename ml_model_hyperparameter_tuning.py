from fl_peer import FLPeer
from MLTSModel import MLTSModel
from csv_ts_data_provider import CSVTSDataProvider

import datetime

import matplotlib.pyplot as plt

from fl_sample_model import model, loss_recorder, train_model
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler

from torch.nn import L1Loss, MSELoss

def strToDateTime(str):
    return datetime.datetime.strptime(str, '%d/%m/%Y')

def apply_datetime_transformations(df):
    df["day"] = df["day"].apply(lambda x: strToDateTime(x))
    return df

def train_model1(model_args, callbacks, csv_ts_data_provider):
    ml_model = MLTSModel(train_model(model_args, callbacks))
    peer = FLPeer(ml_model, csv_ts_data_provider)
    peer.train()

csv_ts_data_provider = CSVTSDataProvider('C:\\Users\\Filip\\Desktop\\P2PFL\\testdata.csv', lambda df: apply_datetime_transformations(df), time_col='day', value_cols=['energy_median'])

tune_callback = TuneReportCheckpointCallback(
    {
        "loss": "val_loss",
    },
    on="validation_end"
)

config = {
    "batch_size": tune.choice([16, 32, 64, 128]),
    "n_rnn_layers": tune.choice([1, 2, 3]),
    "dropout": tune.uniform(0, 0.3),
    "training_length": tune.choice([32, 64]),
    "input_chunk_length": tune.choice([16, 32]),
    "hidden_dim": tune.randint(1, 64),
    "loss_fn": tune.choice([L1Loss(), MSELoss()])
}

resources_per_trial = {"cpu": 8, "gpu": 1}

num_samples = 100

scheduler = ASHAScheduler(max_t=1000, grace_period=3, reduction_factor=2)

train_fn_with_parameters = tune.with_parameters(
    train_model1, callbacks=[tune_callback], csv_ts_data_provider=csv_ts_data_provider
)

reporter = CLIReporter(
    parameter_columns=list(config.keys()),
    metric_columns=["loss", "training_iteration"],
)

analysis = tune.run(
    train_fn_with_parameters,
    resources_per_trial=resources_per_trial,
    metric="loss",
    mode="min",
    config=config,
    num_samples=num_samples,
    scheduler=scheduler,
    progress_reporter=reporter,
    name="tune_darts"
)

print("Best hyperparameters found were: ", analysis.best_config)