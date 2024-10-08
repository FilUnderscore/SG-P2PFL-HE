from p2pfl.fl_peer import FLPeer
from examples.fl_sample_model import SampleTSForecastingModel
from p2pfl.data_provider import CSVTSDataProvider

import datetime

from examples.fl_sample_model import loss_recorder, train_model, recreate_early_stopper
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler

from torch.nn import L1Loss, MSELoss
import random

def strToDateTime(str):
    return datetime.datetime.strptime(str, '%d/%m/%Y %H:%M')

def apply_datetime_transformations(df):
    df["tstp"] = df["tstp"].apply(lambda x: strToDateTime(x))
    return df

def train_model1(model_args, callbacks, csv_ts_data_provider):
    ml_model = SampleTSForecastingModel(train_model(model_args, callbacks))
    peer = FLPeer(ml_model, csv_ts_data_provider)
    peer.train()
    loss_recorder.train_loss_history.clear()
    loss_recorder.val_loss_history.clear()
    recreate_early_stopper()

csv_ts_data_provider = CSVTSDataProvider('dataset/testdata2.csv', lambda df: apply_datetime_transformations(df), time_col='tstp', value_cols=['energy(kWh/hh)'])

tune_callback = TuneReportCheckpointCallback(
    {
        "loss": "val_loss"
    },
    on="validation_end",
    save_checkpoints=False
)

config = {
    "batch_size": tune.choice([16, 32, 64, 128, 256]),
    "n_rnn_layers": tune.choice([1, 2, 3]),
    "dropout": tune.uniform(0, 0.5),
    "training_length": tune.sample_from(lambda spec: random.randint(spec.config.input_chunk_length, spec.config.input_chunk_length * 2)),
    "input_chunk_length": tune.choice([16, 32, 64, 128]),
    "hidden_dim": tune.randint(1, 256),
    "loss_fn": tune.choice([L1Loss(), MSELoss()])
}

resources_per_trial = {"cpu": 8, "gpu": 1}

num_samples = 10

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