from argparse import ArgumentParser

from fl_sample_model import create_new_model, loss_recorder, recreate_early_stopper
from MLTSModel import MLTSModel

from data_provider import CSVTSDataProvider
from datetime import datetime

from fl_peer import FLPeer
import matplotlib.pyplot as plt

from darts.dataprocessing.transformers.scaler import Scaler
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

def strToDateTime(str):
    return datetime.strptime(str, '%d/%m/%Y %H:%M')

def apply_datetime_transformations(df):
    df["tstp"] = df["tstp"].apply(lambda x: strToDateTime(x))
    return df

def train_model(data_csv_file):
    ml_model = MLTSModel(create_new_model())
    csv_ts_data_provider = CSVTSDataProvider(data_csv_file, lambda df: apply_datetime_transformations(df), time_col='tstp', value_cols=['energy(kWh/hh)'])

    print('Training model with data ', data_csv_file)
    peer = FLPeer(ml_model, csv_ts_data_provider)
    peer.train()

    return peer

def init(args):
    data_csv_files = args.data_csv_files
    peers = []
    peer_train_loss = []
    peer_val_loss = []
    
    for i in range(0, len(data_csv_files)):
        data_csv_file = data_csv_files[i]
        peer = train_model(data_csv_file)

        prediction = peer.ml_model.predict(365 * 20)
        plt.figure(i)
        series = peer.data_provider.get_data()
        transformer = Scaler()
        transformer.fit(series)
        prediction_inverse_transformed = transformer.inverse_transform(prediction)
        series.plot(label='actual')
        prediction_inverse_transformed.plot(label='local forecast')

        # Plot training metrics
        plt.figure(len(data_csv_files))
        plt.plot(loss_recorder.train_loss_history, label=f'PEER {i} Training Loss')
        plt.plot(loss_recorder.val_loss_history, label=f'PEER {i} Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Training Accuracy Metrics')

        peer_train_loss.append(loss_recorder.train_loss_history)
        peer_val_loss.append(loss_recorder.val_loss_history)
        loss_recorder.train_loss_history.clear()
        loss_recorder.val_loss_history.clear()
        recreate_early_stopper()

        peers.append(peer)
    
    if len(peers) == 1:
        plt.show()
        return # Only one peer was trained so we can't produce any FL results.
    
    print(f'Trained {len(peers)} peers. Beginning model aggregation.')

    models = []

    for i in range(0, len(peers)):
        peer: FLPeer = peers[i]
        peer.ml_model.save(f'model_{i}.pth')
        models.append(f'model_{i}.pth')
    
    print('Saved models.')

    for i in range(0, len(peers)):
        peer: FLPeer = peers[i]
        peer.aggregate(models)
    
    print('Aggregated all models.')

    for i in range(0, len(peers)):
        peer: FLPeer = peers[i]

        prediction = peer.ml_model.predict(365 * 20)

        plt.figure(i)
        series = peer.data_provider.get_data()
        transformer = Scaler()
        transformer.fit(series)
        prediction_inverse_transformed = transformer.inverse_transform(prediction)
        prediction_inverse_transformed.plot(label='global forecast')
        plt.xlabel('Date (hh)')
        plt.ylabel('Energy Consumption (kWh/hh)')
        plt.legend()
        plt.title(f'PEER {i} predictions')

    global_train_loss = []

    for i in range(0, len(peer_train_loss)):
        value = peer_train_loss[i]
        print(value)

    plt.figure(len(peers))

    plt.show()

parser = ArgumentParser(prog='local_peers_example')

parser.add_argument('data_csv_files', type=str, nargs='+')
args = parser.parse_args()

init(args)