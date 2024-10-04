from p2p_peer import P2PPeer
from MLTSModel import MLTSModel
from data_provider import CSVTSDataProvider

from datetime import datetime

import matplotlib.pyplot as plt
import random

import sys

from fl_sample_model import create_new_model
from darts.dataprocessing.transformers.scaler import Scaler

def strToDateTime(str):
    return datetime.strptime(str, '%d/%m/%Y %H:%M')

def apply_datetime_transformations(df):
    df["tstp"] = df["tstp"].apply(lambda x: strToDateTime(x))
    return df

# Program Code
REGISTRATION_ADDRESS = "http://127.0.0.1:5001"
LOCAL_PORT = random.randint(1000, 2000)

DATASET_CSV_FILE = sys.argv[1]

ml_model = MLTSModel(create_new_model())
csv_ts_data_provider = CSVTSDataProvider(DATASET_CSV_FILE, lambda df: apply_datetime_transformations(df), time_col='tstp', value_cols=['energy(kWh/hh)'])

print('Starting Peer')
peer = P2PPeer(REGISTRATION_ADDRESS, LOCAL_PORT, ml_model, csv_ts_data_provider)
peer.start()

print('Beginning training')
peer.train()

prediction = peer.ml_model.predict(365 * 20)
series = csv_ts_data_provider.get_data()
transformer = Scaler()
transformer.fit(series)
prediction_inverse_transformed = transformer.inverse_transform(prediction)
series.plot(label='actual')
prediction_inverse_transformed.plot(label='local forecast')

print('Waiting for other peers...')
peer.wait_for_other_peers()

print('Aggregating all peer models')
peer.aggregate()

prediction = peer.ml_model.predict(365 * 20)
prediction_inverse_transformed = transformer.inverse_transform(prediction)
prediction_inverse_transformed.plot(label='global forecast')

plt.legend()
plt.title(f'PEER {LOCAL_PORT} predictions')
plt.show()