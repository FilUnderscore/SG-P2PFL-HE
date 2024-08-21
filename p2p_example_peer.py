from p2p_peer import P2PPeer
from MLTSModel import MLTSModel
from csv_ts_data_provider import CSVTSDataProvider

import datetime

import matplotlib.pyplot as plt
import random

import sys

from fl_sample_model import model

def strToDateTime(str):
    return datetime.datetime.strptime(str, '%d/%m/%Y')

def apply_datetime_transformations(df):
    df["day"] = df["day"].apply(lambda x: strToDateTime(x))
    return df

# Program Code
REGISTRATION_ADDRESS = "http://127.0.0.1:5001"
LOCAL_PORT = random.randint(1000, 2000)

DATASET_CSV_FILE = sys.argv[1]

ml_model = MLTSModel(model)
csv_ts_data_provider = CSVTSDataProvider(DATASET_CSV_FILE, lambda df: apply_datetime_transformations(df), time_col='day', value_cols=['energy_median'])

print('Starting Peer')
peer = P2PPeer(REGISTRATION_ADDRESS, LOCAL_PORT, ml_model, csv_ts_data_provider)
peer.start()

print('Beginning training')
peer.train()

csv_ts_data_provider.get_data().plot()

prediction = peer.ml_model.predict(365)
prediction.plot(label='local forecast')

print('Waiting for other peers...')
peer.wait_for_other_peers()

print('Aggregating all peer models')
peer.aggregate()

prediction = peer.ml_model.predict(365)
prediction.plot(label='global forecast')

plt.legend()
plt.title(f'PEER {LOCAL_PORT} predictions')
plt.show()