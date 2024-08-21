from fl_peer import FLPeer
from MLTSModel import MLTSModel
from csv_ts_data_provider import CSVTSDataProvider

import datetime

import matplotlib.pyplot as plt

from fl_sample_model import model

def strToDateTime(str):
    return datetime.datetime.strptime(str, '%d/%m/%Y')

def apply_datetime_transformations(df):
    df["day"] = df["day"].apply(lambda x: strToDateTime(x))
    return df

# Program Code
ml_model = MLTSModel(model)
csv_ts_data_provider = CSVTSDataProvider('testdata.csv', lambda df: apply_datetime_transformations(df), time_col='day', value_cols=['energy_median'])

print('Starting Peer')
peer = FLPeer(ml_model, csv_ts_data_provider)
peer.train()

prediction = peer.ml_model.predict(365)

csv_ts_data_provider.get_data().plot()
prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
plt.legend()
plt.show()