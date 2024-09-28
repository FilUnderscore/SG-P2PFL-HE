from fl_peer import FLPeer
from MLTSModel import MLTSModel
from csv_ts_data_provider import CSVTSDataProvider

import datetime

import matplotlib.pyplot as plt

from fl_sample_model import model, loss_recorder

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

plt.figure(0)
csv_ts_data_provider.get_data().plot()
prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
plt.legend()
plt.title(f'Training Results')

plt.figure(1)
plt.plot(loss_recorder.train_loss_history, label='Train Loss')
plt.plot(loss_recorder.val_loss_history, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title(f'Accuracy')

plt.show()