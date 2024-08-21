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

PEER_DATA = ['testdata.csv', 'testdata1.csv', 'testdata2.csv', 'testdata3.csv']
PEERS = []

for i in range(0, len(PEER_DATA)):
    # Program Code
    ml_model = MLTSModel(model)
    csv_ts_data_provider = CSVTSDataProvider(PEER_DATA[i], lambda df: apply_datetime_transformations(df), time_col='day', value_cols=['energy_median'])

    print('Starting Peer')
    peer = FLPeer(ml_model, csv_ts_data_provider)
    peer.train()

    prediction = peer.ml_model.predict(365)

    plt.figure(i)
    csv_ts_data_provider.get_data().plot()
    prediction.plot(label="local forecast", low_quantile=0.05, high_quantile=0.95)
    plt.legend()
    plt.title(f'PEER {i} predictions')

    PEERS.append(peer)

print(f'All {len(PEERS)} peers trained. Ready for aggregation.')

PEER_MODELS = []

for i in range(0, len(PEERS)):
    peer: FLPeer = PEERS[i]
    peer.ml_model.save(f'model_{i}.pth')
    PEER_MODELS.append(f'model_{i}.pth')

print('Saved models. Starting aggregation.')

for i in range(0, len(PEERS)):
    peer: FLPeer = PEERS[i]
    peer.aggregate(PEER_MODELS)

print('Aggregated all models.')

for i in range(0, len(PEERS)):
    peer: FLPeer = PEERS[i]

    prediction = peer.ml_model.predict(365)

    plt.figure(i)
    prediction.plot(label="global forecast", low_quantile=0.05, high_quantile=0.95)
    plt.legend()
    plt.title(f'PEER {i} predictions')

plt.show()