from MLTSModel import MLTSModel
from data_provider import DataProvider
import torch
from copy import deepcopy

class FLPeer:
    def __init__(self, ml_model: MLTSModel, data_provider: DataProvider):
        self.ml_model = ml_model
        self.data_provider = data_provider

    def train(self):
        print('Training')

        self.ml_model.train(self.data_provider.get_data())

    def aggregate(self, models):
        peer_models = []

        for model in models:
            peer_models.append(torch.load(model))

        print(f'Aggregating {len(peer_models)} models.')

        model_values = FedAvg(peer_models)
        self.ml_model.load_state_dict(model_values)
        print('Aggregated models.')

def FedAvg(w):
    w_avg = deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg