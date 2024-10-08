from abc import ABC, abstractmethod
from torch.nn.modules import Module
from typing import Generic, TypeVar
import torch
from p2pfl.encryption import EncryptedTensor, EncryptedModel
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.timeseries import TimeSeries

T = TypeVar('T')

class Model(ABC, Generic[T]):
    @property
    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def train(self, data: T):
        pass

    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)
    
    def get_state_dict(self):
        return self.model.state_dict()
    
    def save_to_file(self, path):
        torch.save(self.get_state_dict(), path)
    
    def encrypt(self, context):
        state_dict = self.get_state_dict()
        encrypted_tensors = {}

        for k in state_dict.keys():
            tensor: torch.Tensor = state_dict[k]
            encrypted_tensor = EncryptedTensor.encrypt(context, tensor)
            encrypted_tensors[k] = encrypted_tensor
        
        return EncryptedModel(encrypted_tensors)

class TSForecastingModel(Model[TimeSeries]):
    forecasting_model: TorchForecastingModel = None

    def __init__(self, model: TorchForecastingModel):
        self.forecasting_model = model
    
    @property
    def model(self):
        return self.forecasting_model.model

    @abstractmethod
    def train(self, data: TimeSeries):
        pass

    def predict(self, time_step, samples: int = 1):
        return self.forecasting_model.predict(time_step, num_samples=samples)