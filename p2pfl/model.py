from abc import ABC, abstractmethod
from torch.nn.modules import Module
from typing import Generic, TypeVar
import torch
from p2pfl.encryption import EncryptedTensor, EncryptedModel
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.timeseries import TimeSeries

T = TypeVar('T')

class Model(ABC, Generic[T]):
    """Represents a wrapper around a torch.nn module."""
    @property
    @abstractmethod
    def model(self) -> Module:
        """
        Returns the torch.nn model instance (which is a subclass of torch.nn.modules.Module).
        The model instance inherits the state dict which can be used to fetch, store, and encrypt NN model parameters.
        """
        pass

    @abstractmethod
    def train(self, data: T):
        """Begins the model training process with the provided data."""
        pass

    def load_state_dict(self, state_dict):
        """Loads the provided state dict into the model."""
        return self.model.load_state_dict(state_dict)
    
    def get_state_dict(self):
        """Retrieves the state dict from the model."""
        return self.model.state_dict()
    
    def save_to_file(self, path):
        """Saves the model to the provided path."""
        torch.save(self.get_state_dict(), path)
    
    def encrypt(self, context) -> EncryptedModel:
        """
        Encrypts the model and returns an encrypted instance of the model state dict
        (which can be used to decrypt and re-create the model instance later on).
        """
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