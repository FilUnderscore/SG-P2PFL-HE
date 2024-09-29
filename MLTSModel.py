from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.timeseries import TimeSeries
import torch
from darts.utils.timeseries_generation import datetime_attribute_timeseries
import pandas as pd
from darts.dataprocessing.transformers.scaler import Scaler

class MLTSModel:
    def __init__(self, model: TorchForecastingModel):
        self.model = model

    def train(self, series: TimeSeries):
        train_set, val_set = series.split_after(0.75)
        transformer = Scaler()

        print('Train Set:')
        print(train_set)

        train_transformed = transformer.fit_transform(train_set)
        val_transformed = transformer.transform(val_set)

        #year_series = datetime_attribute_timeseries(pd.date_range(start=series.start_time(), freq=series.freq_str, periods=1000), attribute='year', one_hot=False)
        #year_series = Scaler().fit_transform(year_series)

        #month_series = datetime_attribute_timeseries(year_series, attribute='month', one_hot=True)

        #covariates = year_series.stack(month_series)
        #self.model.fit(train_transformed, future_covariates=covariates, val_series=val_transformed, val_future_covariates=covariates)
        self.model.fit(train_transformed, val_series=val_transformed)

    def predict(self, time_step, samples: int = 1):
        return self.model.predict(time_step, num_samples=samples)
    
    def load_state_dict(self, state_dict):
        return self.model.model.load_state_dict(state_dict)
    
    def save(self, path):
        torch.save(self.model.model.state_dict(), path)