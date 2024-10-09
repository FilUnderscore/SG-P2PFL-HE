from abc import ABC, abstractmethod
import pandas as pd
from darts.timeseries import TimeSeries
from typing import Generic, TypeVar

T = TypeVar('T')

class DataProvider(ABC, Generic[T]):
    """Parses data and provides it via the get_data function."""
    @abstractmethod
    def get_data(self) -> T:
        """Returns any data parsed by the inheriting type"""
        pass

class CSVDataProvider(DataProvider[pd.DataFrame]):
    df : pd.DataFrame = None

    """Parses CSV data into a DataFrame, applying an optional transform in the process."""
    def __init__(self, csv_file, transform):
        df = pd.read_csv(csv_file, delimiter=',')

        if transform != None:
            df = transform(df)
        
        self.df = df
    
    def get_data(self) -> pd.DataFrame:
        """Returns CSV data in the form of a Pandas DataFrame."""

        return self.df

class TSDataProvider(DataProvider[TimeSeries]):
    series : TimeSeries = None

    """Parses TS data from a DataFrame into a TimeSeries object."""
    def __init__(self, df, time_col, value_cols):
        self.time_col = time_col
        self.value_cols = value_cols

        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
        df = df.asfreq(freq='h', fill_value=0.0)

        self.series = TimeSeries.from_dataframe(df, value_cols=self.value_cols)
    
    def get_data(self) -> TimeSeries:
        """Returns TS data in the form of a Darts TimeSeries."""
        return self.series

class CSVTSDataProvider(CSVDataProvider, TSDataProvider):
    """Parses CSV data from a CSV file into a Darts TimeSeries."""
    def __init__(self, csv_file, transform, time_col, value_cols):
        CSVDataProvider.__init__(self, csv_file, transform)
        df = CSVDataProvider.get_data(self)
        TSDataProvider.__init__(self, df, time_col, value_cols)
        self.series = TSDataProvider.get_data(self)

    def get_data(self) -> TimeSeries:
        """Returns TS data in the form of a Darts TimeSeries."""
        return self.series