from data_provider import DataProvider
import pandas as pd
from darts.timeseries import TimeSeries

class CSVTSDataProvider(DataProvider):
    def __init__(self, csv_file, transform, time_col, value_cols):
        self.csv_file = csv_file
        self.transform = transform
        self.time_col = time_col
        self.value_cols = value_cols

    def get_data(self):
        df = pd.read_csv(self.csv_file, delimiter=',')
        df = self.transform(df)

        series = TimeSeries.from_dataframe(df, time_col=self.time_col, value_cols=self.value_cols)

        return series