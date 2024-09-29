from csv_ts_data_provider import CSVTSDataProvider
import datetime

def strToDateTime(str):
    return datetime.datetime.strptime(str, '%d/%m/%Y %H:%M')

def apply_datetime_transformations(df):
    df["tstp"] = df["tstp"].apply(lambda x: strToDateTime(x))
    return df

# Program Code
csv_ts_data_provider = CSVTSDataProvider('testdata.csv', lambda df: apply_datetime_transformations(df), time_col='tstp', value_cols=['energy(kWh/hh)'])

print(csv_ts_data_provider.get_data())