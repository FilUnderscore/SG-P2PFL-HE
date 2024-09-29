from argparse import ArgumentParser
from data_provider import CSVDataProvider
import pandas as pd
from datetime import datetime

def strToDateTime(str):
    return datetime.strptime(str, '%Y-%m-%d %H:%M:%S.0000000')

def apply_datetime_transformations(df):
    df["tstp"] = df["tstp"].apply(lambda x: strToDateTime(x))
    return df

parser = ArgumentParser(prog='data_combiner')
parser.add_argument('block_csv_file', type=str, nargs=1)
args = parser.parse_args()

csv_data_provider = CSVDataProvider(args.block_csv_file[0], lambda df: apply_datetime_transformations(df))
df = csv_data_provider.get_data()
df = df[df.duplicated('tstp', keep=False)].groupby('tstp')['energy(kWh/hh)'].apply(list).reset_index() # Eliminate all non-duplicate timestamps
print(df)
df['tstp'] = pd.to_datetime(df['tstp'])
df = df.set_index('tstp')
df = df.asfreq(freq='h', fill_value=0.0)
df = df.rename(columns={'energy(kWh/hh)': 'Energy'})

new_df = pd.DataFrame()
new_df.index = df.index

new_df_cols = {}

for i, row in df.iterrows():
    date = i
    energy_list = row.values[0]

    #print(date)
    #print(len(energy_list))

    for j in range(0, len(energy_list)):
        if 'energy' + str(j) not in new_df_cols:
            new_df_cols['energy' + str(j)] = []

        new_df_cols['energy' + str(j)].append(energy_list[j])

for key in new_df_cols.keys():
    new_df[key] = new_df_cols[key]

#print(new_df)