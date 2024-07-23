import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import argparse

# Define the timezone
KST = timezone(timedelta(hours=9))
current_time_kst = datetime.now(KST)
submission_time = current_time_kst.strftime('%m%d-%H%M')

# Set up argument parser
parser = argparse.ArgumentParser(description='Ensemble two CSV files.')
parser.add_argument('file1', type=str, help='Path to the first CSV file')
parser.add_argument('file2', type=str, help='Path to the second CSV file')
args = parser.parse_args()

# Load the CSV files
df1 = pd.read_csv(args.file1)
df2 = pd.read_csv(args.file2)

# Sort dataframes by 'id' column
df1.sort_values(by='id', inplace=True)
df2.sort_values(by='id', inplace=True)

# Ensure the 'id' columns match
assert (df1['id'] == df2['id']).all(), "The IDs in the two files do not match."

# Create the ensemble dataframe
df_ensemble = df1.copy()
df_ensemble['fake'] = (df1['fake'] + df2['fake']) / 2
df_ensemble['real'] = (df1['real'] + df2['real']) / 2

# Save the ensembled dataframe to a new CSV file
output_path = f'outputs/ensemble_{submission_time}.csv'
df_ensemble.to_csv(output_path, index=False)

print(f'Ensembled results saved to {output_path}')
