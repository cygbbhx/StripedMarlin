import pandas as pd
from datetime import datetime, timedelta, timezone
import os 

KST = timezone(timedelta(hours=9))
current_time_kst = datetime.now(KST)
submission_time = current_time_kst.strftime('%m%d-%H%M')

# Load the two CSV files
file1 = 'ensemble/ensemble_0718-1713.csv'
file2 = 'ensemble/ensemble_0719-0948.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

df1.sort_values(by='id', inplace=True)
df2.sort_values(by='id', inplace=True)

assert (df1['id'] == df2['id']).all(), "The IDs in the two files do not match."

df_ensemble = df1.copy()
df_ensemble['fake'] = (df1['fake'] + df2['fake']) / 2
df_ensemble['real'] = (df1['real'] + df2['real']) / 2

df_ensemble.to_csv(f'ensemble/ensemble_{submission_time}.csv', index=False)

print(f'Ensembled results saved')
