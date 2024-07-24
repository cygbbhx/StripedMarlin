import pandas as pd
import csv
from datetime import datetime, timedelta, timezone
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Rewrite CSV based on vad.csv reference.')
parser.add_argument('prev_file', type=str, help='Path to the previous CSV file')
parser.add_argument('vad_file', type=str, help='Path to the vad.csv file', default='custom_data/vad.csv')
parser.add_argument('out_file', type=str, help='Path to the output CSV file')
args = parser.parse_args()

# Load the CSV files
prev_subm = pd.read_csv(args.prev_file)
ref = pd.read_csv(args.vad_file)

# Define the timezone
KST = timezone(timedelta(hours=9))
current_time_kst = datetime.now(KST)
submission_time = current_time_kst.strftime('%m%d-%H%M')

# Prepare output file
out_dir = os.path.dirname(args.out_file)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

with open(args.out_file, 'w', newline='') as new_subm:
    writer = csv.writer(new_subm)
    header = ['id', 'fake', 'real']
    writer.writerow(header)

    new = []
    for i, row in prev_subm.iterrows():
        if ref.iloc[i]['count'] == 0:
            row['fake'] = 0
            row['real'] = 0
        new.append(row)
    
    for row in new:
        writer.writerow(row)

print(f'Rewritten CSV saved to {args.out_file}')
