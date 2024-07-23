from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import glob
import pandas 
import csv 
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

vad_data = []
output_csv = open('vad.csv', 'w')
writer = csv.writer(output_csv)
header = ['id', 'value', 'count']
writer.writerow(header)

p = pipeline(
    task=Tasks.voice_activity_detection, model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch')

test_files = glob.glob('/home/work/StripedMarlin/contest_data/test/*.wav')[:2]
result = p(test_files)

for r in tqdm(result):
    sample_id = os.path.basename(r['key'])
    writer.writerow([sample_id, r['value'], len(r['value'])])