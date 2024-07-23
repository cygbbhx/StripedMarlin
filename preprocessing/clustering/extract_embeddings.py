import torch
import os
import pickle
from tqdm import tqdm
import numpy as np
import torchaudio
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='input_dir', help='path to train dataset directory', default='/home/work/StripedMarlin/contest_data/train')
args = parser.parse_args()

model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model.eval()

path = args.input_dir
dir_name = path.split('/')[-1]
ids = []
embeddings = []

for file in tqdm(os.listdir(path)):
    tensor, sr = torchaudio.load(os.path.join(path, file))
    tensor = tensor[0,:]
    data = tensor.squeeze(0)
    if data.shape[0] <= 32000:
        quot = 32000 // data.shape[0]
        data = data.repeat(quot+1)
    data = data.numpy()

    result = model.forward(data, sr)
    result_cpu = result.detach().cpu()
    if result_cpu.dim() == 2:
        result_cpu = torch.mean(result_cpu, dim=0)
    assert result_cpu.shape == torch.Size([128])

    ids.append(file)
    embeddings.append(result_cpu)

    del tensor, data, result
    torch.cuda.empty_cache()

embeddings_np = [t.numpy() for t in embeddings]
with open(f'{dir_name}_embedding.pkl', 'wb') as f:
    pickle.dump(embeddings_np, f)
with open(f'{dir_name}_ids.pkl', 'wb') as f:
    pickle.dump(ids, f)