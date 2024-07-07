import argparse
from src.datasets.base_dataset import SimpleAudioFakeDataset
from torch.utils.data.dataset import T_co
import torchaudio
import logging
import pandas as pd 
import os 
import torch 
from pathlib import Path
from typing import Dict, List, Optional, Union
from src import metrics, utils
from src.models import models
from torch.utils.data import DataLoader
from datetime import datetime, timedelta, timezone
import csv
import yaml
import noisereduce as nr
import numpy as np

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

class InferenceDataset(SimpleAudioFakeDataset):
    def __init__(self, path, subset="test", transform=None, data_config=None):
        super().__init__(subset, transform)
        self.path = path
        self.samples = pd.read_csv(os.path.join(self.path, f'test.csv'))
    
        if data_config is None:
            preprocess_sr = 16_000
            duration = 5
        else:
            preprocess_sr = data_config.get("sample_rate", 16_000)
            duration = data_config.get("duration", 5)
            cut_length = duration * preprocess_sr
            
        print(f"using {preprocess_sr} with {duration} seconds")
        self.preprocess_sr = preprocess_sr
        self.cut_length = cut_length
                
    def __getitem__(self, index) -> T_co:
        sample = self.samples.iloc[index]
        sample_path = os.path.join(self.path, sample['path'])
        sample_path = os.path.join(self.path, sample['path'].replace('.ogg', '.wav'))

        waveform, sample_rate = torchaudio.load(sample_path, normalize=True)
        waveform, sample_rate = self.wavefake_preprocessing(
            waveform, sample_rate, wave_fake_sr=self.preprocess_sr, wave_fake_cut=self.cut_length
        )

        return_data = [sample['id'], waveform, sample_rate]
        return return_data
        
def run_inference(
    model_path: Optional[Path],
    data_path: Path,
    model_config: Dict,
    data_config:Dict,
    device: str,
    config_name: str,
    batch_size: int = 128,
):
    header = ['id', 'fake', 'real']
    KST = timezone(timedelta(hours=9))
    current_time_kst = datetime.now(KST)
    submission_time = current_time_kst.strftime('%m%d-%H%M')

    model_name, model_parameters = model_config["name"], model_config["parameters"]

    weights_path = ""

    # Load model architecture
    model = models.get_model(
        model_name=model_name,
        config=model_parameters,
        device=device,
    )
    # If provided weights, apply corresponding ones (from an appropriate fold)
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    dataset = InferenceDataset(path=data_path, data_config=data_config, subset='test')
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    batches_number = len(dataset) // batch_size
    id_list = []
    y_pred = torch.Tensor([]).to(device)

    for i, (batch_id, batch_x, _) in enumerate(dataloader):
        model.eval()
        if i % 10 == 0:
            LOGGER.info(f"Batch [{i}/{batches_number}]")

        with torch.no_grad():
            batch_x = batch_x.to(device)

            batch_pred = model(batch_x).squeeze(1)
            batch_pred = torch.sigmoid(batch_pred)

            id_list += [id_value for id_value in batch_id]
            y_pred = torch.concat([y_pred, batch_pred], dim=0)
            
    
    if data_path == "/home/work/StripedMarlin/sohyun/separate_voice/test_processed":
        data_name = 'mossformer'
    elif data_path == "/home/work/StripedMarlin/sohyun/test_cleaned":
        data_name = 'vocal_remover'
    elif data_path == "/home/work/StripedMarlin/sohyun/noise_reducer":
        data_name = "noise_reducer"
    else:
        data_name = 'raw'

    preprocess_sr = data_config.get("sample_rate", 16_000)
    duration = data_config.get("duration", 5)
    
    output_csv = open(f'submissions/inf_{config_name}_{submission_time}_{data_name}_{preprocess_sr//1000}-{duration}.csv', 'w') 
    writer = csv.writer(output_csv)
    writer.writerow(header)
    for i in range(len(id_list)):
        # writer.writerow([id_list[i], 1 - y_pred[i].item(), y_pred[i].item()]) # for single sigmoid output
        writer.writerow([id_list[i], y_pred[i][0].item(), y_pred[i][1].item()])

def main(args):

    if not args.cpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config_name = os.path.basename(args.config).split('.yaml')[0]

    seed = config["data"].get("seed", 42)
    # fix all seeds - this should not actually change anything
    utils.set_seed(seed)
    
    run_inference(
        model_path=config["checkpoint"].get("path", ""),
        data_path=args.data_path,
        model_config=config["model"],
        data_config=config["data"],
        config_name=config_name,
        device=device,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    # DATA_PATH = "/home/work/StripedMarlin/sohyun/separate_voice/test_processed"
    DATA_PATH = "/home/work/StripedMarlin/sohyun/test_cleaned"
    # DATA_PATH = "/home/work/StripedMarlin/contest_data"
    # DATA_PATH = "/home/work/StripedMarlin/sohyun/noise_reducer"
        
    print(f"using data from {DATA_PATH}")
    parser.add_argument("--data_path", type=str, default=DATA_PATH)

    default_model_config = "config.yaml"
    parser.add_argument(
        "--config",
        help="Model config file path (default: config.yaml)",
        type=str,
        default=default_model_config,
    )
    
    parser.add_argument("--cpu", "-c", help="Force using cpu", action="store_true")

    return parser.parse_args()
    
    
if __name__ == "__main__":
    main(parse_args())
