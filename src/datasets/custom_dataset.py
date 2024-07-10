import logging
from pathlib import Path

import pandas as pd
from typing import List, Optional

from src.datasets.base_dataset import SimpleAudioFakeDataset
import os
import torch
import torchaudio.functional as F
import numpy as np
from torch.utils.data.dataset import T_co
import glob 
import torchaudio
import random

DF_ASVSPOOF_SPLIT = {
    "partition_ratio": [0.8, 0.1],
    "seed": 45
}

LOGGER = logging.getLogger()

class CustomDataset(SimpleAudioFakeDataset):

    def __init__(
        self,
        path,
        data_config,
        subset="train",
        transform=None,
        oversample: bool = True,
        undersample: bool = False,
        return_label: bool = True,
        reduced_number: Optional[int] = None,
        ):
        super().__init__(subset, transform)
        self.path = path
        self.data_config = data_config
        self.anno_data = pd.read_csv(os.path.join(self.path, f'train_w_cl.csv'))

        self.noise_files = glob.glob(data_config["noise_path"])
        self.sample_rate = data_config["sample_rate"]
        self.duration = data_config["duration"] * self.sample_rate + 500 

        self.partition_ratio = DF_ASVSPOOF_SPLIT["partition_ratio"]
        self.seed = DF_ASVSPOOF_SPLIT["seed"]

        self.anno_data = self.cluster_based_sampling()
        self.flac_paths = self.get_file_references()
        self.samples = self.read_protocol()

        self.transform = transform
        
        self.use_rir = self.data_config.get("use_rir", False)
        self.use_bg = self.data_config.get("use_bg", False)
        self.use_lowpass = self.data_config.get("use_lowpass", False)

        LOGGER.info(f"Spoof: {len(self.samples[self.samples['label'] == 'spoof'])}")
        LOGGER.info(f"Original: {len(self.samples[self.samples['label'] == 'bonafide'])}")
        LOGGER.info(f"Mixed: {len(self.samples[self.samples['label'] == 'both'])}")
        LOGGER.info(f"Noise: {len(self.samples[self.samples['label'] == 'noise'])}")

        if oversample:
            self.oversample_dataset()
        elif undersample:
            self.undersample_dataset()

        LOGGER.info(f"==> Oversampled." \
                    f"Spoof: {len(self.samples[self.samples['label'] == 'spoof'])} | " \
                    f"Original: {len(self.samples[self.samples['label'] == 'bonafide'])} | " \
                    f"Mixed: {len(self.samples[self.samples['label'] == 'both'])} | " \
                    f"Noise: {len(self.samples[self.samples['label'] == 'noise'])}")

        if reduced_number:
            LOGGER.info(f"Using reduced number of samples - {reduced_number}!")
            self.samples = self.samples.sample(
                min(len(self.samples), reduced_number),
                random_state=42,
            )
            
        LOGGER.info(f"lowpass: {self.use_lowpass} | RIR: {self.use_rir} | BG: {self.use_bg}")

    def get_file_references(self):
        flac_paths = {}
        flac_paths.update(dict(zip(self.anno_data['id'], self.path + '/' + self.anno_data['path'])))
        noise_filenames = [os.path.splitext(os.path.basename(file))[0] for file in self.noise_files]
        flac_paths.update(dict(zip(noise_filenames, self.noise_files)))

        return flac_paths
    
    def cluster_based_sampling(self):
        clusters = self.anno_data.groupby('cluster')
        cluster0 = clusters.get_group(0)
        cluster1 = clusters.get_group(1)
        cluster2 = clusters.get_group(2)
        
        cluster2_real = pd.DataFrame(cluster2.groupby('label').groups['real'])
        cluster2_fake = pd.DataFrame(cluster2.groupby('label').groups['fake'])
        
        real_sr = int(len(cluster2_real) * 0.05)
        fake_sr = int(len(cluster2_real) * 0.1)
        us_real = cluster2_real.sample(n=real_sr, random_state=42)
        us_fake = cluster2_fake.sample(n=fake_sr, random_state=42)
    
        combined_df = pd.concat([cluster0, us_real, us_fake])
        combined_df['cluster'] = combined_df['cluster'].replace(2, 1)
        
        final_df = pd.concat([combined_df, cluster1])
        
        return final_df.drop(columns=['cluster',0])

    def read_protocol(self):
        samples = {
            "sample_name": [],
            "label": [],
            "path": [],
            # "cluster": []
        }

        real_samples = []
        fake_samples = []
        noise_samples = []

        real_samples.extend((self.path + '/' + self.anno_data.loc[self.anno_data['label'] == 'real', 'path']).tolist())
        fake_samples.extend((self.path + '/' + self.anno_data.loc[self.anno_data['label'] == 'fake', 'path']).tolist())

        # Handle new_data
        noise_samples.extend(self.noise_files)

        fake_samples = self.split_samples(fake_samples)
        for path in fake_samples:
            samples = self.add_sample(samples, path, label="spoof")

        real_samples = self.split_samples(real_samples)
        for path in real_samples:
            samples = self.add_sample(samples, path, label="bonafide")


        noise_samples = self.split_samples(noise_samples)
        for path in noise_samples:
            samples = self.add_sample(samples, path, label="noise")

        return pd.DataFrame(samples)

    def add_sample(self, samples, path, label):
        sample_name = os.path.basename(path).split('.')[0]

        samples["sample_name"].append(sample_name)
        samples["label"].append(label)

        sample_path = self.flac_paths[sample_name]
        samples["path"].append(sample_path)

        return samples

    def __getitem__(self, index) -> T_co:
        sample = self.samples.iloc[index]
        path = str(sample["path"])
        label = sample["label"]

        waveform, sample_rate = torchaudio.load(path)

        if label != "noise":
            p = np.random.rand()
            if np.random.rand() < 0.5:
                mix_sample = self.samples.sample(1).iloc[0]
                mix_sample_path = str(mix_sample['path'])
                target_waveform, sample_rate = torchaudio.load(mix_sample_path)
                target_label = mix_sample['label']
                
                if target_label == "bonafide" and label == "bonafide":
                    label = "bonafide"
                elif  target_label == "spoof" and label == "spoof":
                    label = "spoof"
                else:
                    label = "both"
                    
                waveform = self.mix_voice(waveform, target_waveform)
            
            if p < 0.3 and self.use_lowpass:
                waveform = self.lowpass_filter(waveform, sample_rate)
            if p < 0.8:
                waveform = self.add_random_noise(waveform)
        
        # normalize and resampling is done below code               
        waveform, sample_rate = self.wavefake_preprocessing(
            waveform, sample_rate, wave_fake_sr=self.sample_rate, wave_fake_cut=self.duration
        )

        label_mapping = {
            "noise": [0,0],
            "both": [1,1],
            "bonafide": [0,1],
            "spoof": [1,0]
        }

        label = label_mapping[label]
        return [waveform, sample_rate, label]

    @staticmethod
    def lowpass_filter(waveform, sample_rate):
        effect = ",".join(
            [
                "lowpass=frequency=300:poles=1",  # apply single-pole lowpass filter
                "atempo=0.8",  # reduce the speed
                "aecho=in_gain=0.8:out_gain=0.9:delays=200:decays=0.3|delays=400:decays=0.3"
                # Applying echo gives some dramatic feeling
            ],
        )
        effector = torchaudio.io.AudioEffector(effect=effect)
        return effector.apply(waveform.T, sample_rate).T

    @staticmethod
    def rir_effect(speech_waveform, noise):
        waveform, sample_rate = noise
        rir = waveform[:, int(sample_rate * 1.01) : int(sample_rate * 1.3)]
        rir = rir / torch.linalg.vector_norm(rir, ord=2)
        augmented = F.fftconvolve(speech_waveform, rir)
        return augmented

    @staticmethod
    def background_effect(speech_waveform, noise_waveform):
        end = min(noise_waveform.shape[1], speech_waveform.shape[1])
        noise_waveform = noise_waveform[:, :end]
        snr_dbs = torch.tensor([20, 10, 3])
        noisy_speeches = F.add_noise(speech_waveform[:, :end], noise_waveform, snr_dbs) # TODO: modify to random position

        return noisy_speeches
    
    def mix_voice(self, waveform, target_waveform):        
        end = min(waveform.shape[1], target_waveform.shape[1])
    
        if random.choice([True, False]):
            volume_change = random.randint(-10, 10)
            waveform = change_volume(waveform, volume_change)
        else:
            volume_change = random.randint(-10, 10) 
            target_waveform = change_volume(target_waveform, volume_change)

        shorter, longer = (waveform, target_waveform) if waveform.shape[1] <= target_waveform.shape[1] else (target_waveform, waveform)
        start_position = random.randrange(max(1,longer.shape[1] - shorter.shape[1]))      
        end = min(end, start_position + shorter.shape[1])
        
        waveform[:, start_position:end] += target_waveform[:, start_position:end]
        
        return waveform

    def add_random_noise(self, waveform):
        random_noise_file = random.choice(self.noise_files)
        noise = torchaudio.load(random_noise_file)
        noise_waveform, _ = noise
        noise_choice_prob = np.random.rand()

        if noise_choice_prob < 0.4 and self.use_rir:
            waveform = self.rir_effect(waveform, noise)
        elif noise_choice_prob < 0.8 and self.use_bg:
            noisy_speeches = self.background_effect(waveform, noise_waveform)
            prob = np.random.rand()

            if prob < 0.5:
                waveform = noisy_speeches[0:1]
            elif prob < 0.8:
                waveform = noisy_speeches[1:2]
            else:
                waveform = noisy_speeches[2:3]
        else:
            end = min(noise_waveform.shape[1], waveform.shape[1])
            waveform[:, :end] += noise_waveform[:, :end]

        return waveform

    def oversample_dataset(self):
        samples = self.samples.groupby(by=['label'])
        bona_length = len(samples.groups["bonafide"])
        spoof_length = len(samples.groups["spoof"])

        diff_length = spoof_length - bona_length

        if diff_length < 0:
            raise NotImplementedError

        if diff_length > 0:
            bonafide = samples.get_group("bonafide").sample(diff_length, replace=True)
            self.samples = pd.concat([self.samples, bonafide], ignore_index=True)

        # mixed_length = len(samples.groups["both"])
        # diff_length = spoof_length - mixed_length

        # if diff_length > 0:
        #     mixed = samples.get_group("both").sample(diff_length, replace=True)
        #     self.samples = pd.concat([self.samples, mixed], ignore_index=True)

        noise_length = len(samples.groups["noise"])
        diff_length = spoof_length // 4 - noise_length
        if diff_length > 0:
            noise = samples.get_group("noise").sample(diff_length, replace=True)
            self.samples = pd.concat([self.samples, noise], ignore_index=True)

    def undersample_dataset(self):
        samples = self.samples.groupby(by=['label'])
        bona_length = len(samples.groups["bonafide"])
        spoof_length = len(samples.groups["spoof"])
        both_length = len(samples.groups["both"])

        # if spoof_length > bona_length:
        #     spoofs = samples.get_group("spoof").sample(bona_length, replace=True)
        #     self.samples = pd.concat([samples.get_group("bonafide"), spoofs], ignore_index=True)
        spoofs = samples.get_group("spoof").sample(both_length // 2, replace=True)
        bonas = samples.get_group("bonafide").sample(both_length // 2, replace=True)

        self.samples = pd.concat([samples.get_group("mixed"), samples.get_group("noise"), spoofs, bonas], ignore_index=True)


if __name__ == "__main__":
    DATASET_PATH = "/home/work/StripedMarlin/contest_data"

    real = 0
    fake = 0
    datasets = []

    for subset in ['train', 'test', 'val']:
        dataset = CustomDataset(DATASET_PATH, subset=subset)

        real_samples = dataset.samples[dataset.samples['label'] == 'bonafide']
        real += len(real_samples)

        print('real', len(real_samples))

        spoofed_samples = dataset.samples[dataset.samples['label'] == 'spoof']
        fake += len(spoofed_samples)

        print('fake', len(spoofed_samples))

        datasets.append(dataset)

    paths_0 = [str(p) for p in datasets[0].samples.path.values]  # pathlib -> str
    paths_1 = [str(p) for p in datasets[1].samples.path.values]
    paths_2 = [str(p) for p in datasets[2].samples.path.values]

    assert len(paths_0) == len(set(paths_0)), "duplicated paths in subset"
    assert len(paths_1) == len(set(paths_1)), "duplicated paths in subset"
    assert len(paths_2) == len(set(paths_2)), "duplicated paths in subset"

    assert len(set(paths_0).intersection(set(paths_1))) == 0, "duplicated paths"
    assert len(set(paths_1).intersection(set(paths_2))) == 0, "duplicated paths"
    assert len(set(paths_0).intersection(set(paths_2))) == 0, "duplicated paths"

    print("All correct!")

    # TODO(PK): points to fulfill
    # [ ] each attack type should be present in each subset
    # [x] no duplicates


def change_volume(waveform, decibel_change):
    return waveform * (10 ** (decibel_change / 20))