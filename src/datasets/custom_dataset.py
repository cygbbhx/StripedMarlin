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
import librosa

DF_ASVSPOOF_SPLIT = {
    "partition_ratio": [0.9, 0.1],
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
        self.full_anno_data = pd.read_csv(os.path.join(self.path, f'train_w_cl.csv'))

        self.noise_files = glob.glob(data_config["noise_path"])
        self.sample_rate = data_config["sample_rate"]
        self.duration = data_config["duration"] * self.sample_rate

        self.partition_ratio = DF_ASVSPOOF_SPLIT["partition_ratio"]
        self.seed = DF_ASVSPOOF_SPLIT["seed"]

        # self.anno_data = pd.read_csv(os.path.join(self.path, f'train.csv'))
        self.custom_spoofs = glob.glob('/home/work/StripedMarlin/sohyun/generate_spoof/generated/*.ogg')

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
        vc_filenames = [os.path.splitext(os.path.basename(file))[0] for file in self.custom_spoofs]
        flac_paths.update(dict(zip(noise_filenames, self.noise_files)))
        flac_paths.update(dict(zip(vc_filenames, self.custom_spoofs)))

        return flac_paths

    def resample(self):
        print("=> Resampling...")
        self.anno_data = self.cluster_based_sampling()
        self.flac_paths = self.get_file_references()
        self.samples = self.read_protocol()
        self.oversample_dataset()

        LOGGER.info(f"Spoof: {len(self.samples[self.samples['label'] == 'spoof'])}")
        LOGGER.info(f"Original: {len(self.samples[self.samples['label'] == 'bonafide'])}")
        LOGGER.info(f"Mixed: {len(self.samples[self.samples['label'] == 'both'])}")
        LOGGER.info(f"Noise: {len(self.samples[self.samples['label'] == 'noise'])}")
    
    def cluster_based_sampling(self):
        clusters = self.full_anno_data.groupby('cluster')
        cluster0 = clusters.get_group(0)
        cluster1 = clusters.get_group(1)
        cluster2 = clusters.get_group(2)
        
        cluster2_real = pd.DataFrame(cluster2.groupby('label').groups['real'])
        cluster2_fake = pd.DataFrame(cluster2.groupby('label').groups['fake'])
        
        real_sr = int(len(cluster2_real) * 0.1)
        # fake_sr = int(len(cluster2_real) * 0.1)
        us_real = cluster2_real.sample(n=real_sr)
        # us_fake = cluster2_fake.sample(n=fake_sr, random_state=42)
    
        combined_df = pd.concat([cluster0, us_real, cluster2_fake])
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
        mixed_real_samples = []
        mixed_fake_samples = []
        mixed_samples = []
        noise_samples = []

        single_sample_num = 5000
        mixing_sample_num = 5000
        vc_sample_num = 0

        all_real_samples = (self.path + '/' + self.anno_data.loc[self.anno_data['label'] == 'real', 'path']).tolist()
        single_real_samples = random.sample(all_real_samples, k=single_sample_num)
        remaining_real_samples = list(set(all_real_samples) - set(single_real_samples))
        mixing_real_samples = random.sample(remaining_real_samples, k=single_sample_num)

        all_fake_samples = (self.path + '/' + self.anno_data.loc[self.anno_data['label'] == 'fake', 'path']).tolist()
        new_fake_samples = self.custom_spoofs

        single_fake_samples = random.sample(all_fake_samples, k=single_sample_num - vc_sample_num) + random.sample(new_fake_samples, k=vc_sample_num)
        remaining_fake_samples = list(set(all_fake_samples) - set(single_fake_samples))
        mixing_fake_samples = random.sample(remaining_fake_samples, k=single_sample_num - vc_sample_num) + random.sample(new_fake_samples, k=vc_sample_num)

        for _ in range(single_sample_num):
            mixed_real_samples.append(random.sample(mixing_real_samples, 2))

        for _ in range(single_sample_num):
            mixed_fake_samples.append(random.sample(mixing_fake_samples, 2))

        for _ in range(single_sample_num * 2):
            real_sample = random.choice(mixing_real_samples)
            fake_sample = random.choice(mixing_fake_samples)
            mixed_samples.append([real_sample, fake_sample])

        real_samples.extend(single_real_samples)
        fake_samples.extend(single_fake_samples)
        noise_samples.extend(self.noise_files)

        samples = self._add_samples_to_dict(samples, fake_samples, "spoof")
        samples = self._add_samples_to_dict(samples, real_samples, "bonafide")
        samples = self._add_samples_to_dict(samples, mixed_real_samples, "mixed_bonafide")
        samples = self._add_samples_to_dict(samples, mixed_fake_samples, "mixed_spoof")
        samples = self._add_samples_to_dict(samples, mixed_samples, "mixed_both")
        samples = self._add_samples_to_dict(samples, noise_samples, "noise")
        
        return pd.DataFrame(samples)

    def _add_samples_to_dict(self, samples_dict, paths, label):
        split_paths = self.split_samples(paths)
        for path in split_paths:
            samples_dict = self.add_sample(samples_dict, path, label)
        return samples_dict

    def add_sample(self, samples, path, label):
        if 'mixed' in label:
            sample_name = os.path.basename(path[0]).split('.')[0] + '_' + os.path.basename(path[1]).split('.')[0]
            sample_path = list(path)
            label = label.split('mixed_')[-1]
        else:
            sample_name = os.path.basename(path).split('.')[0]
            sample_path = self.flac_paths[sample_name]

        samples["sample_name"].append(sample_name)
        samples["label"].append(label)

        samples["path"].append(sample_path)

        return samples

    def __getitem__(self, index) -> T_co:
        sample = self.samples.iloc[index]
        path = str(sample["path"])
        label = sample["label"]

        if type(sample["path"]) is list: # Mixed sample
            path1, path2 = sample["path"]
            waveform1, sample_rate1 = torchaudio.load(path1)
            waveform1, sample_rate1 = self.wavefake_preprocessing(
            waveform1, sample_rate1, wave_fake_sr=self.sample_rate, wave_fake_cut=self.duration
            )
            # waveform1 = apply_random_pitch_change(waveform1, sample_rate1)
            waveform2, sample_rate2 = torchaudio.load(path2)
            waveform2, sample_rate2 = self.wavefake_preprocessing(
            waveform2, sample_rate2, wave_fake_sr=self.sample_rate, wave_fake_cut=self.duration
            )
            # waveform2 = apply_random_pitch_change(waveform2, sample_rate2)

            seconds = self.duration // self.sample_rate
            waveform = self.mix_voices(waveform1, waveform2, sample_rate1, duration=seconds)
            sample_rate = sample_rate1
        else:
            waveform, sample_rate = torchaudio.load(path)
            waveform, sample_rate = self.wavefake_preprocessing(
            waveform, sample_rate, wave_fake_sr=self.sample_rate, wave_fake_cut=self.duration
            )
            # waveform = apply_random_pitch_change(waveform, sample_rate)

        p = np.random.rand()
        if label != "noise":
            if p < 0.3 and self.use_lowpass:
                waveform = self.lowpass_filter(waveform, sample_rate)
                waveform = waveform.squeeze(0)
            if p < 0.8:
                waveform = self.add_random_noise(waveform)
                waveform = waveform.squeeze(0)
        else:
            volume_change = random.choice([0, 1, 2.5, 5, 10])
            waveform = change_volume(waveform, volume_change)
            if p < 0.3:
                waveform = self.add_random_noise(waveform)
                waveform = waveform.squeeze(0)

        max_val = torch.max(torch.abs(waveform))
        if max_val > 1:
            normalized_waveform = waveform / max_val

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
    
    def mix_voices(self, waveform, target_waveform, sample_rate, duration=5):
        combined_length = duration * sample_rate
        min_length = sample_rate
        combined_waveform = torch.zeros(combined_length)

        volume_change = random.choice([-15, -10, -5, 0, 5])
        waveform = change_volume(waveform, volume_change)
        
        volume_change = random.choice([-15, -10, -5, 0, 5])
        target_waveform = change_volume(target_waveform, volume_change)

        combined_waveform += waveform + target_waveform

        return combined_waveform

    def add_random_noise(self, waveform):
        random_noise_file = random.choice(self.noise_files)
        noise = torchaudio.load(random_noise_file)
        noise_waveform, sample_rate = noise
        noise_waveform, sample_rate = self.wavefake_preprocessing(
                                        noise_waveform, sample_rate, wave_fake_sr=self.sample_rate, wave_fake_cut=self.duration
                                        )
        volume_change = random.choice([0, 1, 2.5, 5, 10])
        noise_waveform = change_volume(noise_waveform, volume_change)

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
            waveform += noise_waveform

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
        diff_length = spoof_length - noise_length
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

def apply_random_pitch_change(waveform, sample_rate):
    step = random.randint(-5, 5)
    # torchaudio.functional.pitch_shift()
    waveform = librosa.effects.pitch_shift(np.array(waveform[0]),sample_rate,n_steps=step)
    return torch.Tensor(waveform).unsqueeze(0)

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