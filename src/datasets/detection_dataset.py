import logging
from typing import List, Optional

import pandas as pd

from src.datasets.base_dataset import SimpleAudioFakeDataset
from src.datasets.deepfake_asvspoof_dataset import DeepFakeASVSpoofDataset
from src.datasets.fakeavceleb_dataset import FakeAVCelebDataset
from src.datasets.wavefake_dataset import WaveFakeDataset
from src.datasets.korean_dataset import KoreanKaggleDataset
from src.datasets.custom_dataset import CustomDataset


LOGGER = logging.getLogger()


class DetectionDataset(SimpleAudioFakeDataset):

    def __init__(
        self,
        asvspoof_path=None,
        wavefake_path=None,
        fakeavceleb_path=None,
        custom_path=None,
        subset: str = "val",
        transform=None,
        oversample: bool = True,
        undersample: bool = False,
        return_label: bool = True,
        reduced_number: Optional[int] = None,
        return_meta: bool = False,
        return_raw: bool = False
    ):
        super().__init__(
            subset=subset,
            transform=transform,
            return_label=return_label,
            return_meta=return_meta,
            return_raw=return_raw,
        )
        datasets = self._init_datasets(
            asvspoof_path=asvspoof_path,
            wavefake_path=wavefake_path,
            fakeavceleb_path=fakeavceleb_path,
            custom_path=custom_path,
            subset=subset,
        )
        self.samples = pd.concat(
            [ds.samples for ds in datasets],
            ignore_index=True
        )

        if oversample:
            self.oversample_dataset()
        elif undersample:
            self.undersample_dataset()

        if reduced_number:
            LOGGER.info(f"Using reduced number of samples - {reduced_number}!")
            self.samples = self.samples.sample(
                min(len(self.samples), reduced_number),
                random_state=42,
            )

    def _init_datasets(
        self,
        asvspoof_path: Optional[str],
        wavefake_path: Optional[str],
        fakeavceleb_path: Optional[str],
        custom_path: Optional[str],
        subset: str,
    ) -> List[SimpleAudioFakeDataset]:
        datasets = []

        # if asvspoof_path is not None:
        #     asvspoof_dataset = DeepFakeASVSpoofDataset(asvspoof_path, subset=subset)
        #     datasets.append(asvspoof_dataset)

        # if wavefake_path is not None:
        #     wavefake_dataset = WaveFakeDataset(wavefake_path, subset=subset)
        #     datasets.append(wavefake_dataset)

        # if fakeavceleb_path is not None:
        #     fakeavceleb_dataset = FakeAVCelebDataset(fakeavceleb_path, subset=subset)
        #     datasets.append(fakeavceleb_dataset)

        # if koreankaggle_path is not None:
        #     koreankaggle_dataset = KoreanKaggleDataset(koreankaggle_path, subset=subset)
        #     datasets.append(koreankaggle_dataset)
            
        if custom_path is not None:
            custom_dataset = CustomDataset(custom_path, subset=subset)
            datasets.append(custom_dataset)

        return datasets


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

        mixed_length = len(samples.groups["both"])
        diff_length = bona_length - mixed_length

        if diff_length > 0:
            mixed = samples.get_group("both").sample(diff_length, replace=True)
            self.samples = pd.concat([self.samples, mixed], ignore_index=True)

    def undersample_dataset(self):
        samples = self.samples.groupby(by=['label'])
        bona_length = len(samples.groups["bonafide"])
        spoof_length = len(samples.groups["spoof"])

        if spoof_length < bona_length:
            raise NotImplementedError

        if spoof_length > bona_length:
            spoofs = samples.get_group("spoof").sample(bona_length, replace=True)
            self.samples = pd.concat([samples.get_group("bonafide"), spoofs], ignore_index=True)

    def get_bonafide_only(self):
        samples = self.samples.groupby(by=['label'])
        self.samples = samples.get_group("bonafide")
        return self.samples

    def get_spoof_only(self):
        samples = self.samples.groupby(by=['label'])
        self.samples = samples.get_group("spoof")
        return self.samples

