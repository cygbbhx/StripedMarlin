import logging
from pathlib import Path

import pandas as pd

from src.datasets.base_dataset import SimpleAudioFakeDataset
import os

DF_ASVSPOOF_SPLIT = {
    "partition_ratio": [0.7, 0.15],
    "seed": 45
}

LOGGER = logging.getLogger()

class KoreanKaggleDataset(SimpleAudioFakeDataset):
    subset_parts = ("1", "2", "3", "4")

    def __init__(self, path, subset="train", transform=None):
        super().__init__(subset, transform)
        self.path = path

        self.partition_ratio = DF_ASVSPOOF_SPLIT["partition_ratio"]
        self.seed = DF_ASVSPOOF_SPLIT["seed"]

        self.flac_paths = self.get_file_references()
        self.samples = self.read_protocol()

        self.transform = transform
        LOGGER.info(f"Spoof: {len(self.samples[self.samples['label'] == 'spoof'])}")
        LOGGER.info(f"Original: {len(self.samples[self.samples['label'] == 'bonafide'])}")

    def get_file_references(self):
        flac_paths = {}
        for part in self.subset_parts:
            path = Path(self.path) / part
            flac_list = list(path.glob("*.wav"))

            for path in flac_list:
                flac_paths[path.stem] = path

        return flac_paths

    def read_protocol(self):
        samples = {
            "sample_name": [],
            "label": [],
            "path": []
        }

        real_samples = []
        fake_samples = []

        for path in self.flac_paths:
            # For KoreanKaggle , all is fake
            fake_samples.append(path)

        fake_samples = self.split_samples(fake_samples)
        for path in fake_samples:
            samples = self.add_sample(samples, path)

        # real_samples = self.split_samples(real_samples)
        # for line in real_samples:
        #     samples = self.add_line_to_samples(samples, line)

        return pd.DataFrame(samples)

    def add_sample(self, samples, path):
        sample_name = os.path.basename(path)
        label = "spoof"

        samples["sample_name"].append(sample_name)
        samples["label"].append(label)

        sample_path = self.flac_paths[sample_name]
        assert sample_path.exists()
        samples["path"].append(sample_path)
        samples["attack_type"] = "N/A"

        return samples


if __name__ == "__main__":
    KOREANKAGGLE_DATASET_PATH = "/home/work/StripedMarlin/kaggle-dataset/kss"

    real = 0
    fake = 0
    datasets = []

    for subset in ['train', 'test', 'val']:
        dataset = KoreanKaggleDataset(KOREANKAGGLE_DATASET_PATH, subset=subset)

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
