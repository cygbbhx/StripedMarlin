import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import yaml

from src.datasets.detection_dataset import DetectionDataset
from src.models import models
from src.trainer import GDTrainer
from src.utils import set_seed
import wandb
import os
from src.datasets.custom_dataset import CustomDataset
from datetime import datetime, timedelta, timezone

import warnings
warnings.filterwarnings('ignore')

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
LOGGER.addHandler(ch)
WANDB_AUTH_KEY = os.getenv('WANDB_AUTH_KEY')

def save_model(
    model: torch.nn.Module,
    model_dir: Union[Path, str],
    name: str,
    ckpt_name: str = "ckpt"
) -> None:
    full_model_dir = Path(f"{model_dir}/{name}")
    full_model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{full_model_dir}/{ckpt_name}.pth")


def get_datasets(
    data_config,
    datasets_paths: List[Union[Path, str]],
    amount_to_use: Optional[Tuple[int, int]]
# ) -> Tuple[DetectionDataset, DetectionDataset]:
) -> Tuple[CustomDataset, CustomDataset]:

    data_train = CustomDataset(
        path=datasets_paths[3],
        data_config=data_config,
        subset="train",
        reduced_number=amount_to_use[0],
        oversample=True
    )

    data_test = CustomDataset(
        path=datasets_paths[3],
        data_config=data_config,
        subset="test",
        reduced_number=amount_to_use[1],
        oversample=True
    )
    
    return data_train, data_test


def train_nn(
    datasets_paths: List[Union[Path, str]],
    wandb: bool,
    batch_size: int,
    epochs: int,
    device: str,
    config: Dict,
    model_dir: Optional[Path] = None,
    amount_to_use: Optional[Tuple[int, int]] = None,
    config_save_path: str = "configs",
) -> None:

    LOGGER.info("Loading data...")
    model_config = config["model"]
    data_config = config["data"]
    model_name, model_parameters = model_config["name"], model_config["parameters"]
    optimizer_config = model_config["optimizer"]

    # timestamp = time.time()

    KST = timezone(timedelta(hours=9))
    current_time_kst = datetime.now(KST)
    timestamp = current_time_kst.strftime('%m%d-%H%M')

    checkpoint_path = ""

    data_train, data_test = get_datasets(
        data_config=data_config,
        datasets_paths=datasets_paths,
        amount_to_use=amount_to_use,
    )

    current_model = models.get_model(
        model_name=model_name,
        config=model_parameters,
        device=device,
    ).to(device)

    use_scheduler = config['model']['loss']['scheduler']

    LOGGER.info(f"Training '{model_name}' model on {len(data_train)} audio files.")

    current_model = GDTrainer(
        device=device,
        batch_size=batch_size,
        epochs=epochs,
        optimizer_kwargs=optimizer_config,
        use_scheduler=use_scheduler,
        wandb=wandb
    ).train(
        dataset=data_train,
        model=current_model,
        test_dataset=data_test,
        model_dir=model_dir,
        # save_model_name=f"aad__{model_name}__{timestamp}"
        save_model_name=f"{timestamp}_{model_name}",
        loss_config=model_config["loss"]
    )

    if model_dir is not None:
        save_name = f"{timestamp}_{model_name}"
        save_model(
            model=current_model,
            model_dir=model_dir,
            name=save_name,
        )
        checkpoint_path= str(model_dir.resolve() / save_name / "ckpt.pth")

    LOGGER.info(f"Training done!")

    # Save config for testing
    if model_dir is not None:
        config["checkpoint"] = {"path": checkpoint_path}
        config_name = f"{timestamp}_{model_name}.yaml"
        config_save_path = str(Path(config_save_path) / config_name)
        with open(config_save_path, "w") as f:
            yaml.dump(config, f)
        LOGGER.info("Test config saved at location '{}'!".format(config_save_path))


def main():
    args = parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.wandb:
        # wandb.login(key=WANDB_AUTH_KEY)
        KST = timezone(timedelta(hours=9))
        current_time_kst = datetime.now(KST)
        timestamp = current_time_kst.strftime('%m%d-%H%M')
        if args.sweep:
            run = wandb.init(name=f'{timestamp}')
            # config["model"]["name"] = wandb.config.model
            config["model"]["optimizer"]["lr"] = wandb.config.lr
            # config["model"]["parameters"]["frontend_algorithm"] = wandb.config.frontend
            
            # config["data"]["use_rir"] = wandb.config.rir
            # config["data"]["use_bg"] = wandb.config.bg
            # config["data"]["use_lowpass"] = wandb.config.lowpass

            # loss_name = wandb.config.loss_name
            # if 'focal' in loss_name:
            #     alpha_val = int(loss_name.split('_')[-1]) / 100
            #     loss_name = 'focal'
            #     config["model"]["loss"]["alpha"] = alpha_val
            
            # config["model"]["loss"]["name"] = loss_name
            config["model"]["loss"]["gamma"] = wandb.config.gamma
            config["model"]["loss"]["alpha"] = wandb.config.alpha
            config["model"]["loss"]["fake_weight"] = wandb.config.fake_weight
            # config["model"]["loss"]["use_reg"] = wandb.config.reg

            
        else:
            model_name = config["model"]["name"]
            wandb.init(entity="cygbbhx", project="WavLM-pretrain-batchsize4", config=config, name=f"{timestamp}-{model_name}")

    seed = config["data"].get("seed", 42)
    # fix all seeds
    set_seed(seed)

    if not args.cpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model_dir = Path(args.ckpt)
    model_dir.mkdir(parents=True, exist_ok=True)

    if config["model"]["name"] in ["rawnet3", "HuBERT"]:
        args.batch_size = args.batch_size // 2

    train_nn(
        # datasets_paths=[args.asv_path, args.wavefake_path, args.celeb_path, args.custom_path],
        datasets_paths=[None, None, None, args.custom_path],
        device=device,
        amount_to_use=(args.train_amount, args.test_amount),
        batch_size=args.batch_size,
        epochs=args.epochs,
        model_dir=model_dir,
        config=config,
        wandb=args.wandb
    )


def parse_args():
    parser = argparse.ArgumentParser()

    ASVSPOOF_DATASET_PATH = "/home/work/StripedMarlin/ASVspoof2021/ASVspoof2021_DF_eval"
    # ASVSPOOF_DATASET_PATH = "/home/adminuser/storage/datasets/deep_fakes/ASVspoof2021/DF"
    WAVEFAKE_DATASET_PATH = None
    # WAVEFAKE_DATASET_PATH = "/home/adminuser/storage/datasets/deep_fakes/WaveFake"
    CUSTOM_DATASET_PATH = "/home/work/StripedMarlin/contest_data"

    FAKEAVCELEB_DATASET_PATH = None
    FAKEAVCELEB_DATASET_PATH = "/home/adminuser/storage/datasets/deep_fakes/FakeAVCeleb/FakeAVCeleb_v1.2"

    parser.add_argument(
        "--asv_path",
        type=str,
        default=ASVSPOOF_DATASET_PATH,
        help="Path to ASVspoof2021 dataset directory",
    )
    parser.add_argument(
        "--wavefake_path",
        type=str,
        default=WAVEFAKE_DATASET_PATH,
        help="Path to WaveFake dataset directory",
    )
    parser.add_argument(
        "--celeb_path",
        type=str,
        default=FAKEAVCELEB_DATASET_PATH,
        help="Path to FakeAVCeleb dataset directory",
    )

    parser.add_argument("--custom_path", type=str, default=CUSTOM_DATASET_PATH)
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--sweep", action='store_true')

    default_model_config = "config.yaml"
    parser.add_argument(
        "--config",
        help="Model config file path (default: config.yaml)",
        type=str,
        default=default_model_config,
    )

    default_train_amount = None
    parser.add_argument(
        "--train_amount",
        "-a",
        help=f"Amount of files to load for training.",
        type=int,
        default=default_train_amount,
    )

    default_test_amount = None
    parser.add_argument(
        "--test_amount",
        "-ta",
        help=f"Amount of files to load for testing.",
        type=int,
        default=default_test_amount,
    )

    default_batch_size = 4
    parser.add_argument(
        "--batch_size",
        "-b",
        help=f"Batch size (default: {default_batch_size}).",
        type=int,
        default=default_batch_size,
    )

    default_epochs = 20
    parser.add_argument(
        "--epochs",
        "-e",
        help=f"Epochs (default: {default_epochs}).",
        type=int,
        default=default_epochs,
    )

    default_model_dir = "trained_models"
    parser.add_argument(
        "--ckpt",
        help=f"Checkpoint directory (default: {default_model_dir}).",
        type=str,
        default=default_model_dir,
    )

    parser.add_argument("--cpu", "-c", help="Force using cpu?", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    # main()
    args = parse_args()
    if args.wandb:
        wandb.login(key=WANDB_AUTH_KEY)
        if args.sweep:
            sweep_configuration = {
                'method': 'random',
                'name': 'WavLM pretrain experiments',
                'metric': {'goal': 'minimize', 'name': 'best_test_score'},
                'parameters': 
                {
                    # 'model': {'values': ['lcnn', 'rawnet3', 'specrnet']},
                    # 'loss_name': {'values': ['focal_50', 'focal_75', 'focal_80']},
                    'gamma': {'values': [2.0, 3.0]},
                    'lr': {'values': [0.0001, 0.0002, 0.00005]},
                    'alpha': {'values': [0.50, 0.75, 0.80]},
                    'fake_weight': {'values': [0.7, 0.8, 2.0]},
                    # 'rir': {'values': [True, False]},
                    # 'bg': {'values': [True, False]},
                    # 'lowpass': {'values': [True, False]}
                    # 'sam': {'values': [True, False]}
                }
            }

            # Initialize sweep by passing in config. (Optional) Provide a name of the project.
            sweep_id = wandb.sweep(sweep=sweep_configuration, entity='cygbbhx', project='StripedMarlin')
            wandb.agent(sweep_id, function=main, count=25)
        else:
            main()
    else:
        main()