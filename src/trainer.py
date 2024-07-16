"""A generic training wrapper."""
import functools
import logging
import random
from copy import deepcopy
from pathlib import Path
from typing import Callable, List, Optional, Union, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.aa import utils
from src.aa.aa_types import AttackEnum
from src.loss import CombinedBrierFocalLoss
from src.sam import SAM
import wandb
from eval_utils import get_metric
import torchvision

LOGGER = logging.getLogger(__name__)


def save_model(
    model: torch.nn.Module,
    model_dir: Union[Path, str],
    name: str,
    epoch: Optional[int] = None,
    ckpt_name: str = None
) -> None:
    full_model_dir = Path(f"{model_dir}/{name}")
    full_model_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_name:
        ckpt_name = f"ckpt{epoch_str}"

    if epoch is not None:
        epoch_str = f"_{epoch:02d}"
    else:
        epoch_str = ""
    torch.save(model.state_dict(), f"{full_model_dir}/{ckpt_name}.pth")
    LOGGER.info(f"Training model saved under: {full_model_dir}/{ckpt_name}.pth")


class Trainer():
    """This is a lightweight wrapper for training models with gradient descent.

    Its main function is to store information about the training process.

    Args:
        epochs (int): The amount of training epochs.
        batch_size (int): Amount of audio files to use in one batch.
        device (str): The device to train on (Default 'cpu').
        batch_size (int): The amount of audio files to consider in one batch (Default: 32).
        optimizer_fn (Callable): Function for constructing the optimzer.
        optimizer_kwargs (dict): Kwargs for the optimzer.
    """

    def __init__(
        self,
        wandb,
        epochs: int = 20,
        batch_size: int = 32,
        device: str = "cpu",
        optimizer_fn: Callable = torch.optim.Adam,
        optimizer_kwargs: dict = {"lr": 1e-3},
        use_scheduler: bool = False,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.optimizer_fn = optimizer_fn
        self.optimizer_kwargs = optimizer_kwargs
        self.epoch_test_losses: List[float] = []
        self.use_scheduler = use_scheduler
        self.wandb = wandb


def forward_and_loss(model, criterion, batch_x, batch_y, **kwargs):
    output = model(batch_x)
    batch_out = torch.sigmoid(output)
    loss_config = kwargs["loss_config"]
    
    fake_weight = loss_config["fake_weight"]

    if fake_weight == 0.5:
        batch_loss = criterion(batch_out, batch_y)
    else:
        real_weight = 1 - fake_weight if fake_weight < 1 else 1
        fake_loss = criterion(batch_out[:,0], batch_y[:,0])
        real_loss = criterion(batch_out[:,1], batch_y[:,1])
        batch_loss = fake_weight * fake_loss + real_weight * real_loss

    return batch_out, batch_loss


class GDTrainer(Trainer):

    def train(
        self,
        dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
        model_dir: Optional[str] = None,
        save_model_name: Optional[str] = None,
        loss_config: Optional[Dict] = None,
        test_len: Optional[float] = None,
        test_dataset: Optional[torch.utils.data.Dataset] = None,
    ):
        patience = []

        if test_dataset is not None:
            train = dataset
            test = test_dataset
        else:
            test_len = int(len(dataset) * test_len)
            train_len = len(dataset) - test_len
            lengths = [train_len, test_len]
            train, test = torch.utils.data.random_split(dataset, lengths)

        train_loader = DataLoader(
            train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=6,
        )
        test_loader = DataLoader(
            test,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=6,
        )
        
        loss_map = {
            "bce": torch.nn.BCEWithLogitsLoss(),
            "focal": CombinedBrierFocalLoss(alpha=loss_config.get("alpha", 0.25), gamma=loss_config.get("gamma", 2.0))
        }
        
        criterion = loss_map[loss_config["name"]]

        optim = self.optimizer_fn(model.parameters(), **self.optimizer_kwargs)
        if loss_config.get('sam', False):
            optim = SAM(model.parameters(), self.optimizer_fn, lr=self.optimizer_kwargs['lr'],
                        weight_decay=float(loss_config['weight_decay']))

        best_model = None
        best_acc = 0
        best_score = 1.0

        LOGGER.info(f"Starting training for {self.epochs} epochs!")

        forward_and_loss_fn = forward_and_loss

        if self.use_scheduler:
            batches_per_epoch = len(train_loader) * 2  # every 2nd epoch
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optim,
                T_0=batches_per_epoch,
                T_mult=1,
                eta_min=5e-6,
                # verbose=True,
            )
            
        use_cuda = self.device != "cpu"

        for epoch in range(self.epochs):
            LOGGER.info(f"Epoch num: {epoch}")

            running_loss = 0
            num_correct = 0.0
            num_total = 0.0
            model.train()
            y_train_true = torch.tensor([]).cuda().to(dtype=torch.float64)
            y_train_pred = torch.tensor([]).cuda().to(dtype=torch.float64)
            

            for i, (batch_x, _, batch_y) in enumerate(train_loader):
                if i % 50 == 0:
                    lr = scheduler.get_last_lr()[0] if self.use_scheduler else self.optimizer_kwargs["lr"]

                batch_size = batch_x.size(0)
                num_total += batch_size
                batch_x = batch_x.to(self.device)
                batch_y = torch.stack(batch_y, dim=1).type(torch.float32).to(self.device)

                batch_out, batch_loss = forward_and_loss_fn(model, criterion, batch_x, batch_y, use_cuda=use_cuda, loss_config=loss_config)
                batch_pred = (batch_out + .5).int()
                num_correct += torch.all(batch_pred == batch_y, dim=1).sum().item()

                # y_train_pred = torch.cat((y_train_pred, batch_out), 0)
                # y_train_true = torch.cat((y_train_true, batch_y), 0)
                running_loss += (batch_loss.item() * batch_size)

                if i % 100 == 0:
                    train_loss = running_loss / num_total
                    train_acc = num_correct/num_total*100
                    # train_auc, train_brier, train_ece, train_combined = get_metric(y_train_true, y_train_pred)

                    LOGGER.info(f"[{epoch:04d}][{i:05d}]: {train_loss} {train_acc}")
                    # LOGGER.info(f"[{epoch:04d}][{i:05d}]: {train_loss} {train_acc} | {train_combined} (AUC: {train_auc} BRI: {train_brier} ECE: {train_ece})")
                    
                if self.wandb:
                    wandb.log({"step_loss": train_loss, "step_acc": train_acc})
                                # "step_auc": train_auc, "step_brier": train_brier, "step_ece": train_ece, "step_combined":train_combined})

                if loss_config.get('sam', False):
                    batch_loss.backward()
                    optim.first_step(zero_grad=True)
                    output = model(batch_x)
                    criterion(torch.sigmoid(output), batch_y).backward()
                    optim.second_step(zero_grad=True)
                else:
                    optim.zero_grad()
                    batch_loss.backward()
                    optim.step()

                if self.use_scheduler:
                    scheduler.step()

            running_loss /= num_total
            train_accuracy = (num_correct / num_total) * 100

            if self.wandb:
                # train_auc, train_brier, train_ece, train_combined = get_metric(y_train_true, y_train_pred)
                wandb.log({"epoch": epoch, "train_loss": running_loss, "train_acc": train_accuracy})
                        #    "train_auc": train_auc, "train_brier": train_brier, "train_ece": train_ece, "train_combined": train_combined})
                
            LOGGER.info(f"Epoch [{epoch+1}/{self.epochs}]: train/loss: {running_loss}, train/accuracy: {train_accuracy}")

            test_running_loss = 0.0
            num_correct = 0.0
            num_total = 0.0
            y_test_true = torch.tensor([]).cuda().to(dtype=torch.float64)
            y_test_pred = torch.tensor([]).cuda().to(dtype=torch.float64)
            model.eval()
            
            for batch_x, _, batch_y in test_loader:
                batch_size = batch_x.size(0)
                num_total += batch_size
                batch_x = batch_x.to(self.device)

                with torch.no_grad():
                    output = model(batch_x)
                    batch_out = torch.sigmoid(output)
                batch_y = torch.stack(batch_y, dim=1).type(torch.float32).to(self.device)

                fake_weight = loss_config["fake_weight"]
                loss_name = loss_config["name"]

                if fake_weight == 0.5:
                    batch_loss = criterion(batch_out, batch_y)
                else:
                    fake_loss = criterion(batch_out[:,0], batch_y[:,0])
                    real_loss = criterion(batch_out[:,1], batch_y[:,1])

                    real_weight = 1 - fake_weight if fake_weight < 1 else 1
                    batch_loss = fake_weight * fake_loss + real_weight * real_loss
                test_running_loss += (batch_loss.item() * batch_size)
                
                batch_pred_label = (batch_out + .5).int()
                num_correct += torch.all(batch_pred_label == batch_y, dim=1).sum().item()
                
                y_test_true = torch.cat((y_test_true, batch_y), 0)
                y_test_pred = torch.cat((y_test_pred, batch_out), 0)
            
            test_auc, test_brier, test_ece, test_combined = get_metric(y_test_true, y_test_pred)

            if num_total == 0:
                num_total = 1

            test_running_loss /= num_total
            test_acc = 100 * (num_correct / num_total)
            
            LOGGER.info(
                f"Epoch [{epoch+1}/{self.epochs}]:\n"
                f"    test/loss: {test_running_loss}, test/accuracy: {test_acc},\n"
                f"    test/accuracy: {test_acc}, "
            )
            
            if self.wandb:
                wandb.log({"epoch": epoch, "test_loss": test_running_loss, "test_acc": test_acc,
                           "test_auc": test_auc, "test_brier": test_brier, "test_ece": test_ece, "test_combined": test_combined})
                
            LOGGER.info(
                f"Epoch [{epoch:04d}]:\n"
                f"      loss: {running_loss}, train acc: {train_accuracy}, test_acc: {test_acc},\n"
                # f"      train_combined: {train_combined} (AUC: {train_auc} BRI: {train_brier} ECE: {train_ece}),"
                f"      test_combined: {test_combined} (AUC: {test_auc} BRI: {test_brier} ECE: {test_ece})"
            )

            if best_model is None or best_score > test_combined:
                best_score = test_combined
                best_model = deepcopy(model.state_dict())
                train_acc_fm = str(train_acc).replace('.', '')[:4].zfill(4)
                # train_acc_fm = str(train_combined).replace('.', '')[:4].zfill(4)
                test_acc_fm  = str(test_combined).replace('.', '')[:4].zfill(4)
                ckpt_name = f"{epoch:04d}_tr{train_acc_fm}_ts{test_acc_fm}"

            # if best_model is None or test_acc > best_acc:
            #     best_acc = test_acc
            #     best_model = deepcopy(model.state_dict())
            #     train_acc_fm = str(train_accuracy).replace('.', '')[:4].zfill(4)
            #     test_acc_fm  = str(test_acc).replace('.', '')[:4].zfill(4)
            #     ckpt_name = f"{epoch:04d}_tr{train_acc_fm}_ts{test_acc_fm}"

            if model_dir is not None:
                save_model(
                    model=model,
                    model_dir=model_dir,
                    epoch=epoch,
                    ckpt_name=ckpt_name,
                    name=save_model_name,
                )
        
            if self.wandb:
                wandb.log({"best_test_score": best_score})

            train_loader.dataset.resample()
            patience.append(train_acc)
            if len(patience) > 3:
                if patience[-1] > train_acc and patience[-2] > train_acc and patience[-3] > train_acc:
                    model.load_state_dict(best_model)
                    return model
        model.load_state_dict(best_model)
        return model
