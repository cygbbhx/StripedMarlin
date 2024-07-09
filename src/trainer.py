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
    batch_out = model(batch_x)
    loss_config = kwargs["loss_config"]
    
    fake_weight = loss_config["fake_weight"]
    assert 0 < fake_weight and fake_weight < 1, "weight should be value between 0 and 1"
    loss_name = loss_config["name"]

    if loss_name == "focal":
        alpha_val = loss_config.get("alpha", 0.25)
        reduction = loss_config.get("reduction", "mean")
        fake_loss = criterion(batch_out[:,0], batch_y[:,0], alpha=alpha_val, reduction=reduction)
        real_loss = criterion(batch_out[:,1], batch_y[:,1], alpha=alpha_val, reduction=reduction)
    elif loss_name == "bce":
        fake_loss = criterion(batch_out[:,0], batch_y[:,0])
        real_loss = criterion(batch_out[:,1], batch_y[:,1])
    
    batch_loss = fake_weight * fake_loss + (1 - fake_weight) * real_loss
    batch_out = model(batch_x)
    batch_loss = criterion(batch_out, batch_y)
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
            "focal": torchvision.ops.sigmoid_focal_loss
        }
        
        criterion = loss_map[loss_config["name"]]
        optim = self.optimizer_fn(model.parameters(), **self.optimizer_kwargs)

        best_model = None
        best_acc = 0

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
            
        use_reg = loss_config.get("use_reg", False)

        if use_reg:
            lambda_l1 = 0.01 
            l1_regularization = 0.0
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l1_regularization += torch.sum(torch.abs(param))

            l1_regularization *= lambda_l1
            l1_regularization = l1_regularization.detach()
            
        use_cuda = self.device != "cpu"

        for epoch in range(self.epochs):
            LOGGER.info(f"Epoch num: {epoch}")

            running_loss = 0
            num_correct = 0.0
            y_train_true = torch.tensor([]).cuda().to(dtype=torch.float64)
            y_train_pred = torch.tensor([]).cuda().to(dtype=torch.float64)
            
            model.train()

            for i, (batch_x, _, batch_y) in enumerate(train_loader):
                if i % 50 == 0:
                    lr = scheduler.get_last_lr()[0] if self.use_scheduler else self.optimizer_kwargs["lr"]

                batch_size = batch_x.size(0)
                num_total += batch_size
                batch_y = torch.stack(batch_y, dim=1).type(torch.float32).to(self.device)

                batch_out, batch_loss = forward_and_loss_fn(model, criterion, batch_x, batch_y, use_cuda=use_cuda, loss_config=loss_config)
                batch_pred = (torch.sigmoid(batch_out) + .5).int()
                num_correct += torch.all(batch_pred == batch_y, dim=1).sum().item()

                if use_reg:
                    condition_1 = (batch_y[:, 0] == 1) & (batch_y[:, 1] == 0)
                    condition_2 = (batch_y[:, 0] == 0) & (batch_y[:, 1] == 1)

                    combined_condition = condition_1 | condition_2
                    count = torch.sum(combined_condition).item()
                    ratio = count / batch_size
                    batch_loss += ratio * l1_regularization

                y_train_pred = torch.cat((y_train_pred, torch.sigmoid(batch_out)), 0)
                y_train_true = torch.cat((y_train_true, batch_y), 0)
                running_loss += (batch_loss.item() * batch_size)

                train_loss = running_loss / num_total
                train_acc = num_correct/num_total*100
                train_auc, train_brier, train_ece, train_combined = get_metric(y_train_true, y_train_pred)

                LOGGER.info(f"[{epoch:04d}][{i:05d}]: {train_loss} {train_acc} | {train_combined} (AUC: {train_auc} BRI: {train_brier} ECE: {train_ece})")
                    
                if self.wandb:
                    wandb.log({"step_loss": train_loss, "step_acc": train_acc,
                                "step_auc": train_auc, "step_brier": train_brier, "step_ece": train_ece, "step_combined":train_combined})
                LOGGER.info(
                        f"[{epoch:04d}][{i:05d}]: {running_loss / num_total} {num_correct/num_total*100}")

                optim.zero_grad()
                batch_loss.backward()
                optim.step()
                if self.use_scheduler:
                    scheduler.step()

            if self.wandb:
                train_auc, train_brier, train_ece, train_combined = get_metric(y_train_true, y_train_pred)
                wandb.log({"epoch": epoch, "train_loss": running_loss, "train_acc": train_accuracy,
                           "train_auc": train_auc, "train_brier": train_brier, "train_ece": train_ece, "train_combined": train_combined})
                
            running_loss /= num_total
            train_accuracy = (num_correct / num_total) * 100

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
                    batch_pred = model(batch_x)
                batch_y = torch.stack(batch_y, dim=1).type(torch.float32).to(self.device)

                fake_weight = loss_config["fake_weight"]
                loss_name = loss_config["name"]

                if loss_name == "focal":
                    alpha_val = loss_config.get("alpha", 0.25)
                    reduction = loss_config.get("reduction", "mean")
                    fake_loss = criterion(batch_out[:,0], batch_y[:,0], alpha=alpha_val, reduction=reduction)
                    real_loss = criterion(batch_out[:,1], batch_y[:,1], alpha=alpha_val, reduction=reduction)
                elif loss_name == "bce":
                    fake_loss = criterion(batch_out[:,0], batch_y[:,0])
                    real_loss = criterion(batch_out[:,1], batch_y[:,1])
                
                batch_loss = fake_weight * fake_loss + (1 - fake_weight) * real_loss
                test_running_loss += (batch_loss.item() * batch_size)
                
                batch_pred = torch.sigmoid(batch_pred)
                batch_pred_label = (batch_pred + .5).int()
                num_correct += torch.all(batch_pred_label == batch_y, dim=1).sum().item()
                
                y_test_true = torch.cat((y_test_true, batch_y), 0)
                y_test_pred = torch.cat((y_test_pred, torch.sigmoid(batch_pred)), 0)
            
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
                f"      train_combined: {train_combined} (AUC: {train_auc} BRI: {train_brier} ECE: {train_ece}),"
                f"      test_combined: {test_combined} (AUC: {test_auc} BRI: {test_brier} ECE: {test_ece})"
            )

            if best_model is None or test_acc > best_acc:
                best_acc = test_acc
                best_model = deepcopy(model.state_dict())
                train_acc_fm = str(train_accuracy).replace('.', '')[:4].zfill(4)
                test_acc_fm  = str(test_acc).replace('.', '')[:4].zfill(4)
                ckpt_name = f"{epoch:04d}_tr{train_acc_fm}_ts{test_acc_fm}"

            if model_dir is not None:
                save_model(
                    model=model,
                    epoch=epoch,
                    ckpt_name=ckpt_name,
                    name=save_model_name,
                )
        
            if self.wandb:
                wandb.log({"best_test_acc": best_acc})

        model.load_state_dict(best_model)
        return model
