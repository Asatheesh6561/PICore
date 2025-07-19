import copy
from requests import get
import torch
import wandb
from tqdm import tqdm
import time
from neuraloperator.neuralop.training.adamw import AdamW
from utils import *
from scipy.spatial.distance import cdist
import numpy as np
from weighted_dataset import WeightedDataset
from coreset import CoresetSelection
from pi_losses import PILoss, ICLoss, WeightedSumLoss


class Trainer:
    def __init__(
        self,
        args,
        model,
        train_loader,
        test_loaders,
        optimizer,
        scheduler,
        data_processor,
    ):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.test_loaders = test_loaders
        self.pretrain_loss = WeightedSumLoss(
            [PILoss(self.args.dataset.dataset_name), ICLoss(), nrmse_loss],
            weights=[self.args.eqn_weight, self.args.ic_weight, self.args.nrmse_weight],
        )
        self.train_loss = nrmse_loss
        self.eval_losses = {
            "mse_loss": mse_loss,
            "rmse": rmse_loss,
            "nrmse": nrmse_loss,
        }
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = []
        self.model = self.model.to(self.args.device)
        self.num_samples = int(
            len(self.train_loader.dataset) * self.args.subset_percentage / 100
        )
        self.data_processor = data_processor
        self.coreset_selector = CoresetSelection(
            self.args,
            self.train_loader,
            self.model,
            self.pretrain_loss,
            self.optimizer,
            self.data_processor,
        )

    def coreset_selection(self):
        indices, gamma, coreset_selection_time = (
            self.coreset_selector.coreset_selection()
        )
        subset_dataset = WeightedDataset(self.train_loader.dataset, indices, gamma)
        self.train_loader = torch.utils.data.DataLoader(
            subset_dataset, batch_size=self.args.dataset.train_batch_size
        )
        return indices, coreset_selection_time

    def train(self):
        for epoch in tqdm(range(self.args.epochs), desc="Training epochs"):
            self.model.train()
            train_losses = []
            epoch_start_time = time.time()
            for batch_idx, data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                inputs, targets, gamma = (
                    data[0].to(self.args.device),
                    data[1].to(self.args.device),
                    data[2].to(self.args.device),
                )
                data = self.data_processor.preprocess({"x": inputs, "y": targets})
                inputs, targets = data["x"].to(self.args.device), data["y"].to(
                    self.args.device
                )
                outputs = self.model(inputs)
                loss = self.train_loss(outputs, targets, reduction="none")
                loss = torch.sum(loss * gamma) / torch.sum(gamma)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.args.grad_clip
                )
                self.optimizer.step()
                train_losses.append(loss.item())

            self.scheduler.step()
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time

            epoch_result = {
                "train_loss": sum(train_losses) / len(train_losses),
                "train_time": epoch_time,
            }
            if self.args.wandb:
                wandb.log(
                    epoch_result,
                    step=epoch,
                )
            self.logger.append(epoch_result)

    def test(self):
        test_losses = {}
        with torch.no_grad():
            for i, test_loader in self.test_loaders.items():
                losses = {f"val_{name}": [] for name in self.eval_losses}
                for batch_idx, data in tqdm(enumerate(test_loader), desc=f"Test {i}"):
                    data = self.data_processor.preprocess(
                        {k: v.to(self.args.device) for k, v in data.items()}
                    )
                    inputs, targets = data["x"].to(self.args.device), data["y"].to(
                        self.args.device
                    )
                    outputs = self.model(inputs)
                    output, data = self.data_processor.postprocess(outputs, data)
                    for name, loss_fn in self.eval_losses.items():
                        loss_value = loss_fn(outputs, targets)
                        losses[f"val_{name}"].append(loss_value.item())

                test_losses[f"{i}"] = {
                    name: sum(values) / len(values) for name, values in losses.items()
                }

        epoch_result = {
            **{
                f"{i}_{name}": loss
                for i, losses in test_losses.items()
                for name, loss in losses.items()
            },
        }
        if self.args.wandb:
            wandb.log(
                epoch_result,
                step=self.args.epochs,
            )
        self.logger.append(epoch_result)
