import hydra
from omegaconf import DictConfig, OmegaConf
from train import Trainer
from utils import *
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import wandb
import logging
import os
import time
import pickle
import random
import numpy as np


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  # Debug print
    OmegaConf.set_struct(cfg, False)

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    name = f"{cfg.dataset.dataset_name}_{cfg.model.model_name}{cfg.dataset.dim}d_{cfg.dataset.train_resolution}_{cfg.subset_percentage}_{cfg.coreset_algorithm}_{cfg.seed}"
    cfg.pretrain_model_save_path = os.path.join(
        cfg.log_dir,
        "models",
        f"{cfg.dataset.dataset_name}_{cfg.model.model_name}{cfg.dataset.dim}d_{cfg.seed}.pth",
    )

    if cfg.attack != "":
        name += f"_{cfg.attack}"
        cfg.pretrain_model_save_path = cfg.pretrain_model_save_path.replace(
            ".pth", f"_{cfg.attack}.pth"
        )

    os.makedirs(os.path.join(cfg.log_dir, "models"), exist_ok=True)

    result_path = os.path.join(cfg.log_dir, f"{name}.pkl")
    if os.path.exists(result_path) and not cfg.overwrite:
        print(f"File {result_path} already exists")
        return

    if cfg.wandb:
        wandb.init(
            project=cfg.wandb_project,
            config=OmegaConf.to_container(cfg, resolve=True),
            entity=cfg.wandb_entity,
            dir=cfg.wandb_dir,
            name=name,
        )

    train_loader, test_loaders, data_processor = get_dataset(cfg)
    model = get_model(cfg)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    trainer = Trainer(
        cfg, model, train_loader, test_loaders, optimizer, scheduler, data_processor
    )

    _, coreset_time = trainer.coreset_selection()
    trainer.model = get_model(cfg).to(cfg.device)
    trainer.optimizer = AdamW(
        trainer.model.parameters(), lr=float(cfg.lr), weight_decay=1e-4
    )
    trainer.scheduler = CosineAnnealingLR(trainer.optimizer, T_max=cfg.epochs)
    trainer.train()
    trainer.test()

    with open(result_path, "wb") as f:
        pickle.dump(
            (trainer.logger, coreset_time, OmegaConf.to_container(cfg, resolve=True)), f
        )


if __name__ == "__main__":
    main()
