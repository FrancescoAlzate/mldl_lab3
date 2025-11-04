import os
from typing import Tuple

import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder


def build_dataloaders(config) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    data_root = config["data"]["root"]
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    transform = T.Compose([
        T.Resize((config["data"]["resize_h"], config["data"]["resize_w"])),
        T.ToTensor(),
        T.Normalize(mean=config["data"]["norm_mean"], std=config["data"]["norm_std"]),
    ])

    train_ds = ImageFolder(root=train_dir, transform=transform)
    val_ds = ImageFolder(root=val_dir, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=config["eval"]["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader

