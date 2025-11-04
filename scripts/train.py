import os
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from mldl_lab3.utils import (
    load_config,
    configure_logging,
    setup_wandb,
    log_metrics,
    finish as wandb_finish,
    watch_model,
)
from mldl_lab3.models import CustomNet
from mldl_lab3.data import build_dataloaders
from mldl_lab3.engine import train_one_epoch, validate


def main() -> None:
    configure_logging()
    logger = logging.getLogger("train")

    cfg_path = os.environ.get("CFG", os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    cfg = load_config(cfg_path)

    device = torch.device("cuda" if cfg["device"]["use_cuda"] and torch.cuda.is_available() else "cpu")

    use_wandb = cfg.get("wandb", {}).get("enabled", False) and cfg.get("wandb", {}).get("mode", "online") != "disabled"
    if use_wandb:
        setup_wandb(cfg)

    train_loader, val_loader = build_dataloaders(cfg)
    model = CustomNet(num_classes=cfg["model"]["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg["train"]["lr"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    if use_wandb:
        watch_model(model)

    best_acc = 0.0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(epoch, model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        logger.info(f"Epoch {epoch}: train loss={train_loss:.4f} acc={train_acc:.2f}% | val loss={val_loss:.4f} acc={val_acc:.2f}%")
        if use_wandb:
            log_metrics({
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)
        best_acc = max(best_acc, val_acc)

    logger.info(f"Best val acc: {best_acc:.2f}%")
    if use_wandb:
        log_metrics({"best/val_acc": best_acc})
        wandb_finish()


if __name__ == "__main__":
    main()

