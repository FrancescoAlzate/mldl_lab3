import os
import logging

import torch
import torch.nn as nn

from mldl_lab3.utils import (
    load_config,
    configure_logging,
    setup_wandb,
    log_metrics,
    finish as wandb_finish,
)
from mldl_lab3.models import CustomNet
from mldl_lab3.data import build_dataloaders
from mldl_lab3.engine import validate


def main() -> None:
    configure_logging()
    logger = logging.getLogger("eval")

    cfg_path = os.environ.get("CFG", os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    cfg = load_config(cfg_path)

    device = torch.device("cuda" if cfg["device"]["use_cuda"] and torch.cuda.is_available() else "cpu")

    use_wandb = cfg.get("wandb", {}).get("enabled", False) and cfg.get("wandb", {}).get("mode", "online") != "disabled"
    if use_wandb:
        setup_wandb(cfg)

    _, val_loader = build_dataloaders(cfg)
    model = CustomNet(num_classes=cfg["model"]["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()

    val_loss, val_acc = validate(model, val_loader, criterion)
    logger.info(f"Validation: loss={val_loss:.4f} acc={val_acc:.2f}%")
    if use_wandb:
        log_metrics({
            "val/loss": val_loss,
            "val/acc": val_acc,
        })
        wandb_finish()


if __name__ == "__main__":
    main()

