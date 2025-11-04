from typing import Tuple

import torch


def validate(model: torch.nn.Module, val_loader, criterion) -> Tuple[float, float]:
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = val_loss / max(1, len(val_loader))
    accuracy = 100.0 * correct / max(1, total)
    return avg_loss, accuracy

