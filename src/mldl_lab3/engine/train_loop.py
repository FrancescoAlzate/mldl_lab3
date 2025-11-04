from typing import Tuple

import torch
from tqdm import tqdm


def train_one_epoch(epoch: int, model: torch.nn.Module, train_loader, criterion, optimizer) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / max(1, len(train_loader))
    train_accuracy = 100.0 * correct / max(1, total)
    return train_loss, train_accuracy

