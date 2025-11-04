import torch

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy

def main():
    from models.model_lab2 import CustomNet
    from dataset.dataset_preparation import val_loader
    import torch.nn as nn

    model = CustomNet(num_classes=200).cuda()
    criterion = nn.CrossEntropyLoss()

    validate(model, val_loader, criterion)

if __name__ == '__main__':
    main()