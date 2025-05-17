import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import datasets, transforms
from tqdm import tqdm

from resnet import ResNet50

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def get_data_loaders(batch_size=128, num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


# Training function
def train(model, trainloader, criterion, optimizer, scheduler, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    loop = tqdm(trainloader, desc=f"Epoch {epoch + 1}")
    for batch_idx, (inputs, targets) in enumerate(loop):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        loop.set_postfix(loss=train_loss / (batch_idx + 1), acc=100. * correct / total)

    scheduler.step()

    return train_loss / len(trainloader), 100. * correct / total


# Testing function
def val(model, testloader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return test_loss / len(testloader), 100.0 * correct / total


def main():
    num_epochs = 100
    batch_size = 128
    lr = 0.05
    T_0 = 10
    T_mult = 2

    os.makedirs('results', exist_ok=True)

    trainloader, testloader, classes = get_data_loaders(batch_size)

    model = ResNet50().to(device)
    print(f"Created ResNet50 model with {sum(p.numel() for p in model.parameters())} parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # warm restarts
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=1e-5)

    start_epoch = 0
    if os.path.exists('results/resnet50_checkpoint.pt'):
        checkpoint = torch.load('results/resnet50_checkpoint.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    best_acc = 0

    for epoch in range(start_epoch, num_epochs):
        # Train
        train_loss, train_acc = train(model, trainloader, criterion, optimizer, scheduler, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Test
        test_loss, test_acc = val(model, testloader, criterion)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f'\nEpoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_acc': train_acc,
            'test_acc': test_acc
        }, 'results/resnet50_checkpoint.pt')

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'results/resnet50_best.pt')
            print('\nBest model saved!')

        if epoch % 10 == 0:
            lrs = scheduler.get_last_lr()
            print(f'Current LR: {lrs[0]:.6f}')

        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(test_losses, label='Test Loss')
            plt.title('Loss vs. Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(train_accs, label='Train Accuracy')
            plt.plot(test_accs, label='Test Accuracy')
            plt.title('Accuracy vs. Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy (%)')
            plt.legend()

            plt.tight_layout()
            plt.savefig(f'results/training_curves_epoch_{epoch + 1}.png')
            plt.close()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/final_training_curves.png')
    plt.show()

    print(f'Best accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()