import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATA_DIR = "./data"


# ==== load_data.py ====

def load_data(batch_size):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Випадкове горизонтальне віддзеркалення
        # Випадкове обрізання з додаванням padding
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        # Нормалізація до [-1, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            # Без додаткових трансформацій для тесту
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
