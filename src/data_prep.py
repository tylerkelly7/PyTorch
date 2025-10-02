from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloaders(
	data_dir: str = "data",
	batch_size: int = 64,
	num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
	"""
    Returns (train_loader, val_loader) for MNIST.
    Normalization values are standard for MNIST: mean=0.1307, std=0.3081.
    """
    transform = transforms.Compose([
    	transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    val = datasets.MNIST(rood=data_dir, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader