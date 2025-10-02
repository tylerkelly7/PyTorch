# tests/test_model.py
import torch
from src.model import SimpleCNN

def test_forward_pass_shape():
    model = SimpleCNN(num_classes=10)
    x = torch.randn(2, 1, 28, 28)
    y = model(x)
    assert y.shape == (2, 10)