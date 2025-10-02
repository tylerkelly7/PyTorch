# tests/test_training.py
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import SimpleCNN

def test_single_training_step():
    model = SimpleCNN(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    
    logits = model(x)
    loss = criterion(logits, y)
    assert torch.isfinite(loss).item()
    
    optimizer.zero_grad()
    loss.backward()
    
    # Ensure at least one parameter has a grad
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    
    optimizer.step()