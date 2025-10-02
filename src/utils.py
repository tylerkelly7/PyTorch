from pathlib import Path
import random
import numpy as np
import torch

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters if p.requires_grad()
    
def save_checkpoint(model: torch.nn.Module, path: str):
	Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    
def load_checkpoint(model: torch.nn.Module, path: str, map_location=None):
	state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    return model

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
	preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()