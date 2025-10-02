# src/train.py
import argparse, yaml
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from data_prep import get_dataloaders
from models import SimpleCNN
from utils import seed_everything, count_parameters, save_checkpoint, accuracy_from_logits
from evaluate import evaluate

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    return ap.parse_args()
    
def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    
    seed_everything(cfg["seed"])
    
    # Data
    train_loader, val_loader = get_dataloaders(
    	data_dir=cfg["data_dir"],
    	batch_size=cfg["batch_size"],
    	num_workers=cfg["num_workers"],
    )
    
    # Device handling
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available()
    
    # Model, loss, optimizer
    model = SimpleCNN(num_classes=cfg["model"]["num_classes"]).to(device)
    print(f"Trainable params: {count_parameters(model):,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    
    best_val_acc = 0.0
    epochs = cfg["train"]["epochs"]
    ckpt_path = cfg["train"]["ckpt_path"]
    
    # Training loop with checkpointing
    for epoch in range(1, epochs + 1):
    	model.train()
    	running_loss, running_acc = 0.0, 0.0
        
    	for images, targets in train_loader:
        	images, targets = images.to(device), targets.to(device)
            
        	optimizer.zero_grad()				# 1) clear old accumulated gradients
            logits = model(images)				# 2) forward pass builds graph
            loss   = criterion(logits, targets)
            loss.backward()						# 3) backprop: compute gradients
            optimizer.step()					# 4) update parameters
            
            running_loss += loss.item(0
            running_acc  += accuracy_from_logits(logits, targets)
        
        train_loss = running_loss / len(train_loader)
        train_acc  = running_acc  / len(train_loader)
        
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"[Epoch {epoch}/{epochs}] "
        	  f"train_loss={train_loss:.4f} acc={train_acc.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
        	best_val_acc = val_acc
            save_checkpoint(model, ckpt_path)
    
    # Print best model accuracy with checkpoint path after looping through epochs    
    print(f"Best val acc: {best_val_acc:.4f} (ckpt: {ckpt_path})")

if __name__ == "__main__":
    main()
	
        
        
        
        
        
        
        
        
        
        
        
        
        
        