# src/evaluate.py
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from pathlib import Path
from utils import accuracy_from_logits

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    all_targets, all_preds = [], []
    
    with torch.no_grad():
    	for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            
            total_loss += loss.item()
            total_acc += accuracy_from_logits(logits, targets)
            n += 1
            
            all_targets.extend(targets.cpu().tolist())
            all_preds.extend(logits.argmax(1).cpu().tolist())
            
    avg_loss = total_loss / n
    avg_acc = total_acc / n
        
    # Confusion matrix figure
    cm = confusion_matrix(all_targets, all_preds)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    Path("outputs").mkdir(parents=True, exist_ok=True)
    fig.savefig("outputs/confusion_matrix.png")
    plt.close(fig)
        
    # Report
    report = classification_report(all_targets, all_preds, digits=4)
    with open("outputs/classification_report.txt", "w") as f:
    	f.write(report)
            
    return avg_loss, avg_acc
        