# PyTorch Demo

A clean and testable PyTorch project that trains a small CNN on MNIST, using config-driven experiments, checkpoints and simple evaluation.

## Features
- `nn.Module` model (`SimpleCNN`)
- Config-driven training (`configs/config.yaml`)
- Checkpointing and evaluation reports
- Unit tests with `pytest`

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate # (Windows: .venv\Stripts\activate)
pip install -r requirements.txt
pip install -e .

python -m src.train --config configs/config.yaml
pytest -q
ls outputs/
```

## Files
- src/model.py - CNN definition
- src/dataset.py - MNIST loaders
- src/train.py - training loop with best checkpoint saving
- src/evaluate.py - evaluation + confusion matrix
- ssrc/utils.py - seeding, checkpoint helpers, metrics
- tests/ unit tests for forward pass and one training step

## Results
- Expect ~99% validation accuracy after a few epochs
- Artifacts: outpus/confusion_matrix.png, outputs/classification_report.txt
- Best model saved to checkpoints/best_model.pt.

## Next Steps
- Add TensorBoard/Weights and Biases logging
- Add early stopping and LR scheduling
- Try CIFAR-10 with a deeper CNN or ResNet
- Export to TorchScript / ONNX for deployment
- Add Docker + GitHub Actions (CI)