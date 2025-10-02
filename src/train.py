# configs/config.yaml
seed: 42
data_dir: "data"
batch_size: 64
num_workers: 0

model:
	num_classes: 10
    
train:
	epochs: 5
    lr: 0.001
    weight_decay: 0.0
    device: "cuda" # or "cpu" (auto-detected in code)
    ckpt_path: "checkpoints/best_model.pt"