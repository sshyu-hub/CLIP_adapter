"""Test best model with TTA (horizontal flip)."""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

import config as cfg
from dataset import OpenFaceDataset, build_label_dicts
from model import CLIPMoEEmotionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

_, test_label_dict, _, test_names, num_classes = build_label_dicts(cfg.DATA_ROOT)
test_ds = OpenFaceDataset(test_names, test_label_dict, openface_dir=cfg.OPENFACE_DIR,
                          num_frames=cfg.NUM_FRAMES, image_size=cfg.IMAGE_SIZE,
                          mean=cfg.CLIP_MEAN, std=cfg.CLIP_STD)
test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

ckpt_path = os.path.join(cfg.MODEL_DIR, "best_model.pt")
print(f"Loading {ckpt_path}")
model = CLIPMoEEmotionModel(
    clip_model_path=cfg.CLIP_MODEL_PATH, adapter_layers=cfg.ADAPTER_LAYERS,
    num_classes=num_classes, num_experts=cfg.NUM_EXPERTS,
    adapter_bottleneck=cfg.ADAPTER_BOTTLENECK,
    expert_dims=cfg.EXPERT_DIMS, hf_bottleneck=cfg.HF_BOTTLENECK,
).to(device)
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()

all_preds, all_labels = [], []
for batch in tqdm(test_loader, desc="Testing"):
    frames = batch["frames"].to(device)
    labels = batch["label"].to(device)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        logits_orig = model(frames)["logits"]
        logits_flip = model(torch.flip(frames, dims=[-1]))["logits"]
        logits = (logits_orig + logits_flip) / 2
    all_preds.extend(logits.argmax(-1).cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="weighted")
print(f"\nTest Results (TTA):")
print(f"  Accuracy: {acc:.4f}")
print(f"  F1 (weighted): {f1:.4f}")
print(classification_report(all_labels, all_preds, target_names=cfg.EMOTIONS, digits=4))
