"""
CLIP Layer Search — no training on CLIP. Extract layer features once,
then linear probe each 3-layer combination to find the best fusion.
"""
import os, sys, itertools, random, json, pickle
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from transformers import CLIPModel

DATA_ROOT = "/home/shy/emotion/MER2025_Mini"
CLIP_MODEL_PATH = "/home/shy/emotion/tool/transformers/clip-vit-large-patch14"
FEAT_CACHE = os.path.join(os.path.dirname(__file__), "layer_features.pkl")
NUM_FRAMES, IMAGE_SIZE, NUM_CLASSES = 8, 224, 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EMOTION_TO_IDX, CLIP_MEAN, CLIP_STD

# ── Data loading ─────────────────────────────────────────────────
def build_data():
    train_csv = os.path.join(DATA_ROOT, "track1_train_disdim.csv")
    test_csv = os.path.join(DATA_ROOT, "track1_test_dis.csv")
    train_df = pd.read_csv(train_csv)[["name","discrete"]].dropna(subset=["discrete"])
    test_df = pd.read_csv(test_csv)[["name","discrete"]].dropna(subset=["discrete"])
    valid = lambda n: os.path.exists(os.path.join(DATA_ROOT, "openface_face", f"{n}.npy"))
    train_names = [n for n in train_df["name"] if valid(n)]
    test_names = [n for n in test_df["name"] if valid(n)]
    train_labels = {n: EMOTION_TO_IDX[train_df[train_df["name"]==n]["discrete"].values[0]] for n in train_names}
    test_labels = {n: EMOTION_TO_IDX[test_df[test_df["name"]==n]["discrete"].values[0]] for n in test_names}
    random.seed(42); random.shuffle(train_names)
    split = int(0.9 * len(train_names))
    return train_names[:split], train_names[split:], test_names, train_labels, test_labels

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, names, label_dict):
        self.names = names; self.label_dict = label_dict
        self.openface_dir = os.path.join(DATA_ROOT, "openface_face")
    def __len__(self): return len(self.names)
    def __getitem__(self, idx):
        name = self.names[idx]
        frames = np.load(os.path.join(self.openface_dir, f"{name}.npy"))
        total = frames.shape[0]
        step = max(total / NUM_FRAMES, 1)
        indices = [min(int(i * step), total - 1) for i in range(NUM_FRAMES)]
        frames = frames[indices]
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)
        frames = nn.functional.interpolate(frames, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        mean = torch.tensor(CLIP_MEAN).view(1,3,1,1); std = torch.tensor(CLIP_STD).view(1,3,1,1)
        return (frames - mean) / std, self.label_dict.get(name, -1)

# ── Step 1: Extract all 24-layer CLS tokens once ─────────────────
@torch.no_grad()
def extract_all_features(names):
    ds = SimpleDataset(names, {n:0 for n in names})
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    clip = CLIPModel.from_pretrained(CLIP_MODEL_PATH, local_files_only=True).to(device)
    vision = clip.vision_model; vision.eval(); del clip

    cls_by_layer = {l: [] for l in range(24)}
    hooks = []
    for l in range(24):
        hooks.append(vision.encoder.layers[l].register_forward_hook(
            lambda m,i,o, l=l: cls_by_layer[l].append(o[0][:,0,:].cpu())))

    results = {}
    for batch_idx, (frames, _) in enumerate(tqdm(loader, desc="Extracting 24 layers")):
        for v in cls_by_layer.values(): v.clear()
        for t in range(frames.shape[1]):
            vision(pixel_values=frames[:, t].to(device))
        # Mean pool over T frames: (24, 1024)
        stacked = {l: torch.cat(cls_by_layer[l], dim=0).mean(dim=0) for l in range(24)}
        results[names[batch_idx]] = torch.stack([stacked[l] for l in range(24)], dim=0)
    for h in hooks: h.remove()
    return results

# ── Step 2: Linear probe each 3-layer combo ──────────────────────
def eval_combo(feats_train, labels_train, feats_val, labels_val, feats_test, labels_test, layers):
    # Concat 3 layers
    X_tr = np.concatenate([feats_train[:, l] for l in layers], axis=1)
    X_val = np.concatenate([feats_val[:, l] for l in layers], axis=1)
    X_te = np.concatenate([feats_test[:, l] for l in layers], axis=1)
    # L2 normalize
    X_tr = X_tr / (np.linalg.norm(X_tr, axis=1, keepdims=True) + 1e-8)
    X_val = X_val / (np.linalg.norm(X_val, axis=1, keepdims=True) + 1e-8)
    X_te = X_te / (np.linalg.norm(X_te, axis=1, keepdims=True) + 1e-8)
    # Logistic regression (fast linear probe)
    clf = LogisticRegression(max_iter=500, C=1.0, multi_class="multinomial", n_jobs=-1)
    clf.fit(X_tr, labels_train)
    val_f1 = f1_score(labels_val, clf.predict(X_val), average="weighted")
    test_f1 = f1_score(labels_test, clf.predict(X_te), average="weighted")
    return val_f1, test_f1

# ── Main ──────────────────────────────────────────────────────────
def main():
    tr_names, val_names, test_names, train_labels, test_labels = build_data()
    all_names = tr_names + val_names + test_names
    all_labels = {**train_labels, **test_labels}
    print(f"Train: {len(tr_names)}, Val: {len(val_names)}, Test: {len(test_names)}")

    # Extract or load
    if os.path.exists(FEAT_CACHE):
        print(f"Loading cached features...")
        with open(FEAT_CACHE, "rb") as f:
            features = pickle.load(f)
    else:
        features = extract_all_features(all_names)
        with open(FEAT_CACHE, "wb") as f:
            pickle.dump(features, f)

    # Build numpy arrays: (N, 24, 1024)
    def to_array(names):
        return np.stack([features[n].numpy() for n in names])
    feats_tr = to_array(tr_names); labs_tr = np.array([train_labels[n] for n in tr_names])
    feats_val = to_array(val_names); labs_val = np.array([train_labels[n] for n in val_names])
    feats_te = to_array(test_names); labs_te = np.array([test_labels[n] for n in test_names])

    # Search all 3-layer combos from every 2nd layer
    candidates = list(range(24))  # 0-23, all layers
    combos = list(itertools.combinations(candidates, 3))
    results = []

    for i, layers in enumerate(combos):
        layers = list(layers)
        val_f1, test_f1 = eval_combo(feats_tr, labs_tr, feats_val, labs_val, feats_te, labs_te, layers)
        results.append({"layers": layers, "val_f1": val_f1, "test_f1": test_f1})
        print(f"[{i+1}/{len(combos)}] {layers}  Val={val_f1:.4f}  Test={test_f1:.4f}")

    results.sort(key=lambda r: r["val_f1"], reverse=True)
    print("\n=== Top 10 ===")
    for r in results[:10]:
        print(f"  {r['layers']}  Val={r['val_f1']:.4f}  Test={r['test_f1']:.4f}")

    with open(os.path.join(os.path.dirname(__file__), "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
