"""
CLIP-Large + MLP 基线（无 Adapter）
完全复用第一份 Adapter 代码的数据加载和特征提取方式。
"""
import os, sys, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
from transformers import CLIPModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import EMOTION_TO_IDX, CLIP_MEAN, CLIP_STD

DATA_ROOT = "/home/shy/emotion/MER2025_Mini"
OPENFACE_DIR = os.path.join(DATA_ROOT, "openface_face")
CLIP_PATH = "/home/shy/emotion/tool/transformers/clip-vit-large-patch14"
NUM_FRAMES, IMAGE_SIZE = 16, 224
NUM_CLASSES = 6
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3
WD = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

import pandas as pd

# ── 数据加载（完全复用 Adapter 代码）───────────────────────────
def load_data():
    train_csv = os.path.join(DATA_ROOT, "track1_train_disdim.csv")
    test_csv = os.path.join(DATA_ROOT, "track1_test_dis.csv")
    train_df = pd.read_csv(train_csv)[["name","discrete"]].dropna(subset=["discrete"])
    test_df = pd.read_csv(test_csv)[["name","discrete"]].dropna(subset=["discrete"])
    valid = lambda n: os.path.exists(os.path.join(OPENFACE_DIR, f"{n}.npy"))
    tr_names = [n for n in train_df["name"] if valid(n)]
    te_names = [n for n in test_df["name"] if valid(n)]
    tr_labels = {n: EMOTION_TO_IDX[train_df[train_df["name"]==n]["discrete"].values[0]] for n in tr_names}
    te_labels = {n: EMOTION_TO_IDX[test_df[test_df["name"]==n]["discrete"].values[0]] for n in te_names}
    return tr_names, tr_labels, te_names, te_labels

class FDataset(Dataset):
    def __init__(self, names, labels):
        self.names, self.labels = names, labels
        self.dir = OPENFACE_DIR
    def __len__(self): return len(self.names)
    def __getitem__(self, idx):
        name = self.names[idx]
        frames = np.load(os.path.join(self.dir, f"{name}.npy"))
        total = frames.shape[0]
        step = max(total / NUM_FRAMES, 1)
        indices = [min(int(i * step), total - 1) for i in range(NUM_FRAMES)]
        frames = frames[indices]                           # (T, H, W, C)
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)               # (T, C, H, W)
        frames = F.interpolate(frames, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        m = torch.tensor(CLIP_MEAN).view(1,3,1,1)
        s = torch.tensor(CLIP_STD).view(1,3,1,1)
        frames = (frames - m) / s
        return frames, self.labels.get(name, -1)

# ── 构建 CLIP 视觉编码器（冻结）───────────────────────────────
clip = CLIPModel.from_pretrained(CLIP_PATH, local_files_only=True).to(device)
vision = clip.vision_model
for p in vision.parameters():
    p.requires_grad = False
vision.eval()

# ── MLP 分类头 ──────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim=1024, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, NUM_CLASSES)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# ── 主函数 ──────────────────────────────────────────────────
def main():
    tr_names, tr_labels, te_names, te_labels = load_data()
    print(f"训练样本: {len(tr_names)}, 测试样本: {len(te_names)}")

    # 固定划分 90% 训练，10% 验证
    random.seed(42)
    random.shuffle(tr_names)
    split = int(0.9 * len(tr_names))
    tr_names_train = tr_names[:split]
    tr_names_val = tr_names[split:]

    train_ds = FDataset(tr_names_train, tr_labels)
    val_ds = FDataset(tr_names_val, tr_labels)
    test_ds = FDataset(te_names, te_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=True)

    model = MLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    best_val_f1 = 0.0
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0
        for frames, labels in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            frames, labels = frames.to(device), labels.to(device)
            B, T = frames.shape[0], frames.shape[1]
            # 提取特征：逐帧过 vision_model，取 CLS token
            all_cls = []
            with torch.no_grad():
                for t in range(T):
                    out = vision(pixel_values=frames[:, t])
                    all_cls.append(out.last_hidden_state[:, 0, :])  # (B, 1024)
            cls = torch.stack(all_cls, dim=1).mean(dim=1)           # (B, 1024)
            logits = model(cls)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(device)
                B, T = frames.shape[0], frames.shape[1]
                all_cls = []
                for t in range(T):
                    out = vision(pixel_values=frames[:, t])
                    all_cls.append(out.last_hidden_state[:, 0, :])
                cls = torch.stack(all_cls, dim=1).mean(dim=1)
                logits = model(cls)
                preds.extend(logits.argmax(-1).cpu().numpy())
                gts.extend(labels.numpy())
        val_f1 = f1_score(gts, preds, average="weighted")
        print(f"Epoch {epoch:2d}  Loss={total_loss/len(train_loader):.4f}  Val F1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "best_mlp.pt"))

    # 加载最佳模型测试
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "best_mlp.pt")))
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for frames, labels in test_loader:
            frames = frames.to(device)
            B, T = frames.shape[0], frames.shape[1]
            all_cls = []
            for t in range(T):
                out = vision(pixel_values=frames[:, t])
                all_cls.append(out.last_hidden_state[:, 0, :])
            cls = torch.stack(all_cls, dim=1).mean(dim=1)
            logits = model(cls)
            preds.extend(logits.argmax(-1).cpu().numpy())
            gts.extend(labels.numpy())
    test_acc = accuracy_score(gts, preds)
    test_f1 = f1_score(gts, preds, average="weighted")
    print(f"\n{'='*60}")
    print(f"Test Acc = {test_acc:.4f}   Test F1 = {test_f1:.4f}")
    print(f"{'='*60}")
    print(classification_report(gts, preds, target_names=["neutral","angry","happy","sad","worried","surprise"]))

if __name__ == "__main__":
    main()