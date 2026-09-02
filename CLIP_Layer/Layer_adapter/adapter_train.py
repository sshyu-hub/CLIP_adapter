"""
CLIP-Large + Adapter 基线（在 base_simple.py 基础上加 Adapter）。

结构: CLIP ViT 冻结，在第 15/21/23 层后插入 bottleneck Adapter，
      取 CLS token mean pool → MLP 分类头。
Adapter: x → Linear(1024→128) → GELU → Linear(128→1024) → +x
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
ADAPTER_LAYERS = [13, 21, 23]
NUM_EXPERTS = 3
BOTTLENECK = 128
NUM_FRAMES, IMAGE_SIZE = 16, 224
NUM_CLASSES = 6
BATCH_SIZE = 8
EPOCHS = 9
LR = 1e-3
WD = 1e-4
EMOS = ["neutral", "angry", "happy", "sad", "worried", "surprise"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}, Adapter 层: {ADAPTER_LAYERS}")

import pandas as pd

# ── 数据加载（与 adapter_train.py / base_simple.py 一致）────────
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
        frames = frames[indices]
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)
        frames = F.interpolate(frames, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        m = torch.tensor(CLIP_MEAN).view(1,3,1,1)
        s = torch.tensor(CLIP_STD).view(1,3,1,1)
        frames = (frames - m) / s
        return frames, self.labels.get(name, -1)

# ── Adapter ────────────────────────────────────────────────────────
class Adapter(nn.Module):
    def __init__(self, dim=1024, bottleneck=BOTTLENECK):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.normal_(self.down.weight, std=0.02); nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, std=0.02); nn.init.zeros_(self.up.bias)
    def forward(self, x): return self.up(F.gelu(self.down(x)))

# ── MLP 分类头 ─────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim=1024, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, NUM_CLASSES)
        self.relu = nn.ReLU()
    def forward(self, x): return self.fc2(self.relu(self.fc1(x)))

# ── 构建 CLIP + Adapter ───────────────────────────────────────────
clip = CLIPModel.from_pretrained(CLIP_PATH, local_files_only=True).to(device)
vision = clip.vision_model
for p in vision.parameters(): p.requires_grad = False
vision.eval()

adapters = nn.ModuleDict({str(l): Adapter() for l in ADAPTER_LAYERS}).to(device)
for l in ADAPTER_LAYERS:
    a = adapters[str(l)]
    vision.encoder.layers[l].register_forward_hook(lambda m, i, o, a=a: (o[0] + a(o[0]),) + o[1:])

print(f"Adapter 可训练参数: {sum(p.numel() for p in adapters.parameters())/1e6:.3f}M")

def encode(frames):
    """frames: (B, T, C, H, W) → (B, 1024) CLS mean pool（含 MoA 增强）。"""
    B, T = frames.shape[0], frames.shape[1]
    all_cls = []
    for t in range(T):
        out = vision(pixel_values=frames[:, t])
        all_cls.append(out.last_hidden_state[:, 0, :])
    return torch.stack(all_cls, dim=1).mean(dim=1)

# ── 主函数 ─────────────────────────────────────────────────────────
def main():
    tr_names, tr_labels, te_names, te_labels = load_data()
    print(f"训练样本: {len(tr_names)}, 测试样本: {len(te_names)}")

    random.seed(42); random.shuffle(tr_names)
    split = int(0.9 * len(tr_names))
    tr_train, tr_val = tr_names[:split], tr_names[split:]

    train_loader = DataLoader(FDataset(tr_train, tr_labels), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(FDataset(tr_val, tr_labels), batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(FDataset(te_names, te_labels), batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=True)

    mlp = MLP().to(device)
    optimizer = torch.optim.AdamW(list(adapters.parameters()) + list(mlp.parameters()), lr=LR, weight_decay=WD)

    # metrics.csv 记录（与 base/metrics.csv 格式统一）
    csv_file = open(os.path.join(os.path.dirname(__file__), "metrics.csv"), "w")
    csv_file.write("epoch,train_loss,val_acc,val_f1,lr\n")

    best_val_f1, best_state = 0.0, None
    for epoch in range(1, EPOCHS+1):
        total_loss = 0
        for frames, labels in tqdm(train_loader, desc=f"E{epoch}", leave=False):
            frames, labels = frames.to(device), labels.to(device)
            cls = encode(frames)
            loss = F.cross_entropy(mlp(cls), labels)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(adapters.parameters()) + list(mlp.parameters()), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # 训练期间只看验证集
        preds, gts = [], []
        with torch.no_grad():
            for frames, labels in val_loader:
                preds.extend(mlp(encode(frames.to(device))).argmax(-1).cpu().numpy())
                gts.extend(labels.numpy())
        val_acc = accuracy_score(gts, preds)
        val_f1 = f1_score(gts, preds, average="weighted")
        avg_loss = total_loss / len(train_loader)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu() for k, v in list(adapters.state_dict().items()) + list(mlp.state_dict().items())}
        print(f"Epoch {epoch:2d}  Loss={avg_loss:.4f}  Val F1={val_f1:.4f}")
        csv_file.write(f"{epoch},{avg_loss:.6f},{val_acc:.6f},{val_f1:.6f},{LR:.2e}\n")
        csv_file.flush()

    csv_file.close()

    torch.save(best_state, os.path.join(os.path.dirname(__file__), "best_adapter.pt"))
    adapters.load_state_dict({k: v for k, v in best_state.items() if k in adapters.state_dict()}, strict=False)
    mlp.load_state_dict({k: v for k, v in best_state.items() if k in mlp.state_dict()}, strict=False)
    preds, gts = [], []
    with torch.no_grad():
        for frames, labels in test_loader:
            preds.extend(mlp(encode(frames.to(device))).argmax(-1).cpu().numpy())
            gts.extend(labels.numpy())
    print(f"\n{'='*60}")
    print(f"Test Acc={accuracy_score(gts, preds):.4f}  Test F1={f1_score(gts, preds, average='weighted'):.4f}")
    print(f"{'='*60}")
    print(classification_report(gts, preds, target_names=EMOS, digits=4))

if __name__ == "__main__":
    main()
