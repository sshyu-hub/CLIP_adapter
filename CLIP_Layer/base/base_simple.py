"""
CLIP-Large + MLP 基线（无 Adapter，多种子取平均）。
完全复用第一份 Adapter 代码的数据加载和特征提取方式。

与旧版 base_simple.py 的区别：
  - 多种子重复训练（默认 6），每个 seed 固定随机性（初始化 + batch 顺序）
  - 每个 seed 得到 test acc/f1，最终输出 mean ± std
  - 结果写入 metrics.csv（seed,test_acc,test_f1，末尾加 mean/std 汇总行）

用法:
  python base_simple.py             # 默认 6 个 seed
  python base_simple.py --seeds 10
"""
import os, sys, random, argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
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
N_SEEDS = 6
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

# ── 构建 CLIP 视觉编码器（冻结，全局共享）───────────────────
clip = CLIPModel.from_pretrained(CLIP_PATH, local_files_only=True).to(device)
vision = clip.vision_model
for p in vision.parameters():
    p.requires_grad = False
vision.eval()

def encode(frames):
    """frames: (B, T, C, H, W) → (B, 1024) CLS mean pool。"""
    B, T = frames.shape[0], frames.shape[1]
    all_cls = []
    with torch.no_grad():
        for t in range(T):
            out = vision(pixel_values=frames[:, t])
            all_cls.append(out.last_hidden_state[:, 0, :])  # (B, 1024)
    return torch.stack(all_cls, dim=1).mean(dim=1)           # (B, 1024)

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

# ── 随机性控制 ──────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ── 单个 seed 的完整训练 + 测试 ─────────────────────────────
def run_one_seed(seed, tr_names_train, tr_names_val, tr_labels, te_names, te_labels):
    set_seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(FDataset(tr_names_train, tr_labels), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
                              worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(FDataset(tr_names_val, tr_labels), batch_size=BATCH_SIZE*2,
                            shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(FDataset(te_names, te_labels), batch_size=BATCH_SIZE*2,
                             shuffle=False, num_workers=2, pin_memory=True)

    model = MLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    best_val_f1, best_state = 0.0, None
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0
        for frames, labels in tqdm(train_loader, desc=f"seed{seed} E{epoch}", leave=False):
            frames, labels = frames.to(device), labels.to(device)
            cls = encode(frames)
            loss = F.cross_entropy(model(cls), labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for frames, labels in val_loader:
                preds.extend(model(encode(frames.to(device))).argmax(-1).cpu().numpy())
                gts.extend(labels.numpy())
        val_f1 = f1_score(gts, preds, average="weighted")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # 用最佳模型测试
    model.load_state_dict(best_state)
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for frames, labels in test_loader:
            preds.extend(model(encode(frames.to(device))).argmax(-1).cpu().numpy())
            gts.extend(labels.numpy())
    test_acc = accuracy_score(gts, preds)
    test_f1 = f1_score(gts, preds, average="weighted")
    torch.save(best_state, os.path.join(os.path.dirname(__file__), f"best_mlp_s{seed}.pt"))
    return test_acc, test_f1

# ── 主函数 ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()
    seeds = range(args.seeds)

    tr_names, tr_labels, te_names, te_labels = load_data()
    print(f"训练样本: {len(tr_names)}, 测试样本: {len(te_names)}")

    # 固定划分 90% 训练，10% 验证（与 seed 无关）
    random.seed(42)
    random.shuffle(tr_names)
    split = int(0.9 * len(tr_names))
    tr_names_train = tr_names[:split]
    tr_names_val = tr_names[split:]

    rows = []
    for seed in seeds:
        acc, f1 = run_one_seed(seed, tr_names_train, tr_names_val, tr_labels, te_names, te_labels)
        rows.append((seed, acc, f1))
        print(f"seed {seed}: Test Acc = {acc:.4f}   Test F1 = {f1:.4f}")

    accs = [r[1] for r in rows]
    f1s = [r[2] for r in rows]
    mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))
    mean_f1, std_f1 = float(np.mean(f1s)), float(np.std(f1s))

    # 写 metrics.csv
    csv_path = os.path.join(os.path.dirname(__file__), "metrics.csv")
    with open(csv_path, "w") as f:
        f.write("seed,test_acc,test_f1\n")
        for seed, acc, f1 in rows:
            f.write(f"{seed},{acc:.6f},{f1:.6f}\n")
        f.write(f"mean,{mean_acc:.6f},{mean_f1:.6f}\n")
        f.write(f"std,{std_acc:.6f},{std_f1:.6f}\n")

    print(f"\n{'='*60}")
    print(f"Test Acc = {mean_acc:.4f}±{std_acc:.4f}   Test F1 = {mean_f1:.4f}±{std_f1:.4f}  (n={len(rows)})")
    print(f"结果已写入 {csv_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
