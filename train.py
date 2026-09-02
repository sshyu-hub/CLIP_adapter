"""
CLIP-Large + MoA (Mixture of Adapters) 简洁训练。

在 CLIP ViT 第 15/21/23 层后插入 MoA（N 专家 + CLS 门控），
CLS token mean pool → MLP 分类头，单 CE loss。

MoA:  x → gate(CLS) → Σ gate[i]×Expert_i(x) → +x
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

import config as cfg

DATA_ROOT = cfg.DATA_ROOT
OPENFACE_DIR = cfg.OPENFACE_DIR
CLIP_PATH = cfg.CLIP_MODEL_PATH
ADAPTER_LAYERS = [13, 21, 23]
NUM_EXPERTS = 3
BOTTLENECK = 128
EXPERT_DIMS = [192, 128, 96]   # 专家初始化/结构差异化：不同 bottleneck
LAMBDA_BALANCE = 0.1           # 负载均衡损失权重
NUM_FRAMES, IMAGE_SIZE = cfg.NUM_FRAMES, cfg.IMAGE_SIZE
NUM_CLASSES = cfg.NUM_CLASSES
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3
WD = 1e-4
EMOS = cfg.EMOTIONS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}, MoA 层: {ADAPTER_LAYERS}, 专家数: {NUM_EXPERTS}")

import pandas as pd

# ── 数据加载 ───────────────────────────────────────────────────────
def load_data():
    train_csv = os.path.join(DATA_ROOT, "track1_train_disdim.csv")
    test_csv = os.path.join(DATA_ROOT, "track1_test_dis.csv")
    train_df = pd.read_csv(train_csv)[["name","discrete"]].dropna(subset=["discrete"])
    test_df = pd.read_csv(test_csv)[["name","discrete"]].dropna(subset=["discrete"])
    valid = lambda n: os.path.exists(os.path.join(OPENFACE_DIR, f"{n}.npy"))
    tr_names = [n for n in train_df["name"] if valid(n)]
    te_names = [n for n in test_df["name"] if valid(n)]
    tr_labels = {n: cfg.EMOTION_TO_IDX[train_df[train_df["name"]==n]["discrete"].values[0]] for n in tr_names}
    te_labels = {n: cfg.EMOTION_TO_IDX[test_df[test_df["name"]==n]["discrete"].values[0]] for n in te_names}
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
        m = torch.tensor(cfg.CLIP_MEAN).view(1,3,1,1)
        s = torch.tensor(cfg.CLIP_STD).view(1,3,1,1)
        frames = (frames - m) / s
        return frames, self.labels.get(name, -1)

# ── MoA 组件 ───────────────────────────────────────────────────────
class ExpertMLP(nn.Module):
    def __init__(self, dim=1024, hidden=BOTTLENECK):
        super().__init__()
        self.down = nn.Linear(dim, hidden)
        self.up = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        nn.init.normal_(self.down.weight, std=0.02); nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, std=0.02); nn.init.zeros_(self.up.bias)
    def forward(self, x): return self.up(self.act(self.down(x)))

class Gating(nn.Module):
    """CLS token 门控 → 每样本专家权重。"""
    def __init__(self, dim=1024, n_experts=NUM_EXPERTS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.GELU(),
            nn.Linear(dim // 4, n_experts),
        )
    def forward(self, cls_token): return F.softmax(self.net(cls_token), dim=-1)  # (B, E)

class MoA(nn.Module):
    def __init__(self, dim=1024, expert_dims=EXPERT_DIMS, n_experts=NUM_EXPERTS):
        super().__init__()
        # 专家初始化/结构差异化：每个专家不同 bottleneck
        self.experts = nn.ModuleList([ExpertMLP(dim, expert_dims[i]) for i in range(n_experts)])
        self.gate = Gating(dim, n_experts)
        self.last_gates = None       # 最近一次 forward 的 gates（供负载均衡）
    def forward(self, x):
        gates = self.gate(x[:, 0, :])                              # (B, E)
        self.last_gates = gates                                    # 保持梯度，供负载均衡
        outs = torch.stack([e(x) for e in self.experts], dim=-2)   # (B, N, E, D) 独立专家输出 z1/z2/z3
        return (gates[:, None, :, None] * outs).sum(dim=-2)        # (B, N, D) gate 融合 z

# ── MLP 分类头 ─────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim=1024, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, NUM_CLASSES)
        self.relu = nn.ReLU()
    def forward(self, x): return self.fc2(self.relu(self.fc1(x)))

# ── 构建 CLIP + MoA ───────────────────────────────────────────────
clip = CLIPModel.from_pretrained(CLIP_PATH, local_files_only=True).to(device)
vision = clip.vision_model
for p in vision.parameters(): p.requires_grad = False
vision.eval()

moas = nn.ModuleDict({str(l): MoA() for l in ADAPTER_LAYERS}).to(device)
for l in ADAPTER_LAYERS:
    a = moas[str(l)]
    vision.encoder.layers[l].register_forward_hook(lambda m, i, o, a=a: (o[0] + a(o[0]),) + o[1:])

print(f"MoA 可训练参数: {sum(p.numel() for p in moas.parameters())/1e6:.3f}M")

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
    optimizer = torch.optim.AdamW(list(moas.parameters()) + list(mlp.parameters()), lr=LR, weight_decay=WD)

    # 保存目录（与 AAAsaved 结构一致：best + latest + metrics.csv）
    SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved", "model")
    os.makedirs(SAVE_DIR, exist_ok=True)
    csv_file = open(os.path.join(SAVE_DIR, "metrics.csv"), "w")
    csv_file.write("epoch,train_loss,val_acc,val_f1,test_acc,test_f1,lr\n")

    @torch.no_grad()
    def evaluate(loader):
        preds, gts = [], []
        for frames, labels in loader:
            frames = frames.to(device)
            preds.extend(mlp(encode(frames)).argmax(-1).cpu().numpy())
            gts.extend(labels.numpy())
        return accuracy_score(gts, preds), f1_score(gts, preds, average="weighted")

    best_val_f1, best_state = 0.0, None
    for epoch in range(1, EPOCHS+1):
        total_loss = 0
        for frames, labels in tqdm(train_loader, desc=f"E{epoch}", leave=False):
            frames, labels = frames.to(device), labels.to(device)
            cls = encode(frames)
            ce = F.cross_entropy(mlp(cls), labels)

            # 负载均衡：CV² 惩罚 gate 坍缩到单专家，鼓励均匀使用
            all_gates = torch.cat([m.last_gates for m in moas.values()], dim=0)  # (L*B*T, E)
            f = all_gates.mean(dim=0)                                            # (E,) 每专家平均权重
            balance = (f.std() / f.mean().clamp(min=1e-8)).pow(2)                # 均匀=0，坍缩=大

            loss = ce + LAMBDA_BALANCE * balance
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(moas.parameters()) + list(mlp.parameters()), 1.0)
            optimizer.step()
            total_loss += loss.item()

        mlp.eval()
        val_acc, val_f1 = evaluate(val_loader)
        test_acc, test_f1 = evaluate(test_loader)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:2d}  Loss={avg_loss:.4f}  Val F1={val_f1:.4f}  Test F1={test_f1:.4f}")
        csv_file.write(f"{epoch},{avg_loss:.6f},{val_acc:.6f},{val_f1:.6f},{test_acc:.6f},{test_f1:.6f},{LR:.2e}\n")
        csv_file.flush()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu() for k, v in list(moas.state_dict().items()) + list(mlp.state_dict().items())}
            torch.save(best_state, os.path.join(SAVE_DIR, "best_model.pt"))
        # 每 epoch 保存 latest
        latest_state = {k: v.cpu() for k, v in list(moas.state_dict().items()) + list(mlp.state_dict().items())}
        torch.save(latest_state, os.path.join(SAVE_DIR, "latest_model.pt"))

    csv_file.close()

    # 用最佳模型最终测试
    moas.load_state_dict({k: v for k, v in best_state.items() if k in moas.state_dict()}, strict=False)
    mlp.load_state_dict({k: v for k, v in best_state.items() if k in mlp.state_dict()}, strict=False)
    mlp.eval()
    test_acc, test_f1 = evaluate(test_loader)
    print(f"\n{'='*60}")
    print(f"Best Val F1={best_val_f1:.4f}  Test Acc={test_acc:.4f}  Test F1={test_f1:.4f}")
    print(f"{'='*60}")
    print(f"Saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()
