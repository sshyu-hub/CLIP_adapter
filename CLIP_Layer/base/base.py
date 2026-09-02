"""
单模态 CLIP-Large 视觉基线 — 完整复刻 MER2025 论文表 2 流程。

复刻要点（论文 2.4 节）:
  - 五折交叉验证 (StratifiedKFold, Train&Val 集)
  - 每折随机采样超参 50 次，短训练用验证集评估，锁定最优
  - 锁定后换 6 个随机种子全量训练，报告 Test Acc/F1 的均值 ± 标准差

分类器: MLP (Linear→ReLU→Linear)，即论文的 fully connected layers。
特征: CLIP-Large image embedding (512 维), 16 帧平均。

用法:
  python base.py            # 首次提取特征, 然后三阶段
  python base.py --reload   # 强制重新提取特征
  python base.py --quick    # 快速模式(少搜索少重复)验证可跑通
"""
import os, sys, random, json, argparse, pickle
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from transformers import CLIPModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import EMOTION_TO_IDX, CLIP_MEAN, CLIP_STD

DATA_ROOT = "/home/shy/emotion/MER2025_Mini"
TRAIN_VIDEO_DIR = os.path.join(DATA_ROOT, "train/video")
TEST_VIDEO_DIR = os.path.join(DATA_ROOT, "test/video")
CLIP_PATH = "/home/shy/emotion/tool/transformers/clip-vit-large-patch14"
NUM_FRAMES, IMAGE_SIZE = 16, 224
NUM_CLASSES = 6
SAVE = os.path.dirname(__file__)
FEAT_CACHE = os.path.join(SAVE, "clip_video_feats.pkl")
N_FOLDS, N_SEARCH, N_REPEATS = 5, 50, 6
SEARCH_EPOCHS = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import decord
import pandas as pd


# ── 数据 ───────────────────────────────────────────────────────────
def readable(path):
    """检查视频文件是否可被 decord 解码（跳过损坏文件）。"""
    try:
        vr = decord.VideoReader(path)
        return len(vr) > 0
    except Exception:
        return False


def load_data():
    train_csv = os.path.join(DATA_ROOT, "track1_train_disdim.csv")
    test_csv = os.path.join(DATA_ROOT, "track1_test_dis.csv")
    train_df = pd.read_csv(train_csv)[["name","discrete"]].dropna(subset=["discrete"])
    test_df = pd.read_csv(test_csv)[["name","discrete"]].dropna(subset=["discrete"])
    tr_names = [n for n in train_df["name"] if readable(os.path.join(TRAIN_VIDEO_DIR, f"{n}.mp4"))]
    te_names = [n for n in test_df["name"] if readable(os.path.join(TEST_VIDEO_DIR, f"{n}.mp4"))]
    tr_labels = {n: EMOTION_TO_IDX[train_df[train_df["name"]==n]["discrete"].values[0]] for n in tr_names}
    te_labels = {n: EMOTION_TO_IDX[test_df[test_df["name"]==n]["discrete"].values[0]] for n in te_names}
    return tr_names, tr_labels, te_names, te_labels


class VideoDataset(Dataset):
    def __init__(self, names, video_dir):
        self.names, self.video_dir = names, video_dir
    def __len__(self): return len(self.names)
    def __getitem__(self, idx):
        name = self.names[idx]
        vr = decord.VideoReader(os.path.join(self.video_dir, f"{name}.mp4"))
        total = len(vr)
        step = max(total / NUM_FRAMES, 1)
        indices = [min(int(i * step), total - 1) for i in range(NUM_FRAMES)]
        frames = vr.get_batch(indices).asnumpy()
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        _, _, H, W = frames.shape
        s = min(H, W); top, left = (H-s)//2, (W-s)//2
        frames = frames[:, :, top:top+s, left:left+s]
        frames = F.interpolate(frames, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        frames = frames.float() / 255.0
        m = torch.tensor(CLIP_MEAN).view(1,3,1,1); s = torch.tensor(CLIP_STD).view(1,3,1,1)
        return (frames - m) / s, name


# ── 阶段 0: 提取 CLIP image embedding (512维) ─────────────────────
def extract_features(names, video_dir):
    ds = VideoDataset(names, video_dir)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    clip = CLIPModel.from_pretrained(CLIP_PATH, local_files_only=True).to(device)
    clip.eval()
    for p in clip.parameters(): p.requires_grad = False
    feats = {}
    with torch.no_grad():
        for frames, names_b in tqdm(loader, desc="提取 CLIP 视觉特征"):
            B, T, C, H, W = frames.shape
            frames_flat = frames.view(B * T, C, H, W).to(device)
            img = clip.get_image_features(pixel_values=frames_flat)  # (B*T, 512)
            img = img.view(B, T, -1).mean(dim=1)                     # (B, 512)
            img = img / img.norm(dim=-1, keepdim=True)
            for i, n in enumerate(names_b):
                feats[n] = img[i].cpu().numpy()
    return feats


# ── MLP 分类头 + 训练 ─────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim=768, hidden=256, ncls=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, ncls),
        )
    def forward(self, x): return self.net(x)


def train_mlp(X_tr, y_tr, X_va, y_va, hp, seed, epochs):
    """训练 MLP，返回验证 acc/f1。全量 batch（特征小，速度快）。"""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = MLP(hidden=hp["hidden"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    X_tr_t = torch.from_numpy(X_tr).float().to(device)
    y_tr_t = torch.from_numpy(y_tr).long().to(device)
    X_va_t = torch.from_numpy(X_va).float().to(device)
    for _ in range(epochs):
        model.train()
        loss = F.cross_entropy(model(X_tr_t), y_tr_t)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        preds = model(X_va_t).argmax(-1).cpu().numpy()
    return accuracy_score(y_va, preds), f1_score(y_va, preds, average="weighted")


def sample_hparams():
    return {
        "lr": 10 ** random.uniform(-4, -2),      # 1e-4 ~ 1e-2
        "wd": 10 ** random.uniform(-5, -2),      # 1e-5 ~ 1e-2
        "hidden": random.choice([128, 256, 512]),
    }


def main():
    tr_names, tr_labels, te_names, te_labels = load_data()
    print(f"Train: {len(tr_names)}, Test: {len(te_names)}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    n_search = 5 if args.quick else N_SEARCH
    n_repeats = 2 if args.quick else N_REPEATS

    # 特征
    if os.path.exists(FEAT_CACHE) and not args.reload:
        print(f"加载缓存特征 {FEAT_CACHE}")
        with open(FEAT_CACHE, "rb") as f:
            feats = pickle.load(f)
    else:
        feats = {}
        feats.update(extract_features(tr_names, TRAIN_VIDEO_DIR))
        feats.update(extract_features(te_names, TEST_VIDEO_DIR))
        with open(FEAT_CACHE, "wb") as f:
            pickle.dump(feats, f)

    X_tr_full = np.stack([feats[n] for n in tr_names])
    y_tr_full = np.array([tr_labels[n] for n in tr_names])
    X_te = np.stack([feats[n] for n in te_names])
    y_te = np.array([te_labels[n] for n in te_names])

    # 五折分层
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    folds = list(skf.split(X_tr_full, y_tr_full))

    # ── 阶段 1+2: 超参搜索 → 锁定 ──
    print(f"\n[搜索] {N_FOLDS}折 × {n_search}组超参...")
    results = []
    for _, (tr_idx, va_idx) in enumerate(folds):
        X_tr, y_tr = X_tr_full[tr_idx], y_tr_full[tr_idx]
        X_va, y_va = X_tr_full[va_idx], y_tr_full[va_idx]
        for _ in range(n_search):
            hp = sample_hparams()
            _, vf = train_mlp(X_tr, y_tr, X_va, y_va, hp, seed=42, epochs=SEARCH_EPOCHS)
            results.append({"val_f1": vf, **hp})
    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        agg[(r["lr"], r["wd"], r["hidden"])].append(r["val_f1"])
    best_key = max(agg, key=lambda k: np.mean(agg[k]))
    best = {"lr": best_key[0], "wd": best_key[1], "hidden": best_key[2], "epochs": 10,
            "avg_val_f1": float(np.mean(agg[best_key]))}
    print(f"锁定超参: lr={best['lr']:.2e} wd={best['wd']:.2e} hidden={best['hidden']}  avg_val_f1={best['avg_val_f1']:.4f}")

    # ── 阶段 3: 重复执行（全量 Train&Val 训练 → Test 评估）──
    print(f"\n[重复] {n_repeats} 种子 全量训练...")
    test_accs, test_f1s = [], []
    for seed in range(n_repeats):
        # 全量 Train&Val 训练
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        model = MLP(hidden=best["hidden"]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=best["lr"], weight_decay=best["wd"])
        X_tr_t = torch.from_numpy(X_tr_full).float().to(device)
        y_tr_t = torch.from_numpy(y_tr_full).long().to(device)
        X_te_t = torch.from_numpy(X_te).float().to(device)
        for _ in range(best["epochs"]):
            model.train()
            loss = F.cross_entropy(model(X_tr_t), y_tr_t)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            preds = model(X_te_t).argmax(-1).cpu().numpy()
        acc = accuracy_score(y_te, preds)
        f1 = f1_score(y_te, preds, average="weighted")
        test_accs.append(acc); test_f1s.append(f1)
        print(f"  seed={seed}: Test Acc={acc:.4f}  Test F1={f1:.4f}")

    mean_acc, std_acc = float(np.mean(test_accs)), float(np.std(test_accs))
    mean_f1, std_f1 = float(np.mean(test_f1s)), float(np.std(test_f1s))
    print(f"\n{'='*60}")
    print(f"CLIP-Large 单模态:  Test Acc = {mean_acc:.4f}±{std_acc:.4f}   Test F1 = {mean_f1:.4f}±{std_f1:.4f}")
    print(f"{'='*60}")
    with open(os.path.join(SAVE, "final_results.json"), "w") as f:
        json.dump({"test_acc": mean_acc, "test_acc_std": std_acc,
                   "test_f1": mean_f1, "test_f1_std": std_f1,
                   "seeds_acc": test_accs, "seeds_f1": test_f1s}, f, indent=2)


if __name__ == "__main__":
    main()
