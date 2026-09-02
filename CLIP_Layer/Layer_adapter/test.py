"""
测试 CLIP-Large + Adapter 模型（best_adapter.pt）

用法:
    python test_adapter.py
"""
import os, sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
from transformers import CLIPModel
import pandas as pd

# 路径设置（请根据实际环境调整）
DATA_ROOT = "/home/shy/emotion/MER2025_Mini"
OPENFACE_DIR = os.path.join(DATA_ROOT, "openface_face")
CLIP_PATH = "/home/shy/emotion/tool/transformers/clip-vit-large-patch14"
MODEL_PATH = "/home/shy/emotion/CLIP_adapter/CLIP_Layer/Layer_adapter/best_adapter.pt"

# 模型超参数（需与训练时一致）
ADAPTER_LAYERS = [15, 21, 23]          # 插入 Adapter 的层索引
BOTTLENECK = 128                       # Adapter 瓶颈维度
NUM_FRAMES = 16                        # 采样帧数
IMAGE_SIZE = 224                       # 输入尺寸
NUM_CLASSES = 6                        # 情感类别数
BATCH_SIZE = 8                         # 测试批大小
EMOS = ["neutral", "angry", "happy", "sad", "worried", "surprise"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

# 从 config 导入 CLIP 均值和标准差（如果失败则手动定义）
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from config import EMOTION_TO_IDX, CLIP_MEAN, CLIP_STD
except ImportError:
    CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
    EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOS)}

# ── 数据加载（测试集）──────────────────────────────────────────
def load_test_data():
    test_csv = os.path.join(DATA_ROOT, "track1_test_dis.csv")
    test_df = pd.read_csv(test_csv)[["name", "discrete"]].dropna(subset=["discrete"])
    valid = lambda n: os.path.exists(os.path.join(OPENFACE_DIR, f"{n}.npy"))
    te_names = [n for n in test_df["name"] if valid(n)]
    te_labels = {n: EMOTION_TO_IDX[test_df[test_df["name"] == n]["discrete"].values[0]] for n in te_names}
    return te_names, te_labels

class FDataset(Dataset):
    def __init__(self, names, labels):
        self.names, self.labels = names, labels
        self.dir = OPENFACE_DIR
    def __len__(self):
        return len(self.names)
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
        m = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        s = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        frames = (frames - m) / s
        return frames, self.labels.get(name, -1)

# ── Adapter 模块（与训练时相同）────────────────────────────────
class Adapter(nn.Module):
    def __init__(self, dim=1024, bottleneck=BOTTLENECK):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.zeros_(self.up.bias)
    def forward(self, x):
        return self.up(F.gelu(self.down(x)))

# ── MLP 分类头（与训练时相同）────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim=1024, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, NUM_CLASSES)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# ── 构建模型并加载权重 ────────────────────────────────────────
def build_model():
    # 加载 CLIP vision 并冻结
    clip = CLIPModel.from_pretrained(CLIP_PATH, local_files_only=True)
    vision = clip.vision_model
    for p in vision.parameters():
        p.requires_grad = False
    vision.eval()

    # 创建 Adapters（使用 ModuleDict，键为层索引字符串）
    adapters = nn.ModuleDict({str(l): Adapter() for l in ADAPTER_LAYERS})
    # 注册 forward hook
    for l in ADAPTER_LAYERS:
        a = adapters[str(l)]
        vision.encoder.layers[l].register_forward_hook(
            lambda m, i, o, a=a: (o[0] + a(o[0]),) + o[1:]
        )

    mlp = MLP()
    return vision, adapters, mlp

def load_model(vision, adapters, mlp, model_path):
    state_dict = torch.load(model_path, map_location=device)
    # 分别加载适配器和 MLP 的参数（strict=False 忽略不匹配的键）
    adapters.load_state_dict(state_dict, strict=False)
    mlp.load_state_dict(state_dict, strict=False)
    adapters.to(device)
    mlp.to(device)
    vision.to(device)
    vision.eval()
    adapters.eval()
    mlp.eval()
    return vision, adapters, mlp

# ── 特征编码（与训练时一致：逐帧处理）────────────────────────
@torch.no_grad()
def encode(frames, vision):
    B, T, C, H, W = frames.shape
    all_cls = []
    for t in range(T):
        out = vision(pixel_values=frames[:, t])
        all_cls.append(out.last_hidden_state[:, 0, :])  # (B, 1024)
    cls = torch.stack(all_cls, dim=1).mean(dim=1)       # (B, 1024)
    return cls

# ── 主测试流程 ────────────────────────────────────────────────
def main():
    te_names, te_labels = load_test_data()
    print(f"测试样本数: {len(te_names)}")

    # 构建数据加载器
    test_ds = FDataset(te_names, te_labels)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    # 构建模型
    vision, adapters, mlp = build_model()
    vision, adapters, mlp = load_model(vision, adapters, mlp, MODEL_PATH)
    print("模型加载完成。")

    # 推理
    preds, gts = [], []
    with torch.no_grad():
        for frames, labels in tqdm(test_loader, desc="测试中"):
            frames = frames.to(device)
            cls = encode(frames, vision)
            logits = mlp(cls)
            preds.extend(logits.argmax(-1).cpu().numpy())
            gts.extend(labels.numpy())

    # 计算指标
    test_acc = accuracy_score(gts, preds)
    test_f1 = f1_score(gts, preds, average="weighted")
    print("\n" + "="*60)
    print(f"Test Acc = {test_acc:.4f}   Test F1 = {test_f1:.4f}")
    print("="*60)
    print(classification_report(gts, preds, target_names=EMOS, digits=4))

if __name__ == "__main__":
    main()