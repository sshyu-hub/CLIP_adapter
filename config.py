"""
Configuration for MoE Adapter + CLIP visual emotion recognition.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────
DATA_ROOT = "/home/shy/emotion/MER2025_Mini"
CLIP_MODEL_PATH = "/home/shy/emotion/tool/transformers/clip-vit-large-patch14"
OPENFACE_DIR = os.path.join(DATA_ROOT, "openface_face")
TRAIN_VIDEO_DIR = os.path.join(DATA_ROOT, "train/video")
TEST_VIDEO_DIR = os.path.join(DATA_ROOT, "test/video")
LABEL_CSV = os.path.join(DATA_ROOT, "label.csv")
TRAIN_CSV = os.path.join(DATA_ROOT, "track1_train_disdim.csv")
TEST_CSV = os.path.join(DATA_ROOT, "track1_test_dis.csv")

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved")
MODEL_DIR = os.path.join(SAVE_DIR, "model")
LOG_DIR = os.path.join(SAVE_DIR, "log")

# ── Model ──────────────────────────────────────────────────────────
CLIP_HIDDEN_DIM = 1024
CLIP_INTERMEDIATE_DIM = 4096
CLIP_NUM_LAYERS = 24

# MoE Adapter insertion points (1-indexed layer numbers: 12, 18, 24)
# CLIP ViT-L/14 has 24 layers, 0-indexed: [11, 17, 23]
ADAPTER_LAYERS = [13, 21, 23]

# Number of experts per adapter (per-token MoE: 通用 + VA)
NUM_EXPERTS = 2

# Expert bottleneck hidden dim
ADAPTER_BOTTLENECK = CLIP_HIDDEN_DIM // 8  # 128
# Per-expert bottleneck dims: [E0=通用, E1=VA]
EXPERT_DIMS = [256, 128]

# ── 视频级三专家（新方案，作用于 last_hidden_state）────────────────
# E0 通用/布局 + E1 唤醒/动态 + E2 细粒度/纹理，瓶颈递减 256/128/64
VIDEO_EXPERT_DIMS = [256, 128, 64]
# E2 细粒度专家：时间维度 top-k mean（取特征范数最强的 k 帧）
TOP_K = 5
# 门控融合：特征融合 + logit 融合（否则 aux 损失无法进入最终预测）
FUSION_FEATURE = True
FUSION_LOGIT = True
# 专家集成在最终 logits 上的权重
FUSION_ALPHA = 0.5

# ── Emotion Classes ────────────────────────────────────────────────
EMOTIONS = ["neutral", "angry", "happy", "sad", "worried", "surprise"]
NUM_CLASSES = len(EMOTIONS)

# Emotion → index mapping
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}

# Arousal grouping for video-level Expert E1 (唤醒/动态专家) — 4 级单调强度
# 替换原 3 级 valence：neutral=0 / sad=1 / worried=2 / angry·happy·surprise=3
# 同时解 layer1（neutral vs rest）+ layer2（sad/worried/angry 按强度分开）
EMOTION_AROUSAL = {
    "neutral": 0,
    "sad": 1,
    "worried": 2,
    "angry": 3,
    "happy": 3,
    "surprise": 3,
}
NUM_AROUSAL = 4

# Confusable emotion pairs for fine-grained expert contrastive learning
CONFUSABLE_PAIRS = [
    ("angry", "worried"),     # both negative, easily confused
    ("happy", "surprise"),    # both positive / high arousal
    ("sad", "worried"),       # both low arousal
    ("neutral", "sad"),       # both low activation
]

# ── Data ───────────────────────────────────────────────────────────
NUM_FRAMES = 16          # uniformly sampled frames per video
IMAGE_SIZE = 224         # CLIP input size (resize from 112)
BATCH_SIZE = 8          # 3090 24GB, ~16-18 GB VRAM
NUM_WORKERS = 4
USE_AMP = True           # automatic mixed precision (saves ~40% GPU memory)

# CLIP normalization
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# ── Training ───────────────────────────────────────────────────────
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 3
LR_SCHEDULER = "cosine"

# Loss weights（对齐新三专家方案）
LOSS_WEIGHT_CE = 1.0
LOSS_WEIGHT_AROUSAL = 0.3      # E1 arousal 4 级
LOSS_WEIGHT_FINE_GRAINED = 0.2  # E2 混淆对对比
LOSS_WEIGHT_EXPERT3_AUX = 0.2   # E2 Aux CE
LOSS_WEIGHT_GATE_ENTROPY = 0.02  # 负载均衡 CV²（异构专家弱均衡，原 0.1）

# 长尾修正：主 CE 的 logit-adjustment 强度（先验对数加权）
LOGIT_ADJ_TAU = 0.3

# Contrastive loss temperature
CONTRASTIVE_TEMP = 0.07

# ── Device ─────────────────────────────────────────────────────────
DEVICE = "cuda"
