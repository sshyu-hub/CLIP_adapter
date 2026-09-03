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

# MoE Adapter 注入层（0-indexed，第 13/21/23 层后）
ADAPTER_LAYERS = [13, 21, 23]

# 每层专家数（单级 3-expert：E0 通用 / E1 VA / E2 细粒度）
NUM_EXPERTS = 3

# Expert bottleneck hidden dim
ADAPTER_BOTTLENECK = CLIP_HIDDEN_DIM // 8  # 128
# 每专家 bottleneck（统一 128）
EXPERT_DIMS = [128, 128, 128]

# ── LN tuning ────────────────────────────────────────────────
# 解冻指定层 LayerNorm 做 tuning（lr 单独设，见 LN_LR）
LN_TUNING_LAYERS = [13, 21, 23]   # 与 ADAPTER_LAYERS 一致
LN_LR = 1e-5

# ── Emotion Classes ────────────────────────────────────────────────
EMOTIONS = ["neutral", "angry", "happy", "sad", "worried", "surprise"]
NUM_CLASSES = len(EMOTIONS)

# Emotion → index mapping
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}

# E1 唤醒/动态专家监督：arousal 4 级单调强度
EMOTION_AROUSAL = {
    "neutral": 0,
    "sad": 1,
    "worried": 2,
    "angry": 3,
    "happy": 3,
    "surprise": 3,
}
NUM_AROUSAL = 4

# E1 VA 监督：valence 3 级（neutral / positive / negative）
EMOTION_VALENCE = {
    "neutral": 0,    # neutral
    "happy": 1,      # positive
    "surprise": 1,   # positive
    "angry": 2,      # negative
    "sad": 2,        # negative
    "worried": 2,    # negative
}
NUM_VALENCE = 3  # neutral / positive / negative

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

# Loss weights（单级 3-expert 分工监督）
LOSS_WEIGHT_CE = 1.0           # 主 CE + Balanced Softmax(τ)
LOSS_WEIGHT_CONTRASTIVE = 0.2  # 各专家分别监督对比
LOSS_WEIGHT_VA = 0.1           # E1 valence(3)+arousal(4)，各 0.05
LOSS_WEIGHT_HFCL = 0.1         # E2 混淆对 hard-negative 对比
LOSS_WEIGHT_AUX = 0.2          # E2 Aux CE
LOSS_WEIGHT_DIVERSITY = 0.05   # 专家输出正交正则

# 长尾修正：主 CE 的 logit-adjustment 强度（Balanced Softmax τ）
LOGIT_ADJ_TAU = 0.1

MIXUP_ALPHA = 0.4

# Contrastive loss temperature
CONTRASTIVE_TEMP = 0.07

# ── Device ─────────────────────────────────────────────────────────
DEVICE = "cuda"
