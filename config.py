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
# 高频专家 bottleneck (独立分支, patch max pooling)
HF_BOTTLENECK = 64

# ── Emotion Classes ────────────────────────────────────────────────
EMOTIONS = ["neutral", "angry", "happy", "sad", "worried", "surprise"]
NUM_CLASSES = len(EMOTIONS)

# Emotion → index mapping
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}

# Valence grouping for Expert 2 (boundary expert)
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

# Loss weights
LOSS_WEIGHT_CE = 1.0
LOSS_WEIGHT_CONTRASTIVE = 0.2
LOSS_WEIGHT_BOUNDARY = 0.1
LOSS_WEIGHT_FINE_GRAINED = 0.1
LOSS_WEIGHT_EXPERT3_AUX = 0.2
LOSS_WEIGHT_DIVERSITY = 0.05
LOSS_WEIGHT_GATE_ENTROPY = 0.1

MIXUP_ALPHA = 0.4

# Contrastive loss temperature
CONTRASTIVE_TEMP = 0.07

# ── Device ─────────────────────────────────────────────────────────
DEVICE = "cuda"
