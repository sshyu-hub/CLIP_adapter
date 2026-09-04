"""Test best model with TTA (horizontal flip).

用法：
    python test.py                        # 用 cfg.MODEL_DIR/best_model.pt（旧路径）
    python test.py --ckpt saved/exp_3c/best_model.pt   # 指定三专家方案 checkpoint

会从 checkpoint 的 "arch" 字段恢复模型结构（view_decouple / fusion 等），
保证与训练时一致；若 checkpoint 无 arch 字段（旧模型），回退到 config 默认值。
"""
import os
import argparse

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

import config as cfg
from dataset import OpenFaceDataset, build_label_dicts
from model import CLIPMoEEmotionModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=os.path.join(cfg.MODEL_DIR, "best_model.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, test_label_dict, _, test_names, num_classes = build_label_dicts(cfg.DATA_ROOT)
    test_ds = OpenFaceDataset(test_names, test_label_dict, openface_dir=cfg.OPENFACE_DIR,
                              num_frames=cfg.NUM_FRAMES, image_size=cfg.IMAGE_SIZE,
                              mean=cfg.CLIP_MEAN, std=cfg.CLIP_STD)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    ckpt_path = args.ckpt
    print(f"Loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 从 checkpoint 恢复架构 flag（缺省回退 config）
    arch = ckpt.get("arch", {})
    model = CLIPMoEEmotionModel(
        clip_model_path=cfg.CLIP_MODEL_PATH, adapter_layers=cfg.ADAPTER_LAYERS,
        num_classes=num_classes, num_experts=cfg.NUM_EXPERTS,
        adapter_bottleneck=cfg.ADAPTER_BOTTLENECK,
        expert_dims=cfg.EXPERT_DIMS,
        video_expert_dims=arch.get("video_expert_dims", getattr(cfg, "VIDEO_EXPERT_DIMS", [256, 128, 64])),
        top_k=arch.get("top_k", getattr(cfg, "TOP_K", 5)),
        fusion_feature=arch.get("fusion_feature", getattr(cfg, "FUSION_FEATURE", True)),
        fusion_logit=arch.get("fusion_logit", getattr(cfg, "FUSION_LOGIT", True)),
        fusion_alpha=arch.get("fusion_alpha", getattr(cfg, "FUSION_ALPHA", 0.5)),
        num_arousal=arch.get("num_arousal", getattr(cfg, "NUM_AROUSAL", 4)),
        view_decouple=arch.get("view_decouple", True),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    all_preds, all_labels = [], []
    for batch in tqdm(test_loader, desc="Testing"):
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            logits_orig = model(frames)["logits"]
            logits_flip = model(torch.flip(frames, dims=[-1]))["logits"]
            logits = (logits_orig + logits_flip) / 2
        all_preds.extend(logits.argmax(-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    print(f"\nTest Results (TTA):")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 (weighted): {f1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=cfg.EMOTIONS, digits=4))


if __name__ == "__main__":
    main()
