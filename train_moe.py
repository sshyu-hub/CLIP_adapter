"""
三专家分工方案训练脚本（对应 Exp3 消融）。

用法：
    python train_moe.py --exp 3c          # 完整方案（默认）
    python train_moe.py --exp 3a          # 仅 1→3 专家 + 门控，无分工监督、无融合
    python train_moe.py --exp 3b          # + 视图解耦 + 分工监督 + 门控融合

消融逻辑（一次只改一个变量，与 Exp2 单专家基线对照）：
  3a: 三专家 + 视频级门控 + 融合，但视图不解耦（都读 CLS mean）、无 arousal/混淆对监督
      → 回答「多专家本身有没有用」
  3b: 3a + 视图解耦（E1 attention / E2 patch-topk）+ arousal + 混淆对监督
      → 回答「分工有没有用」
  3c: 3b + class-weight + logit-adj 长尾修正（默认开启）
      → 回答「长尾修正贡献多少」

checkpoint 格式与 test.py 兼容（存 {"model": state_dict, "epoch": ..., "val_f1": ...}）。
"""
import os
import sys
import random
import math
import argparse

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

import config as cfg
from dataset import OpenFaceDataset, build_label_dicts
from model import CLIPMoEEmotionModel
from losses import CompositeLoss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", type=str, default="3c", choices=["3a", "3b", "3c"])
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    p.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=cfg.DEVICE)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Exp: {args.exp}")

    # ── 数据 ──────────────────────────────────────────────────
    train_label_dict, test_label_dict, train_names, test_names, num_classes = \
        build_label_dicts(cfg.DATA_ROOT)

    train_ds = OpenFaceDataset(train_names, train_label_dict, openface_dir=cfg.OPENFACE_DIR,
                               num_frames=cfg.NUM_FRAMES, image_size=cfg.IMAGE_SIZE,
                               mean=cfg.CLIP_MEAN, std=cfg.CLIP_STD)
    test_ds = OpenFaceDataset(test_names, test_label_dict, openface_dir=cfg.OPENFACE_DIR,
                              num_frames=cfg.NUM_FRAMES, image_size=cfg.IMAGE_SIZE,
                              mean=cfg.CLIP_MEAN, std=cfg.CLIP_STD)

    # 90/10 划分训练/验证
    idxs = list(range(len(train_ds)))
    random.shuffle(idxs)
    split = int(0.9 * len(idxs))
    tr_idx, val_idx = idxs[:split], idxs[split:]
    tr_subset = Subset(train_ds, tr_idx)
    val_subset = Subset(train_ds, val_idx)

    train_loader = DataLoader(tr_subset, batch_size=args.batch_size, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=cfg.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=cfg.NUM_WORKERS, pin_memory=True)
    print(f"Train: {len(tr_subset)}  Val: {len(val_subset)}  Test: {len(test_ds)}")

    # ── 消融开关 ──────────────────────────────────────────────
    exp = args.exp
    view_decouple = exp in ("3b", "3c")
    use_specialize = exp in ("3b", "3c")      # arousal + 混淆对监督
    use_longtail = exp == "3c"                # class-weight + logit-adj
    fusion_feature = cfg.FUSION_FEATURE
    fusion_logit = cfg.FUSION_LOGIT
    print(f"view_decouple={view_decouple}  specialize={use_specialize}  longtail={use_longtail}")

    # ── 模型 ──────────────────────────────────────────────────
    model = CLIPMoEEmotionModel(
        clip_model_path=cfg.CLIP_MODEL_PATH, adapter_layers=cfg.ADAPTER_LAYERS,
        num_classes=num_classes, num_experts=cfg.NUM_EXPERTS,
        adapter_bottleneck=cfg.ADAPTER_BOTTLENECK, expert_dims=cfg.EXPERT_DIMS,
        video_expert_dims=cfg.VIDEO_EXPERT_DIMS, top_k=cfg.TOP_K,
        fusion_feature=fusion_feature, fusion_logit=fusion_logit,
        fusion_alpha=cfg.FUSION_ALPHA, num_arousal=cfg.NUM_AROUSAL,
        view_decouple=view_decouple,
    ).to(device)

    # 长尾修正：计算训练集每类样本数
    class_counts = torch.zeros(num_classes)
    for n, lab in train_label_dict.items():
        if n in train_names:
            class_counts[lab] += 1
    print(f"Class counts: {class_counts.int().tolist()}")

    criterion = CompositeLoss(
        num_classes=num_classes, idx_to_emotion=cfg.EMOTIONS,
        confusable_pairs=cfg.CONFUSABLE_PAIRS, arousal_map=cfg.EMOTION_AROUSAL,
        contrastive_temp=cfg.CONTRASTIVE_TEMP,
        w_ce=cfg.LOSS_WEIGHT_CE, w_arousal=cfg.LOSS_WEIGHT_AROUSAL if use_specialize else 0.0,
        w_fine=cfg.LOSS_WEIGHT_FINE_GRAINED if use_specialize else 0.0,
        w_aux=cfg.LOSS_WEIGHT_EXPERT3_AUX if use_specialize else 0.0,
        w_balance=cfg.LOSS_WEIGHT_GATE_ENTROPY,
        logit_adj_tau=cfg.LOGIT_ADJ_TAU if use_longtail else 0.0,
    )
    if use_longtail:
        criterion.ce_loss.set_class_counts(class_counts)

    # 可训练参数：MoEAdapter + 视频级专家 + 门控 + 分类头
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    print(f"Trainable params: {n_params/1e6:.3f}M")
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=cfg.WEIGHT_DECAY)

    # ── 学习率调度：warmup + cosine（对齐 config.WARMUP_EPOCHS / LR_SCHEDULER）──
    def get_lr(epoch):
        # epoch 为 1-indexed；warmup 阶段线性爬升，之后 cosine 衰减到 0
        if epoch <= cfg.WARMUP_EPOCHS:
            return args.lr * epoch / cfg.WARMUP_EPOCHS
        progress = (epoch - cfg.WARMUP_EPOCHS) / max(1, args.epochs - cfg.WARMUP_EPOCHS)
        return args.lr * 0.5 * (1 + math.cos(math.pi * progress))

    # ── 保存目录 ──────────────────────────────────────────────
    save_dir = os.path.join(cfg.SAVE_DIR, f"exp_{exp}")
    os.makedirs(save_dir, exist_ok=True)
    csv_file = open(os.path.join(save_dir, "metrics.csv"), "w")
    csv_file.write("epoch,train_loss,val_acc,val_f1,test_acc,test_f1,lr\n")

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        preds, gts = [], []
        for batch in loader:
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)
            out = model(frames)
            preds.extend(out["logits"].argmax(-1).cpu().numpy())
            gts.extend(labels.cpu().numpy())
        return accuracy_score(gts, preds), f1_score(gts, preds, average="weighted")

    best_val_f1, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        lr = get_lr(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"E{epoch}", leave=False):
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)

            out = model(frames)
            losses = criterion(
                out["logits"], labels,
                fine_features=out["fine_features"],
                aux_logits=out["aux_logits"],
                arousal_logits=out["arousal_logits"],
                gates_video=out["gates_video"],
            )
            loss = losses["total"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total_loss += loss.item()

        val_acc, val_f1 = evaluate(val_loader)
        test_acc, test_f1 = evaluate(test_loader)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:2d}  Loss={avg_loss:.4f}  Val F1={val_f1:.4f}  Test F1={test_f1:.4f}  LR={lr:.2e}")
        csv_file.write(f"{epoch},{avg_loss:.6f},{val_acc:.6f},{val_f1:.6f},{test_acc:.6f},{test_f1:.6f},{lr:.6e}\n")
        csv_file.flush()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {
                "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "epoch": epoch, "val_f1": val_f1,
                # 架构 flag：推理时据此重建模型，避免训练/推理结构不一致
                "arch": {
                    "video_expert_dims": cfg.VIDEO_EXPERT_DIMS,
                    "top_k": cfg.TOP_K,
                    "fusion_feature": fusion_feature,
                    "fusion_logit": fusion_logit,
                    "fusion_alpha": cfg.FUSION_ALPHA,
                    "num_arousal": cfg.NUM_AROUSAL,
                    "view_decouple": view_decouple,
                },
            }

    csv_file.close()

    if best_state is not None:
        torch.save(best_state, os.path.join(save_dir, "best_model.pt"))
        print(f"Saved best_model.pt (val_f1={best_val_f1:.4f})")

    # 最终测试
    if best_state is not None:
        model.load_state_dict(best_state["model"])
    model.eval()
    test_acc, test_f1 = evaluate(test_loader)
    print(f"\n{'='*60}")
    print(f"Exp {exp}  Best Val F1={best_val_f1:.4f}  Test Acc={test_acc:.4f}  Test F1={test_f1:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
