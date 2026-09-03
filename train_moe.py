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
    use_specialize = exp in ("3b", "3c")      # Contra/VA/HFCL/Aux/Div 分工监督
    use_longtail = exp == "3c"                # class-weight + logit-adj (BS τ)
    print(f"specialize={use_specialize}  longtail={use_longtail}")

    # ── 模型（单级 3-expert，结构不随 exp 变）────────────────
    model = CLIPMoEEmotionModel(
        clip_model_path=cfg.CLIP_MODEL_PATH, adapter_layers=cfg.ADAPTER_LAYERS,
        num_classes=num_classes, num_experts=cfg.NUM_EXPERTS,
        adapter_bottleneck=cfg.ADAPTER_BOTTLENECK, expert_dims=cfg.EXPERT_DIMS,
        ln_tuning_layers=cfg.LN_TUNING_LAYERS,
        num_arousal=cfg.NUM_AROUSAL, num_valence=cfg.NUM_VALENCE,
    ).to(device)

    # 长尾修正：计算训练集每类样本数
    class_counts = torch.zeros(num_classes)
    for n, lab in train_label_dict.items():
        if n in train_names:
            class_counts[lab] += 1
    print(f"Class counts: {class_counts.int().tolist()}")

    criterion = CompositeLoss(
        num_classes=num_classes, idx_to_emotion=cfg.EMOTIONS,
        confusable_pairs=cfg.CONFUSABLE_PAIRS,
        arousal_map=cfg.EMOTION_AROUSAL, valence_map=cfg.EMOTION_VALENCE,
        contrastive_temp=cfg.CONTRASTIVE_TEMP,
        w_ce=cfg.LOSS_WEIGHT_CE,
        w_contrastive=cfg.LOSS_WEIGHT_CONTRASTIVE if use_specialize else 0.0,
        w_valence=cfg.LOSS_WEIGHT_VA / 2 if use_specialize else 0.0,
        w_arousal=cfg.LOSS_WEIGHT_VA / 2 if use_specialize else 0.0,
        w_hfcl=cfg.LOSS_WEIGHT_HFCL if use_specialize else 0.0,
        w_aux=cfg.LOSS_WEIGHT_AUX if use_specialize else 0.0,
        w_div=cfg.LOSS_WEIGHT_DIVERSITY if use_specialize else 0.0,
        logit_adj_tau=cfg.LOGIT_ADJ_TAU if use_longtail else 0.0,
    )
    if use_longtail:
        criterion.ce_loss.set_class_counts(class_counts)

    # 可训练参数分组：LN tuning 用 LN_LR，其余（MoE/门控/分类头）用主 lr
    ln_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "vision_model.encoder.layers" in n and "layer_norm" in n:
            ln_params.append(p)
        else:
            other_params.append(p)
    n_params = sum(p.numel() for p in ln_params + other_params)
    print(f"Trainable params: {n_params/1e6:.3f}M (LN tuning: {sum(p.numel() for p in ln_params)/1e6:.3f}M)")
    optimizer = torch.optim.AdamW([
        {"params": ln_params, "lr": cfg.LN_LR},
        {"params": other_params, "lr": args.lr},
    ], weight_decay=cfg.WEIGHT_DECAY)

    # ── 保存目录 ──────────────────────────────────────────────
    save_dir = os.path.join(cfg.SAVE_DIR, f"exp_{exp}")
    os.makedirs(save_dir, exist_ok=True)
    csv_file = open(os.path.join(save_dir, "metrics.csv"), "w")
    csv_file.write("epoch,train_loss,val_acc,val_f1,test_acc,test_f1\n")

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
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"E{epoch}", leave=False):
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)

            out = model(frames)
            losses = criterion(
                out["logits"], labels,
                expert_features=out["expert_features"],
                valence_logits=out["valence_logits"],
                arousal_logits=out["arousal_logits"],
                aux_logits=out["aux_logits"],
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
        print(f"Epoch {epoch:2d}  Loss={avg_loss:.4f}  Val F1={val_f1:.4f}  Test F1={test_f1:.4f}")
        csv_file.write(f"{epoch},{avg_loss:.6f},{val_acc:.6f},{val_f1:.6f},{test_acc:.6f},{test_f1:.6f}\n")
        csv_file.flush()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {
                "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "epoch": epoch, "val_f1": val_f1,
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
