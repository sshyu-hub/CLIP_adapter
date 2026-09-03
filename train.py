"""
CLIP-Large + MoE Adapter（per-token）训练脚本。

组件统一：
  model.py    CLIPMoEEmotionModel（MoE 专家 + 高频分支）
  dataset.py  OpenFaceDataset / build_label_dicts（OpenFace 对齐帧）
  losses.py   CompositeLoss（CE + 对比 + VA + HFCL + Aux CE + 负载均衡）

损失权重与模型配置全部从 config.py 读取。
"""
import os
import math
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

import config as cfg
from dataset import OpenFaceDataset, build_label_dicts
from model import CLIPMoEEmotionModel
from losses import CompositeLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = cfg.USE_AMP and device.type == "cuda"


def main():
    # ── 数据 ───────────────────────────────────────────────────────
    train_label_dict, test_label_dict, train_names, test_names, num_classes = build_label_dicts(cfg.DATA_ROOT)

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(train_names))
    split = int(0.9 * len(train_names))
    tr_train = [train_names[i] for i in perm[:split]]
    tr_val = [train_names[i] for i in perm[split:]]

    ds_kwargs = dict(openface_dir=cfg.OPENFACE_DIR, num_frames=cfg.NUM_FRAMES,
                     image_size=cfg.IMAGE_SIZE, mean=cfg.CLIP_MEAN, std=cfg.CLIP_STD)
    train_ds = OpenFaceDataset(tr_train, train_label_dict, **ds_kwargs)
    val_ds = OpenFaceDataset(tr_val, train_label_dict, **ds_kwargs)
    test_ds = OpenFaceDataset(test_names, test_label_dict, **ds_kwargs)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE * 2, shuffle=False,
                            num_workers=cfg.NUM_WORKERS // 2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE * 2, shuffle=False,
                             num_workers=cfg.NUM_WORKERS // 2, pin_memory=True)

    # ── 模型 + 损失 ────────────────────────────────────────────────
    model = CLIPMoEEmotionModel(
        clip_model_path=cfg.CLIP_MODEL_PATH, adapter_layers=cfg.ADAPTER_LAYERS,
        num_classes=num_classes, num_experts=cfg.NUM_EXPERTS,
        adapter_bottleneck=cfg.ADAPTER_BOTTLENECK, expert_dims=cfg.EXPERT_DIMS,
        ln_tuning_layers=cfg.LN_TUNING_LAYERS,
        num_arousal=cfg.NUM_AROUSAL, num_valence=cfg.NUM_VALENCE,
    ).to(device)

    criterion = CompositeLoss(
        num_classes=num_classes, idx_to_emotion=cfg.EMOTIONS,
        confusable_pairs=cfg.CONFUSABLE_PAIRS,
        arousal_map=cfg.EMOTION_AROUSAL, valence_map=cfg.EMOTION_VALENCE,
        contrastive_temp=cfg.CONTRASTIVE_TEMP,
        w_ce=cfg.LOSS_WEIGHT_CE,
        w_contrastive=cfg.LOSS_WEIGHT_CONTRASTIVE,
        w_valence=cfg.LOSS_WEIGHT_VA / 2,
        w_arousal=cfg.LOSS_WEIGHT_VA / 2,
        w_hfcl=cfg.LOSS_WEIGHT_HFCL,
        w_aux=cfg.LOSS_WEIGHT_AUX,
        w_div=cfg.LOSS_WEIGHT_DIVERSITY,
        logit_adj_tau=cfg.LOGIT_ADJ_TAU,
    ).to(device)

    # 类别先验 → class-balanced CE
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for c, n in Counter(train_label_dict[n] for n in tr_train).items():
        counts[c] = n
    criterion.ce_loss.set_class_counts(counts.to(device))

    # 可训练参数分组：LN tuning 用 LN_LR，其余（MoE/门控/分类头）用主 lr
    ln_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "vision_model.encoder.layers" in n and "layer_norm" in n:
            ln_params.append(p)
        else:
            other_params.append(p)
    trainable = ln_params + other_params
    optimizer = torch.optim.AdamW([
        {"params": ln_params, "lr": cfg.LN_LR},
        {"params": other_params, "lr": cfg.LEARNING_RATE},
    ], weight_decay=cfg.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 学习率：线性 warmup + cosine
    def lr_lambda(epoch):
        if epoch < cfg.WARMUP_EPOCHS:
            return (epoch + 1) / cfg.WARMUP_EPOCHS
        p = (epoch - cfg.WARMUP_EPOCHS) / max(cfg.EPOCHS - cfg.WARMUP_EPOCHS, 1)
        return 0.5 * (1 + math.cos(math.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"可训练参数: {sum(p.numel() for p in trainable)/1e6:.3f}M | "
          f"train {len(tr_train)} / val {len(tr_val)} / test {len(test_names)}")

    # ── 评估（只用主分类 logits，同 test.py） ───────────────────────
    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        preds, gts = [], []
        for batch in tqdm(loader, desc="Eval", leave=False):
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(frames)["logits"]
            preds.extend(logits.argmax(-1).cpu().numpy())
            gts.extend(labels.cpu().numpy())
        model.train()
        return accuracy_score(gts, preds), f1_score(gts, preds, average="weighted"), gts, preds

    # ── 训练循环 ───────────────────────────────────────────────────
    save_dir = cfg.MODEL_DIR
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "metrics.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,val_acc,val_f1,lr\n")

    best_val_f1 = 0.0
    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        total_loss, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"E{epoch}/{cfg.EPOCHS}", leave=False)
        for batch in pbar:
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(frames)
                losses = criterion(
                    out["logits"], labels,
                    expert_features=out["expert_features"],
                    valence_logits=out["valence_logits"],
                    arousal_logits=out["arousal_logits"],
                    aux_logits=out["aux_logits"],
                )
                loss = losses["total"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", ce=f"{losses['ce'].item():.3f}")

        scheduler.step()

        val_acc, val_f1, _, _ = evaluate(val_loader)

        avg_loss = total_loss / max(n, 1)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:2d}  Loss={avg_loss:.4f}  Val F1={val_f1:.4f}  LR={lr:.2e}")
        with open(csv_path, "a") as f:
            f.write(f"{epoch},{avg_loss:.6f},{val_acc:.6f},{val_f1:.6f},{lr:.2e}\n")

        ckpt = {"model": model.state_dict(), "criterion": criterion.state_dict(), "epoch": epoch}
        torch.save(ckpt, os.path.join(save_dir, "latest_model.pt"))
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(ckpt, os.path.join(save_dir, "best_model.pt"))

    # ── 训练结束：加载最佳验证模型，测试集只评估这一次 ───────────
    ckpt = torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    test_acc, test_f1, _, _ = evaluate(test_loader)

    print(f"\n{'='*60}")
    print(f"Best Val F1={best_val_f1:.4f}")
    print(f"Test Acc={test_acc:.4f}  Test F1={test_f1:.4f}")
    print(f"Saved to {save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
