#!/usr/bin/env python3
"""
Extract video-level features from trained CLIP+MoE Adapter model.

Output: .npy features for each sample, saved to --output-dir.

Usage:
    # Extract test set features (best model)
    python extract_features.py

    # Extract all features with a specific checkpoint
    python extract_features.py --resume saved/model/best_model.pt --split all

    # Custom output directory
    python extract_features.py --output-dir ./my_features
"""
import os
import sys
import argparse

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from dataset import OpenFaceDataset, build_label_dicts
from model import CLIPMoEEmotionModel


@torch.no_grad()
def extract_features(model, loader, device) -> dict[str, np.ndarray]:
    """Extract video-level features for all samples.

    Returns:
        {sample_name: feature_array (1024,)}
    """
    model.eval()
    features_dict = {}

    for batch in tqdm(loader, desc="Extracting features"):
        frames = batch["frames"].to(device)
        names = batch["name"]

        with torch.amp.autocast("cuda", enabled=cfg.USE_AMP):
            feats = model.extract_features(frames)  # (B, 1024)

        feats = feats.float().cpu().numpy()
        for name, feat in zip(names, feats):
            features_dict[name] = feat

    return features_dict


def main():
    parser = argparse.ArgumentParser(description="Extract video-level features")
    parser.add_argument("--resume", type=str, default=None,
                        help="Model checkpoint (default: auto-find best_model.pt)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test", "all"])
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=cfg.DEVICE)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Default checkpoint
    if args.resume is None:
        default_ckpt = os.path.join(cfg.MODEL_DIR, "best_model.pt")
        if os.path.exists(default_ckpt):
            args.resume = default_ckpt
        else:
            print(f"No checkpoint found at {default_ckpt}. "
                  f"Train first or specify --resume.")
            sys.exit(1)

    # Default output
    if args.output_dir is None:
        args.output_dir = os.path.join(cfg.SAVE_DIR, "features")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────
    print("Loading labels...")
    train_label_dict, test_label_dict, train_names, test_names, num_classes = \
        build_label_dicts(cfg.DATA_ROOT)

    if args.split == "train":
        names = train_names
        label_dict = train_label_dict
    elif args.split == "test":
        names = test_names
        label_dict = test_label_dict
    else:
        names = train_names + test_names
        label_dict = {**train_label_dict, **test_label_dict}

    ds = OpenFaceDataset(
        names=names,
        label_dict=label_dict,
        openface_dir=cfg.OPENFACE_DIR,
        num_frames=cfg.NUM_FRAMES,
        image_size=cfg.IMAGE_SIZE,
        mean=cfg.CLIP_MEAN,
        std=cfg.CLIP_STD,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
    )
    print(f"Split: {args.split}  |  Samples: {len(ds)}")

    # ── Model ─────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.resume}")
    model = CLIPMoEEmotionModel(
        clip_model_path=cfg.CLIP_MODEL_PATH,
        adapter_layers=cfg.ADAPTER_LAYERS,
        num_classes=num_classes,
        num_experts=cfg.NUM_EXPERTS,
        adapter_bottleneck=cfg.ADAPTER_BOTTLENECK,
    ).to(device)

    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded model from epoch {ckpt.get('epoch', '?')}, "
          f"val_f1={ckpt.get('val_f1', 'N/A')}")

    # ── Extract ───────────────────────────────────────────────
    print("Extracting features...")
    features = extract_features(model, loader, device)

    # Save individual .npy files (one per sample)
    for name, feat in tqdm(features.items(), desc="Saving"):
        np.save(os.path.join(args.output_dir, f"{name}.npy"), feat)

    # Also save a combined file for convenience
    combined_path = os.path.join(args.output_dir, f"features_{args.split}.npz")
    np.savez(combined_path, **{n: f for n, f in features.items()})
    print(f"Saved combined features to: {combined_path}")

    # Summary
    feat_dim = next(iter(features.values())).shape[0]
    print(f"\nDone. {len(features)} samples, feature dim = {feat_dim}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
