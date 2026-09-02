"""
Dataset for loading visual frames from MER2025_Mini.

Supports both OpenFace .npy files and raw .mp4 video files (via decord).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

try:
    import decord
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False


class VideoDataset(Dataset):
    """Loads frames from raw .mp4 video files using decord (fast).

    Reads video, uniformly samples `num_frames`, center-crops to square, resizes to 224×224.
    """

    def __init__(
        self,
        names: list[str],
        label_dict: dict[str, int],
        video_dir: str,
        num_frames: int = 16,
        image_size: int = 224,
        mean: tuple = (0.48145466, 0.4578275, 0.40821073),
        std: tuple = (0.26862954, 0.26130258, 0.27577711),
    ):
        self.names = names
        self.label_dict = label_dict
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.image_size = image_size
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx: int) -> dict:
        name = self.names[idx]
        path = os.path.join(self.video_dir, f"{name}.mp4")

        if DECORD_AVAILABLE:
            vr = decord.VideoReader(path)
            total = len(vr)
            # Uniform sample
            step = max(total / self.num_frames, 1)
            indices = [min(int(i * step), total - 1) for i in range(self.num_frames)]
            frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C) uint8
            frames = torch.from_numpy(frames)
        else:
            from torchvision.io import read_video
            frames, _, _ = read_video(path, pts_unit="sec")
            total = frames.shape[0]
            if total <= self.num_frames:
                last = frames[-1:].repeat(self.num_frames - total, 1, 1, 1)
                frames = torch.cat([frames, last], dim=0)
            else:
                step = total / self.num_frames
                indices = [min(int(i * step), total - 1) for i in range(self.num_frames)]
                frames = frames[indices]

        # Center crop to square
        if frames.shape[-1] == 3:  # (T, H, W, C) → (T, C, H, W)
            frames = frames.permute(0, 3, 1, 2)
        _, _, H, W = frames.shape
        s = min(H, W)
        top, left = (H - s) // 2, (W - s) // 2
        frames = frames[:, :, top:top + s, left:left + s]

        frames = TF.resize(frames, [self.image_size, self.image_size], antialias=True)
        frames = frames.float() / 255.0
        frames = TF.normalize(frames, mean=self.mean, std=self.std)

        label = self.label_dict.get(name, -1)

        return {
            "frames": frames,
            "label": torch.tensor(label, dtype=torch.long),
            "name": name,
        }


class OpenFaceDataset(Dataset):
    """Loads OpenFace face-cropped frames.

    Each .npy file: (F, 112, 112, 3) uint8 array.
    Uniformly samples `num_frames` frames, resizes to 224x224, normalizes for CLIP.
    """

    def __init__(
        self,
        names: list[str],
        label_dict: dict[str, int],
        openface_dir: str,
        num_frames: int = 16,
        image_size: int = 224,
        mean: tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
        std: tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711),
    ):
        self.names = names
        self.label_dict = label_dict
        self.openface_dir = openface_dir
        self.num_frames = num_frames
        self.image_size = image_size
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.names)

    def _sample_frames(self, frames: np.ndarray) -> np.ndarray:
        """Uniformly sample num_frames from the frame sequence."""
        total = frames.shape[0]
        if total <= self.num_frames:
            # Pad by repeating last frame if too short
            indices = list(range(total)) + [total - 1] * (self.num_frames - total)
        else:
            step = total / self.num_frames
            indices = [min(int(i * step), total - 1) for i in range(self.num_frames)]
        return frames[indices]

    def __getitem__(self, idx: int) -> dict:
        name = self.names[idx]
        path = os.path.join(self.openface_dir, f"{name}.npy")

        frames = np.load(path)                     # (F, 112, 112, 3) uint8
        frames = self._sample_frames(frames)       # (num_frames, 112, 112, 3)

        # Convert to tensors: (num_frames, C, H, W)
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)       # (T, C, H, W)

        # Resize to CLIP input size
        frames = TF.resize(frames, [self.image_size, self.image_size], antialias=True)
        frames = TF.normalize(frames, mean=self.mean, std=self.std)

        label = self.label_dict.get(name, -1)

        return {
            "frames": frames,                      # (num_frames, 3, 224, 224)
            "label": torch.tensor(label, dtype=torch.long),
            "name": name,
        }


def build_label_dicts(data_root: str) -> tuple[dict, dict, list[str], list[str], int]:
    """Build label dictionaries and train/test name lists from MER2025_Mini.

    Returns:
        train_label_dict: {name: class_idx}
        test_label_dict:  {name: class_idx}
        train_names: list of train sample names
        test_names: list of test sample names
        num_classes: number of emotion classes
    """
    from config import EMOTION_TO_IDX

    train_csv = os.path.join(data_root, "track1_train_disdim.csv")
    test_csv = os.path.join(data_root, "track1_test_dis.csv")

    train_df = pd.read_csv(train_csv)[["name", "discrete"]].dropna(subset=["discrete"])
    test_df = pd.read_csv(test_csv)[["name", "discrete"]].dropna(subset=["discrete"])

    # Verify valid file existence
    openface_dir = os.path.join(data_root, "openface_face")

    def valid(name):
        return os.path.exists(os.path.join(openface_dir, f"{name}.npy"))

    train_names = [n for n in train_df["name"].tolist() if valid(n)]
    test_names = [n for n in test_df["name"].tolist() if valid(n)]

    train_label_dict = {
        n: EMOTION_TO_IDX[row["discrete"]]
        for n, (_, row) in zip(train_df["name"], train_df.iterrows())
        if n in train_names and row["discrete"] in EMOTION_TO_IDX
    }
    test_label_dict = {
        n: EMOTION_TO_IDX[row["discrete"]]
        for n, (_, row) in zip(test_df["name"], test_df.iterrows())
        if n in test_names and row["discrete"] in EMOTION_TO_IDX
    }

    train_names = list(train_label_dict.keys())
    test_names = list(test_label_dict.keys())

    num_classes = len(EMOTION_TO_IDX)

    print(f"Train samples: {len(train_names)}, Test samples: {len(test_names)}, "
          f"Classes: {num_classes}")

    return train_label_dict, test_label_dict, train_names, test_names, num_classes
