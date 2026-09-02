"""
CLIP ViT + MoE with frequency-aware expert.

专家分工：
  E0 (通用): per-token MoE, 无专属 loss, CE+Contra
  E1 (VA):   per-token MoE, VA loss (效价/唤醒度)
  E3 (高频): 独立分支, patch max-pooling 高频纹理, HFCL + Aux CE

低频 = CLS token mean pooling (整体布局/形状)
高频 = patch tokens max pooling (纹理/边缘细节)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import CLIPModel

from moe_adapter import MoEAdapter, CLIPMoEHook, ExpertMLP


class CLIPMoEEmotionModel(nn.Module):

    def __init__(
        self, clip_model_path, adapter_layers=None, num_classes=6,
        num_experts=2, adapter_bottleneck=128, freeze_backbone=True,
        expert_dims=None, hf_bottleneck=64,
    ):
        super().__init__()
        if adapter_layers is None:
            adapter_layers = [13, 21, 23]

        clip = CLIPModel.from_pretrained(clip_model_path, local_files_only=True)
        self.vision_model = clip.vision_model
        self.hidden_dim = clip.config.vision_config.hidden_size
        del clip

        if freeze_backbone:
            for param in self.vision_model.parameters():
                param.requires_grad = False

        self.adapter_layers = adapter_layers
        # per-token MoE: 2 experts (E0 通用 + E1 VA)
        self.moe_adapters = nn.ModuleDict({
            str(idx): MoEAdapter(self.hidden_dim, adapter_bottleneck, num_experts, expert_dims)
            for idx in adapter_layers
        })
        self.hook_mgr = CLIPMoEHook()

        # 主分类器: CLS mean (低频)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, num_classes),
        )

        # 高频专家分支: patch max pooling
        self.hf_expert = ExpertMLP(self.hidden_dim, hf_bottleneck)
        self.hf_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, num_classes),
        )

        self.num_classes = num_classes
        self.num_experts = num_experts

    def _forward_one_frame(self, frame):
        vision_outputs = self.vision_model(pixel_values=frame)
        # CLS (低频全局) + patch tokens (高频纹理)
        cls_token = vision_outputs.last_hidden_state[:, 0, :]       # (B, D)
        patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]    # (B, N-1, D)
        hf_feat = patch_tokens.max(dim=1).values                     # (B, D) 高频
        collected = self.hook_mgr.collect(clear=False)
        return cls_token, hf_feat, collected

    def forward(self, pixel_values):
        B, T, C, H, W = pixel_values.shape

        self.hook_mgr.register(self.vision_model, self.moe_adapters)
        all_cls, all_hf, all_collected = [], [], []
        for t in range(T):
            cls_token, hf_feat, collected_t = self._forward_one_frame(pixel_values[:, t])
            all_cls.append(cls_token)
            all_hf.append(hf_feat)
            all_collected.append(collected_t)
        self.hook_mgr.collect(clear=True)
        self.hook_mgr.remove()

        # 低频主特征: CLS mean over frames
        features = torch.stack(all_cls, dim=1).mean(dim=1)          # (B, D)
        logits = self.classifier(features)

        # 高频分支: patch max mean over frames → 高频专家 → Aux head
        hf = torch.stack(all_hf, dim=1).mean(dim=1)                 # (B, D)
        hf_emb = self.hf_expert(hf)                                 # (B, D) 高频 embedding
        aux_logits = self.hf_head(hf_emb)                           # (B, num_classes)

        collected = self._pool_collected(all_collected, B, T)

        return {
            "logits": logits,
            "features": features,
            "collected": collected,
            "fine_features": hf_emb,        # 给 HFCL
            "aux_logits": aux_logits,       # 给 Aux CE
        }

    def _pool_collected(self, all_collected, B, T):
        if not all_collected:
            return {}
        result = {}
        for layer_name in all_collected[0].keys():
            pooled = {}
            for key in ["gates", "expert_outputs"]:
                stacked = torch.stack([f[layer_name][key] for f in all_collected], dim=0)
                stacked = stacked.permute(1, 0, *range(2, stacked.dim()))
                cls_pooled = stacked[:, :, 0, ...]
                pooled[key] = cls_pooled.mean(dim=1)
            result[layer_name] = pooled
        return result

    def extract_features(self, pixel_values):
        B, T, C, H, W = pixel_values.shape
        self.hook_mgr.register(self.vision_model, self.moe_adapters)
        all_cls = []
        for t in range(T):
            cls_token, _, _ = self._forward_one_frame(pixel_values[:, t])
            all_cls.append(cls_token)
        self.hook_mgr.collect(clear=True)
        self.hook_mgr.remove()
        return torch.stack(all_cls, dim=1).mean(dim=1)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.vision_model.eval()
        return self


def create_model(config=None):
    if config is None:
        from config import CLIP_MODEL_PATH, ADAPTER_LAYERS, NUM_CLASSES, NUM_EXPERTS, ADAPTER_BOTTLENECK, EXPERT_DIMS, HF_BOTTLENECK
        return CLIPMoEEmotionModel(
            clip_model_path=CLIP_MODEL_PATH, adapter_layers=ADAPTER_LAYERS,
            num_classes=NUM_CLASSES, num_experts=NUM_EXPERTS,
            adapter_bottleneck=ADAPTER_BOTTLENECK, expert_dims=EXPERT_DIMS,
            hf_bottleneck=HF_BOTTLENECK,
        )
    return CLIPMoEEmotionModel(
        clip_model_path=config.CLIP_MODEL_PATH, adapter_layers=config.ADAPTER_LAYERS,
        num_classes=config.NUM_CLASSES, num_experts=config.NUM_EXPERTS,
        adapter_bottleneck=config.ADAPTER_BOTTLENECK, expert_dims=config.EXPERT_DIMS,
        hf_bottleneck=config.HF_BOTTLENECK,
    )
