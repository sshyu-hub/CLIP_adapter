"""
CLIP ViT-L/14 + 单级 3-expert MoE Adapter。

架构（单级 MoE，替换原两级方案）：
  - 冻结 CLIP 主干，仅解冻第 13/21/23 层 LayerNorm 做 LN tuning（lr=1e-5）
  - token 级 MoEAdapter：注入第 13/21/23 层（0-indexed），每层 3 个专家
    bottleneck=128，Gate 由 CLS token → MLP(1024→256→3) → 广播全 token
  - 融合 token = x + Σ gᵢ·Eᵢ(x)（残差注入主干）
  - 视频级：cls_pooled（CLS 帧间 mean）+ 门控加权专家增量 → 主分类
  - 分工监督：E1 → valence(3)+arousal(4)，E2 → aux(6) + HFCL，E0 无专属
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

from moe_adapter import MoEAdapter, CLIPMoEHook


class CLIPMoEEmotionModel(nn.Module):

    def __init__(
        self, clip_model_path, adapter_layers=None, num_classes=6,
        num_experts=3, adapter_bottleneck=128, freeze_backbone=True,
        expert_dims=None, ln_tuning_layers=None,
        num_arousal=4, num_valence=3,
    ):
        super().__init__()
        if adapter_layers is None:
            adapter_layers = [13, 21, 23]
        if ln_tuning_layers is None:
            ln_tuning_layers = adapter_layers

        clip = CLIPModel.from_pretrained(clip_model_path, local_files_only=True)
        self.vision_model = clip.vision_model
        self.hidden_dim = clip.config.vision_config.hidden_size
        del clip

        # 冻结主干，再解冻指定层 LayerNorm（LN tuning）
        if freeze_backbone:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        self.ln_tuning_layers = ln_tuning_layers
        for idx in ln_tuning_layers:
            layer = self.vision_model.encoder.layers[idx]
            for name, param in layer.named_parameters():
                if "layer_norm" in name:
                    param.requires_grad = True

        # ── token 级 MoEAdapter（3 层 × num_experts 专家）────────
        self.adapter_layers = adapter_layers
        self.num_experts = num_experts
        self.moe_adapters = nn.ModuleDict({
            str(idx): MoEAdapter(self.hidden_dim, adapter_bottleneck, num_experts, expert_dims)
            for idx in adapter_layers
        })
        self.hook_mgr = CLIPMoEHook()

        # ── 主分类头 ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, num_classes),
        )
        # ── 视频级门控 g = softmax(MLP(cls_pooled)) → (B, E) ────
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 4, num_experts),
        )
        # ── 分工监督头 ──────────────────────────────────────────
        self.valence_head = nn.Linear(self.hidden_dim, num_valence)   # E1 → 3 类
        self.arousal_head = nn.Linear(self.hidden_dim, num_arousal)   # E1 → 4 类
        self.aux_head = nn.Sequential(                                # E2 → 6 类
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, num_classes),
        )

        self.num_classes = num_classes

    def _forward_one_frame(self, frame):
        vision_outputs = self.vision_model(pixel_values=frame)
        cls_token = vision_outputs.last_hidden_state[:, 0, :]       # (B, D)
        collected = self.hook_mgr.collect(clear=False)
        # 立即降维：expert_outputs (B, N, E, D) → mean over tokens → (B, E, D)
        pooled = {
            layer_name: data["expert_outputs"].mean(dim=1)
            for layer_name, data in collected.items()
        }
        return cls_token, pooled

    def forward(self, pixel_values):
        B, T, C, H, W = pixel_values.shape

        self.hook_mgr.register(self.vision_model, self.moe_adapters)
        all_cls, all_pooled = [], []
        for t in range(T):
            cls_token, pooled = self._forward_one_frame(pixel_values[:, t])
            all_cls.append(cls_token)
            all_pooled.append(pooled)
        self.hook_mgr.remove()
        self.hook_mgr.collect(clear=True)

        cls_seq = torch.stack(all_cls, dim=1)          # (B, T, D)
        cls_pooled = cls_seq.mean(dim=1)               # (B, D) 主信号

        # 每专家视频级特征 f0/f1/f2 = mean over (tokens, frames, layers)
        f0, f1, f2 = self._pool_expert_features(all_pooled)

        # 视频级门控 + 融合
        g = F.softmax(self.gate(cls_pooled), dim=-1)   # (B, E)
        g0, g1, g2 = g[:, 0:1], g[:, 1:2], g[:, 2:3]
        fused = cls_pooled + g0 * f0 + g1 * f1 + g2 * f2
        logits = self.classifier(fused)                # (B, C)

        # 分工监督支路
        valence_logits = self.valence_head(f1)         # (B, 3)
        arousal_logits = self.arousal_head(f1)         # (B, 4)
        aux_logits = self.aux_head(f2)                 # (B, C)

        return {
            "logits": logits,               # 主分类 logits
            "features": fused,              # 融合视频级特征
            "expert_features": [f0, f1, f2],  # 各专家特征（Contra/Div/HFCL 用）
            "valence_logits": valence_logits,
            "arousal_logits": arousal_logits,
            "aux_logits": aux_logits,
        }

    def _pool_expert_features(self, all_pooled):
        """all_pooled: list[T] of {layer_name: (B, E, D)} → 返回 (f0, f1, f2) 各 (B, D)。"""
        num_experts = self.num_experts
        acc = [None] * num_experts
        count = 0
        for pooled in all_pooled:
            for expert_out in pooled.values():        # (B, E, D)
                if acc[0] is None:
                    acc = [expert_out[:, e, :].clone() for e in range(num_experts)]
                else:
                    for e in range(num_experts):
                        acc[e] = acc[e] + expert_out[:, e, :]
                count += 1
        return (acc[0] / count, acc[1] / count, acc[2] / count)

    def extract_features(self, pixel_values):
        # 与 forward 一致：融合特征作为视频级表示
        return self.forward(pixel_values)["features"]

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.vision_model.eval()
        return self


def create_model(config=None):
    if config is None:
        import config as c
        return CLIPMoEEmotionModel(
            clip_model_path=c.CLIP_MODEL_PATH, adapter_layers=c.ADAPTER_LAYERS,
            num_classes=c.NUM_CLASSES, num_experts=c.NUM_EXPERTS,
            adapter_bottleneck=c.ADAPTER_BOTTLENECK, expert_dims=c.EXPERT_DIMS,
            ln_tuning_layers=c.LN_TUNING_LAYERS,
            num_arousal=c.NUM_AROUSAL, num_valence=c.NUM_VALENCE,
        )
    return CLIPMoEEmotionModel(
        clip_model_path=config.CLIP_MODEL_PATH, adapter_layers=config.ADAPTER_LAYERS,
        num_classes=config.NUM_CLASSES, num_experts=config.NUM_EXPERTS,
        adapter_bottleneck=config.ADAPTER_BOTTLENECK,
        expert_dims=getattr(config, "EXPERT_DIMS", [128, 128, 128]),
        ln_tuning_layers=getattr(config, "LN_TUNING_LAYERS", [13, 21, 23]),
        num_arousal=getattr(config, "NUM_AROUSAL", 4),
        num_valence=getattr(config, "NUM_VALENCE", 3),
    )
