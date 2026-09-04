"""
CLIP ViT + MoE：token 级残差 MoEAdapter（保留）+ 视频级三专家分工（新方案）。

两级 MoE：
1. token 级（moe_adapter.MoEAdapter）：注入第 13/21/23 层，CLS 门控广播到全 token，
   2 个异构专家（bottleneck 256/128）做残差增强——保主干、与 Exp2 可比，此部分沿用原架构。
2. 视频级三专家（作用于 last_hidden_state，按"失败层 × 正交维度"分工）：
   - E0 通用/布局：CLS 帧间 mean，ExpertMLP(256)，主 CE
   - E1 唤醒/动态：CLS temporal attention，ExpertMLP(128)，arousal 4 级 CE
   - E2 细粒度/纹理：patch − 序列均值 → max → top-k，ExpertMLP(64)，混淆对对比 + Aux CE
   视频级门控 g = softmax(CLS pooled) → 融合：
     fused = cls_pooled + Σ g_i · f_i                     （特征残差融合）
     logits = classifier(fused) + α · Σ g_i · head_i(f_i) （logit 融合）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

from moe_adapter import MoEAdapter, CLIPMoEHook, ExpertMLP


class TemporalAttentionPool(nn.Module):
    """可学习 query 对 T 帧做打分加权：聚焦表情峰值帧。"""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.scale = dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) → (B, D)
        scores = torch.einsum("btd,d->bt", x, self.query) * self.scale
        attn = F.softmax(scores, dim=1)
        return (attn.unsqueeze(-1) * x).sum(dim=1)


class TopKPool(nn.Module):
    """取特征范数最强的 k 帧平均，避免中性帧抹平纹理峰值。"""

    def __init__(self, k: int = 5):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) → (B, D)
        norms = x.norm(dim=-1)                          # (B, T)
        k = min(self.k, x.shape[1])
        idx = norms.topk(k, dim=1).indices              # (B, k)
        gathered = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        return gathered.mean(dim=1)


class CLIPMoEEmotionModel(nn.Module):

    def __init__(
        self, clip_model_path, adapter_layers=None, num_classes=6,
        num_experts=2, adapter_bottleneck=128, freeze_backbone=True,
        expert_dims=None,
        video_expert_dims=None, top_k=5,
        fusion_feature=True, fusion_logit=True, fusion_alpha=0.5,
        num_arousal=4, view_decouple=True,
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

        # ── token 级残差 MoEAdapter（保留原架构）────────────────
        self.adapter_layers = adapter_layers
        self.moe_adapters = nn.ModuleDict({
            str(idx): MoEAdapter(self.hidden_dim, adapter_bottleneck, num_experts, expert_dims)
            for idx in adapter_layers
        })
        self.hook_mgr = CLIPMoEHook()

        # ── 视频级三专家 ───────────────────────────────────────
        if video_expert_dims is None:
            video_expert_dims = [256, 128, 64]
        self.video_expert_dims = video_expert_dims
        self.view_decouple = view_decouple
        self.fusion_feature = fusion_feature
        self.fusion_logit = fusion_logit
        self.fusion_alpha = fusion_alpha

        # E0: CLS mean
        self.expert0 = ExpertMLP(self.hidden_dim, video_expert_dims[0])
        # E1: temporal attention（若不解耦则退化为 mean）
        self.attn_pool = TemporalAttentionPool(self.hidden_dim)
        self.expert1 = ExpertMLP(self.hidden_dim, video_expert_dims[1])
        # E2: patch − 均值 → max → top-k（若不解耦则退化为 CLS mean）
        self.topk_pool = TopKPool(top_k)
        self.expert2 = ExpertMLP(self.hidden_dim, video_expert_dims[2])

        # 视频级门控：CLS pooled → 3
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 4, 3),
        )

        # 共享分类头（主 logits）
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, num_classes),
        )
        # 各专家独立 6 类头（logit 融合用）
        self.expert_head0 = nn.Linear(self.hidden_dim, num_classes)
        self.expert_head1 = nn.Linear(self.hidden_dim, num_classes)
        self.expert_head2 = nn.Linear(self.hidden_dim, num_classes)
        # E1 arousal 头（4 级）
        self.arousal_head = nn.Linear(self.hidden_dim, num_arousal)
        # E2 Aux 头（6 类）
        self.aux_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, num_classes),
        )

        self.num_classes = num_classes
        self.num_experts = num_experts

    def _forward_one_frame(self, frame):
        vision_outputs = self.vision_model(pixel_values=frame)
        cls_token = vision_outputs.last_hidden_state[:, 0, :]       # (B, D)
        patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]   # (B, N, D)
        collected = self.hook_mgr.collect(clear=False)
        return cls_token, patch_tokens, collected

    def forward(self, pixel_values):
        B, T, C, H, W = pixel_values.shape

        self.hook_mgr.register(self.vision_model, self.moe_adapters)
        all_cls, all_patch, all_collected = [], [], []
        for t in range(T):
            cls_token, patch_tokens, collected_t = self._forward_one_frame(pixel_values[:, t])
            all_cls.append(cls_token)
            all_patch.append(patch_tokens)
            all_collected.append(collected_t)
        self.hook_mgr.collect(clear=True)
        self.hook_mgr.remove()

        cls_seq = torch.stack(all_cls, dim=1)          # (B, T, D)
        patch_seq = torch.stack(all_patch, dim=1)      # (B, T, N, D)
        cls_pooled = cls_seq.mean(dim=1)               # (B, D) 原始 CLS 均值（主信号）

        # ── E0：CLS mean ──────────────────────────────────────
        f0 = self.expert0(cls_pooled)                  # (B, D)

        # ── E1：temporal attention（或退化为 mean）────────────
        e1_vec = self.attn_pool(cls_seq) if self.view_decouple else cls_pooled
        f1 = self.expert1(e1_vec)                      # (B, D)

        # ── E2：patch − 均值 → max → top-k（或退化为 CLS mean）─
        if self.view_decouple:
            patch_centered = patch_seq - patch_seq.mean(dim=2, keepdim=True)
            v2_seq = patch_centered.max(dim=2).values   # (B, T, D)
            e2_vec = self.topk_pool(v2_seq)
        else:
            e2_vec = cls_pooled
        f2 = self.expert2(e2_vec)                      # (B, D)

        # ── 视频级门控 + 融合 ────────────────────────────────
        g = F.softmax(self.gate(cls_pooled), dim=-1)   # (B, 3)
        g0, g1, g2 = g[:, 0:1], g[:, 1:2], g[:, 2:3]

        if self.fusion_feature:
            fused = cls_pooled + g0 * f0 + g1 * f1 + g2 * f2
        else:
            fused = cls_pooled
        logits_main = self.classifier(fused)           # (B, C)

        ensemble_logits = (
            g0 * self.expert_head0(f0)
            + g1 * self.expert_head1(f1)
            + g2 * self.expert_head2(f2)
        )
        logits = logits_main + self.fusion_alpha * ensemble_logits if self.fusion_logit else logits_main

        # 专属监督支路
        arousal_logits = self.arousal_head(f1)         # (B, 4)
        aux_logits = self.aux_head(f2)                 # (B, C)

        return {
            "logits": logits,            # 最终 logits（主 CE）
            "logits_main": logits_main,
            "features": fused,           # 融合特征（extract_features 用）
            "fine_features": f2,         # E2 专家特征（混淆对对比）
            "aux_logits": aux_logits,    # E2 Aux CE
            "arousal_logits": arousal_logits,  # E1 arousal CE
            "gates_video": g,            # (B, 3) 负载均衡
            "collected": self._pool_collected(all_collected, B, T),
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
            video_expert_dims=c.VIDEO_EXPERT_DIMS, top_k=c.TOP_K,
            fusion_feature=c.FUSION_FEATURE, fusion_logit=c.FUSION_LOGIT,
            fusion_alpha=c.FUSION_ALPHA, num_arousal=c.NUM_AROUSAL,
        )
    return CLIPMoEEmotionModel(
        clip_model_path=config.CLIP_MODEL_PATH, adapter_layers=config.ADAPTER_LAYERS,
        num_classes=config.NUM_CLASSES, num_experts=config.NUM_EXPERTS,
        adapter_bottleneck=config.ADAPTER_BOTTLENECK, expert_dims=config.EXPERT_DIMS,
        video_expert_dims=getattr(config, "VIDEO_EXPERT_DIMS", [256, 128, 64]),
        top_k=getattr(config, "TOP_K", 5),
        fusion_feature=getattr(config, "FUSION_FEATURE", True),
        fusion_logit=getattr(config, "FUSION_LOGIT", True),
        fusion_alpha=getattr(config, "FUSION_ALPHA", 0.5),
        num_arousal=getattr(config, "NUM_AROUSAL", 4),
    )
