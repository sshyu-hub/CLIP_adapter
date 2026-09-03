"""
Loss functions — 视频级三专家 MoE 的软分工监督。

E0 通用/布局 : 主 CE（class-weight + logit-adj，长尾修正）
E1 唤醒/动态 : arousal 4 级 CE
E2 细粒度/纹理: 混淆对难例对比 + Aux CE
外加：视频级门控负载均衡 CV²（弱权重）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_confusable_mask(labels, idx_to_emotion, confusable_pairs):
    """构造 (B, B) 布尔矩阵，mask[i, j]=True 表示 i、j 属同一易混淆对。

    覆盖 config.CONFUSABLE_PAIRS 的全部 4 对（旧版 _CONFUSABLE_GROUP_ID 只含 2 对）。
    """
    device = labels.device
    B = labels.shape[0]
    group = torch.full((len(idx_to_emotion),), -1, dtype=torch.long, device=device)
    for gid, (a, b) in enumerate(confusable_pairs):
        if a in idx_to_emotion:
            group[idx_to_emotion.index(a)] = gid
        if b in idx_to_emotion:
            group[idx_to_emotion.index(b)] = gid
    g = group[labels]                                    # (B,)
    mask = (g[:, None] == g[None, :]) & (g[:, None] != -1)
    mask = mask & ~torch.eye(B, dtype=torch.bool, device=device)
    return mask


def make_arousal_labels(labels, idx_to_emotion, arousal_map):
    """把 6 类标签映射到 arousal 4 级标签。arousal_map: {emotion: level}"""
    device = labels.device
    out = torch.zeros_like(labels)
    for emo, lvl in arousal_map.items():
        if emo in idx_to_emotion:
            out[labels == idx_to_emotion.index(emo)] = lvl
    return out


def make_valence_labels(labels, idx_to_emotion, valence_map):
    """把 6 类标签映射到 valence 3 级标签。valence_map: {emotion: level}"""
    device = labels.device
    out = torch.zeros_like(labels)
    for emo, lvl in valence_map.items():
        if emo in idx_to_emotion:
            out[labels == idx_to_emotion.index(emo)] = lvl
    return out


class DynamicClassBalancedCE(nn.Module):
    """带长尾修正的 CE：可选 class-weight + logit-adjustment（先验对数加权 τ）。"""

    def __init__(self, label_smoothing=0.1, tau=0.3):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.tau = tau
        self.log_prior = None       # (C,) 先验对数，用于 logit-adj
        self.class_weight = None    # (C,) 逆频率权重，用于 CE weight

    def set_class_counts(self, counts):
        """根据训练集每类样本数计算 class-weight 与 log-prior。"""
        counts = counts.float()
        self.log_prior = (counts / counts.sum()).log()
        w = counts.sum() / (counts * len(counts))
        self.class_weight = w / w.sum() * len(counts)

    def forward(self, logits, labels):
        if self.log_prior is not None:
            logits = logits + self.tau * self.log_prior.to(logits.device).unsqueeze(0)
        w = self.class_weight.to(logits.device) if self.class_weight is not None else None
        return F.cross_entropy(logits, labels, weight=w, label_smoothing=self.label_smoothing)


class SupervisedContrastiveLoss(nn.Module):
    """监督对比：同类拉近，异类拉远；可选对混淆对负样本加权。"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels, confusable_mask=None):
        device, B = features.device, features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=device)
        features = F.normalize(features.float(), dim=1, eps=1e-6)
        sim = torch.matmul(features, features.T) / self.temperature
        sim = sim.clamp(-10.0, 10.0)
        pos_mask = (labels[:, None] == labels[None, :]) & ~torch.eye(B, dtype=torch.bool, device=device)
        valid = pos_mask.any(dim=1)
        if not valid.any():
            return torch.tensor(0.0, device=device)
        all_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        log_weights = torch.zeros(B, B, device=device)
        if confusable_mask is not None:
            log_weights = torch.where(confusable_mask, torch.tensor(0.6931, device=device), 0.0)
        log_denom = torch.logsumexp(sim.masked_fill(~all_mask, float("-inf")) + log_weights, dim=1)
        pos_count = pos_mask.sum(dim=1).clamp(min=1)
        pos_sim_mean = (sim * pos_mask.float()).sum(dim=1) / pos_count
        loss = (log_denom - pos_sim_mean)[valid].mean()
        return loss if not torch.isnan(loss) else torch.tensor(0.0, device=device)


class HardNegativeContrastiveLoss(nn.Module):
    """E2 细粒度：对混淆对负样本加强排斥（hard negative boost）。"""

    def __init__(self, temperature=0.07, hard_boost=4.0):
        super().__init__()
        self.temperature = temperature
        self.hard_boost = hard_boost

    def forward(self, features, labels, confusable_mask=None):
        device, B = features.device, features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=device)
        features = F.normalize(features.float(), dim=1, eps=1e-6)
        sim = torch.matmul(features, features.T) / self.temperature
        sim = sim.clamp(-10.0, 10.0)
        pos_mask = (labels[:, None] == labels[None, :]) & ~torch.eye(B, dtype=torch.bool, device=device)
        valid = pos_mask.any(dim=1)
        if not valid.any():
            return torch.tensor(0.0, device=device)
        all_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        log_weights = torch.zeros(B, B, device=device)
        if confusable_mask is not None:
            log_weights = torch.where(confusable_mask, torch.tensor(float(self.hard_boost), device=device).log(), 0.0)
        log_denom = torch.logsumexp(sim.masked_fill(~all_mask, float("-inf")) + log_weights, dim=1)
        pos_count = pos_mask.sum(dim=1).clamp(min=1)
        pos_sim_mean = (sim * pos_mask.float()).sum(dim=1) / pos_count
        loss = (log_denom - pos_sim_mean)[valid].mean()
        return loss if not torch.isnan(loss) else torch.tensor(0.0, device=device)


class ExpertDiversityLoss(nn.Module):
    """专家输出正交正则（可选，防止退化）。"""

    def forward(self, expert_outputs):
        if expert_outputs is None:
            return torch.tensor(0.0, device="cpu")
        pooled = expert_outputs.mean(dim=0)         # (E, D)
        pooled = F.normalize(pooled, dim=1)         # (E, D)
        sim = torch.matmul(pooled, pooled.T)        # (E, E)
        mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        return sim[mask].pow(2).mean()


class CompositeLoss(nn.Module):
    """单级 3-expert 分工监督总损失。

    输入（对应 model.forward 的返回）：
      logits          主分类 logits
      labels          6 类标签
      expert_features [f0, f1, f2] 各专家视频级特征
      valence_logits  E1 valence 头（3 类）
      arousal_logits  E1 arousal 头（4 类）
      aux_logits      E2 Aux 头（6 类）
    """

    def __init__(
        self, num_classes=6, idx_to_emotion=None, confusable_pairs=None,
        arousal_map=None, valence_map=None, contrastive_temp=0.07,
        w_ce=1.0, w_contrastive=0.2, w_valence=0.05, w_arousal=0.05,
        w_hfcl=0.1, w_aux=0.2, w_div=0.05,
        logit_adj_tau=0.1, label_smoothing=0.1,
    ):
        super().__init__()
        self.ce_loss = DynamicClassBalancedCE(label_smoothing=label_smoothing, tau=logit_adj_tau)
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=contrastive_temp)
        self.hfcl_loss = HardNegativeContrastiveLoss(temperature=contrastive_temp)
        self.div_loss = ExpertDiversityLoss()
        self.idx_to_emotion = idx_to_emotion or ["neutral", "angry", "happy", "sad", "worried", "surprise"]
        self.confusable_pairs = confusable_pairs or [
            ("angry", "worried"), ("happy", "surprise"), ("sad", "worried"), ("neutral", "sad"),
        ]
        self.arousal_map = arousal_map or {
            "neutral": 0, "sad": 1, "worried": 2, "angry": 3, "happy": 3, "surprise": 3,
        }
        self.valence_map = valence_map or {
            "neutral": 0, "happy": 1, "surprise": 1, "angry": 2, "sad": 2, "worried": 2,
        }
        self.w_ce = w_ce
        self.w_contrastive = w_contrastive
        self.w_valence = w_valence
        self.w_arousal = w_arousal
        self.w_hfcl = w_hfcl
        self.w_aux = w_aux
        self.w_div = w_div

    def forward(self, logits, labels, expert_features=None,
                valence_logits=None, arousal_logits=None, aux_logits=None):
        losses = {}
        total = torch.tensor(0.0, device=logits.device)

        # 主 CE + Balanced Softmax
        ce = self.ce_loss(logits, labels)
        losses["ce"] = ce
        total = total + self.w_ce * ce

        # Contra：各专家分别监督对比
        if expert_features is not None and self.w_contrastive > 0:
            contra = sum(self.contrastive_loss(f, labels) for f in expert_features) / len(expert_features)
            losses["contrastive"] = contra
            total = total + self.w_contrastive * contra

        # VA：E1 → valence + arousal
        if valence_logits is not None and self.w_valence > 0:
            vl = make_valence_labels(labels, self.idx_to_emotion, self.valence_map)
            val = F.cross_entropy(valence_logits, vl)
            losses["valence"] = val
            total = total + self.w_valence * val
        if arousal_logits is not None and self.w_arousal > 0:
            al = make_arousal_labels(labels, self.idx_to_emotion, self.arousal_map)
            ar = F.cross_entropy(arousal_logits, al)
            losses["arousal"] = ar
            total = total + self.w_arousal * ar

        # HFCL：E2（f2）混淆对 hard-negative 对比
        if expert_features is not None and len(expert_features) >= 3 and self.w_hfcl > 0:
            f2 = expert_features[2]
            cmask = build_confusable_mask(labels, self.idx_to_emotion, self.confusable_pairs)
            hfcl = self.hfcl_loss(f2, labels, cmask)
            losses["hfcl"] = hfcl
            total = total + self.w_hfcl * hfcl

        # Aux CE：E2
        if aux_logits is not None and self.w_aux > 0:
            aux = F.cross_entropy(aux_logits, labels)
            losses["aux"] = aux
            total = total + self.w_aux * aux

        # Div：专家输出正交
        if expert_features is not None and self.w_div > 0:
            feats = torch.stack(expert_features, dim=1)   # (B, E, D)
            div = self.div_loss(feats)
            losses["diversity"] = div
            total = total + self.w_div * div

        losses["total"] = total
        return losses
