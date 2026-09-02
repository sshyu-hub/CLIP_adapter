"""
Loss functions — 3-expert MoE with soft specialization.

E1: CE + Global Contra → general semantics (scene/object)
E2: VA loss → face expression (valence-arousal)
E3: HFCL → ambiguous emotions (confusable pair contrastive)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_CONFUSABLE_GROUP_ID = {
    "angry": 0, "worried": 0,
    "happy": 1, "surprise": 1,
    # sad & neutral: no confusable pair — distinct enough
}

def get_confusable_group(labels, idx_to_emotion):
    mapping = {}
    for i, emo in enumerate(idx_to_emotion):
        mapping[i] = _CONFUSABLE_GROUP_ID.get(emo, -1)
    return torch.tensor([mapping[l.item()] for l in labels], device=labels.device)


class DynamicClassBalancedCE(nn.Module):
    def __init__(self, label_smoothing=0.1, tau=0.1):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.tau = tau
        self.log_prior = None

    def set_class_counts(self, counts):
        lp = (counts.float() / counts.float().sum()).log()
        self._buffers.pop("log_prior", None)
        self.__dict__.pop("log_prior", None)
        self.register_buffer("log_prior", lp)

    def forward(self, logits, labels, epoch=None):
        if self.log_prior is not None:
            logits = logits + self.tau * self.log_prior.unsqueeze(0)
        return F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels, confusable_groups=None):
        device, B = features.device, features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=device)
        features = F.normalize(features.float(), dim=1, eps=1e-6)
        sim = torch.matmul(features, features.T) / self.temperature
        sim = sim.clamp(-10.0, 10.0)
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask = pos_mask & ~torch.eye(B, dtype=torch.bool, device=device)
        valid = pos_mask.any(dim=1)
        if not valid.any():
            return torch.tensor(0.0, device=device)
        all_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        log_weights = torch.zeros(B, B, device=device)
        if confusable_groups is not None:
            neg_mask = ~(labels.unsqueeze(0) == labels.unsqueeze(1))
            same = confusable_groups.unsqueeze(0) == confusable_groups.unsqueeze(1)
            log_weights = torch.where(same & neg_mask, torch.tensor(0.6931, device=device), 0.0)
        log_denom = torch.logsumexp(sim.masked_fill(~all_mask, float("-inf")) + log_weights, dim=1)
        pos_count = pos_mask.sum(dim=1).clamp(min=1)
        pos_sim_mean = (sim * pos_mask.float()).sum(dim=1) / pos_count
        loss = (log_denom - pos_sim_mean)[valid].mean()
        return loss if not torch.isnan(loss) else torch.tensor(0.0, device=device)


class HardNegativeContrastiveLoss(nn.Module):
    """E3: confusion-guided soft-weighted contrastive.

    Uses confusion matrix to weight negatives — classes that are
    frequently confused get stronger repulsion.
    """
    def __init__(self, temperature=0.07, base_boost=2.0, max_boost=8.0):
        super().__init__()
        self.temperature = temperature
        self.base_boost = base_boost
        self.max_boost = max_boost
        self.confusion_matrix = None  # (C, C), updated from validation

    def update_confusion(self, cm: "torch.Tensor"):
        """Set confusion matrix from validation. cm[C, C]: cm[i,j] = #times i confused as j"""
        self._buffers.pop("confusion_matrix", None)
        self.__dict__.pop("confusion_matrix", None)
        self.register_buffer("confusion_matrix", cm.float())

    def forward(self, features, labels, confusable_groups=None):
        device, B = features.device, features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=device)
        features = F.normalize(features.float(), dim=1, eps=1e-6)
        sim = torch.matmul(features, features.T) / self.temperature
        sim = sim.clamp(-10.0, 10.0)
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask = pos_mask & ~torch.eye(B, dtype=torch.bool, device=device)
        valid = pos_mask.any(dim=1)
        if not valid.any():
            return torch.tensor(0.0, device=device)
        all_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        neg_mask = ~(labels.unsqueeze(0) == labels.unsqueeze(1))

        # Confusion-based or group-based weighting
        log_weights = torch.zeros(B, B, device=device)
        if self.confusion_matrix is not None:
            # Normalize confusion matrix per row → (C, C)
            cm_norm = self.confusion_matrix / (self.confusion_matrix.sum(dim=1, keepdim=True) + 1e-8)
            for i in range(B):
                li = labels[i].item()
                for j in range(B):
                    if neg_mask[i, j]:
                        lj = labels[j].item()
                        cf = cm_norm[li, lj].item()
                        boost = self.base_boost + cf * (self.max_boost - self.base_boost)
                        log_weights[i, j] = torch.tensor(boost, device=device).log()
        elif confusable_groups is not None:
            same = confusable_groups.unsqueeze(0) == confusable_groups.unsqueeze(1)
            hard_log = torch.tensor(self.base_boost * 2, device=device).log()
            log_weights = torch.where(same & neg_mask, hard_log, 0.0)
        else:
            # Uniform: no hard negative emphasis
            pass

        log_denom = torch.logsumexp(sim.masked_fill(~all_mask, float("-inf")) + log_weights, dim=1)
        pos_count = pos_mask.sum(dim=1).clamp(min=1)
        pos_sim_mean = (sim * pos_mask.float()).sum(dim=1) / pos_count
        loss = (log_denom - pos_sim_mean)[valid].mean()
        return loss if not torch.isnan(loss) else torch.tensor(0.0, device=device)


class VALoss(nn.Module):
    def __init__(self, input_dim=1024, num_valence=3, num_arousal=2):
        super().__init__()
        self.valence_head = nn.Linear(input_dim, num_valence)
        self.arousal_head = nn.Linear(input_dim, num_arousal)

    def forward(self, e2_feat, valence_labels, arousal_labels):
        return (F.cross_entropy(self.valence_head(e2_feat), valence_labels),
                F.cross_entropy(self.arousal_head(e2_feat), arousal_labels))


class ExpertDiversityLoss(nn.Module):
    def forward(self, expert_outputs):
        if expert_outputs is None:
            return torch.tensor(0.0, device="cpu")
        pooled = expert_outputs.mean(dim=0)         # (E, D)
        pooled = F.normalize(pooled, dim=1)         # (E, D)
        sim = torch.matmul(pooled, pooled.T)        # (E, E)
        mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        return sim[mask].pow(2).mean()


class CompositeLoss(nn.Module):
    def __init__(
        self, num_classes=6, contrastive_temp=0.07, idx_to_emotion=None,
        w_ce=1.0, w_contrastive=0.2, w_boundary=0.1,
        w_fine_grained=0.1, w_expert3_aux=0.2, w_diversity=0.05, w_gate_entropy=0.005,
    ):
        super().__init__()
        self.ce_loss = DynamicClassBalancedCE()
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=contrastive_temp)
        self.fine_grained_loss = HardNegativeContrastiveLoss(temperature=contrastive_temp)
        self.va_loss = VALoss()
        self.diversity_loss = ExpertDiversityLoss()
        self.idx_to_emotion = idx_to_emotion or ["neutral","angry","happy","sad","worried","surprise"]
        self.w_ce = w_ce
        self.w_contrastive = w_contrastive
        self.w_boundary = w_boundary
        self.w_fine_grained = w_fine_grained
        self.w_expert3_aux = w_expert3_aux
        self.w_diversity = w_diversity
        self.w_gate_entropy = w_gate_entropy

    def forward(self, logits, labels, features, collected=None, fine_features=None, aux_logits=None, epoch=None):
        losses = {}
        total = torch.tensor(0.0, device=logits.device)

        ce = self.ce_loss(logits, labels, epoch=epoch)
        losses["ce"] = ce
        total = total + self.w_ce * ce

        confusable_groups = get_confusable_group(labels, self.idx_to_emotion)
        contra = self.contrastive_loss(features, labels, confusable_groups)
        losses["contrastive"] = contra
        total = total + self.w_contrastive * contra

        # E3 HFCL
        if fine_features is not None:
            fine = self.fine_grained_loss(fine_features, labels, confusable_groups)
            losses["fine_grained"] = fine
            total = total + self.w_fine_grained * fine

        # E3 Aux CE
        if aux_logits is not None:
            aux_ce = self.ce_loss(aux_logits, labels)
            losses["expert3_aux"] = aux_ce
            total = total + self.w_expert3_aux * aux_ce

        if collected:
            e2_feats, all_eo, all_gates = [], [], []
            for layer_name in sorted(collected.keys(), key=int):
                eo = collected[layer_name]["expert_outputs"]
                g = collected[layer_name]["gates"]       # (B, E)
                all_eo.append(eo)
                all_gates.append(g)
                e2_feats.append(eo[:, 1, :])
            e2 = torch.stack(e2_feats, dim=0).mean(dim=0)

            # E2 VA loss
            valence_labels = labels.clone()
            for emo, val in [("neutral",0),("happy",1),("surprise",1),("angry",2),("sad",2),("worried",2)]:
                valence_labels[labels == self.idx_to_emotion.index(emo)] = val
            arousal_labels = labels.clone()
            for emo, ar in [("neutral",0),("sad",0),("angry",1),("happy",1),("surprise",1),("worried",1)]:
                arousal_labels[labels == self.idx_to_emotion.index(emo)] = ar
            v_loss, a_loss = self.va_loss(e2, valence_labels, arousal_labels)
            losses["va"] = v_loss + a_loss
            total = total + self.w_boundary * (v_loss + a_loss)

            # Gate load balance (CV² of per-expert average weight)
            gates = all_gates[-1]                          # (B, E) 最后一层
            f = gates.mean(dim=0)                          # (E,) 每专家 batch 平均权重
            cv2 = (f.std() / f.mean().clamp(min=1e-8)).pow(2)
            losses["gate_balance"] = cv2
            total = total + self.w_gate_entropy * cv2

            # Diversity loss (专家输出正交)
            div = self.diversity_loss(all_eo[-1])
            losses["diversity"] = div
            total = total + self.w_diversity * div

        losses["total"] = total
        return losses
