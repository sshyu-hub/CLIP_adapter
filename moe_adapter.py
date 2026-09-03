"""
MoE (Mixture of Experts) Adapter for CLIP Vision Transformer.

Inserted after Transformer layers 12, 18, 24 with 3 experts:
  - Expert 1: General emotion semantic expert (all emotions)
  - Expert 2: Positive-negative boundary expert
  - Expert 3: Fine-grained emotion discrimination expert

Dynamic gating network routes each token to expert combination.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertMLP(nn.Module):
    """Bottleneck MLP expert: Linear↓ → GELU → Linear↑"""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.down = nn.Linear(dim, hidden_dim)
        self.up = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()

        # Initialize with small values for stable residual connection
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


class GatingNetwork(nn.Module):
    """Per-sample gating: gate from CLS token → broadcast to all tokens.

    Using the CLS token (which aggregates emotion signal) to decide per-sample
    expert weights.  All tokens in a sample share the same gate — the gate now
    reflects "what emotion is this video" rather than "what is this token".
    """

    def __init__(self, dim: int, num_experts: int = 3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, num_experts),
        )
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) → gate from CLS (pos 0) → broadcast → (B, N, E)"""
        cls_token = x[:, 0, :]                                  # (B, D)
        weights = F.softmax(self.gate(cls_token), dim=-1)       # (B, E)
        return weights.unsqueeze(1).expand(-1, x.shape[1], -1)  # (B, N, E)


class MoEAdapter(nn.Module):
    """Mixture-of-Experts Adapter with sample-level emotion-aware gating.

    Gate uses CLS token → per-sample expert weights → all tokens in the
    sample share the same expert mix.  CE gradients on the CLS token's
    final representation provide emotion supervision to the gate.
    """

    def __init__(self, dim: int, hidden_dim: int, num_experts: int = 3, expert_dims=None):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts

        # Per-expert bottleneck dims — structural differentiation.
        # expert_dims: list of length num_experts, or None → all = hidden_dim.
        if expert_dims is None:
            expert_dims = [hidden_dim] * num_experts
        self.experts = nn.ModuleList([
            ExpertMLP(dim, expert_dims[i]) for i in range(num_experts)
        ])
        self.gate = GatingNetwork(dim, num_experts)

    def forward(self, x: torch.Tensor):
        gates = self.gate(x)                                    # (B, N, E)

        expert_outs = []
        for expert in self.experts:
            expert_outs.append(expert(x))                       # each (B, N, D)
        expert_outputs = torch.stack(expert_outs, dim=-2)       # (B, N, E, D)

        adapted = (gates.unsqueeze(-1) * expert_outputs).sum(dim=-2)  # (B, N, D) 纯增量

        return adapted, gates, expert_outputs


class CLIPMoEHook(nn.Module):
    """Manages forward hooks for injecting MoE adapters into CLIP ViT layers.

    Must be an nn.Module so DataParallel can replicate internal references correctly.

    Usage:
        hook_mgr = CLIPMoEHook()
        hook_mgr.register(vision_model, moe_adapters)
        # ... forward pass ...
        outputs = hook_mgr.collect()
        hook_mgr.remove()
    """

    def __init__(self):
        super().__init__()
        self._hooks: list = []
        self._collected: dict = {}

    def _make_hook(self, layer_name: str, moe_adapters):
        def hook(module, input, output):
            adapter = moe_adapters[layer_name]
            hidden_states = output[0]
            adapted, gates, expert_outs = adapter(hidden_states)
            self._collected[layer_name] = {
                "gates": gates,
                "expert_outputs": expert_outs,
            }
            # 残差在 hook 里做：hidden_states + Adapter(x)
            return (hidden_states + adapted,) + output[1:]
        return hook

    def register(self, vision_model: nn.Module, moe_adapters: nn.ModuleDict):
        """Register hooks on vision_model layers. Call before each forward pass."""
        for layer_name in moe_adapters.keys():
            layer_idx = int(layer_name)
            layer = vision_model.encoder.layers[layer_idx]
            h = layer.register_forward_hook(self._make_hook(layer_name, moe_adapters))
            self._hooks.append(h)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def collect(self, clear: bool = True):
        """Return collected gates and expert outputs."""
        result = dict(self._collected)
        if clear:
            self._collected.clear()
        return result
