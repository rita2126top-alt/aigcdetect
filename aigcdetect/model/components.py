from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F


class LengthRouter(nn.Module):
    def __init__(self, image_dim: int, candidates=(1, 3, 5, 7), hidden_dim: int = 512):
        super().__init__()
        self.candidates = tuple(int(x) for x in candidates)
        self.net = nn.Sequential(nn.Linear(image_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, len(self.candidates)))

    def forward(self, image_feature: torch.Tensor, training: bool, temperature: float = 1.0):
        logits = self.net(image_feature.detach())
        probs = logits.softmax(dim=-1)
        if training:
            hard = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
        else:
            idx = probs.argmax(dim=-1)
            hard = F.one_hot(idx, num_classes=len(self.candidates)).to(probs.dtype)
        return logits, probs, hard

    def expected_length_loss(self, probs: torch.Tensor, max_private: int):
        lengths = probs.new_tensor(self.candidates)
        return (probs * lengths).sum(dim=-1).mean() / float(max_private)


class TokenCrossAttention(nn.Module):
    def __init__(self, text_dim: int, visual_dim: int, attn_dim: int = 256,
                 heads: int = 4, dropout: float = 0.1, ffn_ratio: int = 4):
        super().__init__()
        assert attn_dim % heads == 0
        self.q_proj = nn.Linear(text_dim, attn_dim)
        self.k_proj = nn.Linear(visual_dim, attn_dim)
        self.v_proj = nn.Linear(visual_dim, attn_dim)
        self.q_ln = nn.LayerNorm(attn_dim)
        self.k_ln = nn.LayerNorm(attn_dim)
        self.v_ln = nn.LayerNorm(attn_dim)
        self.attn = nn.MultiheadAttention(attn_dim, heads, dropout=dropout, batch_first=True)
        self.ffn_ln = nn.LayerNorm(attn_dim)
        self.ffn = nn.Sequential(
            nn.Linear(attn_dim, attn_dim * ffn_ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(attn_dim * ffn_ratio, attn_dim), nn.Dropout(dropout)
        )

    def forward(self, text_tokens: torch.Tensor, visual_tokens: torch.Tensor):
        q = self.q_proj(text_tokens)
        k = self.k_proj(visual_tokens)
        v = self.v_proj(visual_tokens)
        out, attn = self.attn(self.q_ln(q), self.k_ln(k), self.v_ln(v), need_weights=False)
        x = q + out
        x = x + self.ffn(self.ffn_ln(x))
        return x


class InteractivePrototypeFusion(nn.Module):
    def __init__(self, attn_dim: int, joint_dim: int, alpha_max: float = 0.5, alpha_init: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(attn_dim, joint_dim)
        self.alpha_max = float(alpha_max)
        ratio = min(max(alpha_init / alpha_max, 1e-6), 1 - 1e-6)
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32))

    @property
    def alpha(self):
        return self.alpha_max * torch.sigmoid(self.alpha_logit)

    def forward(self, eos_feature: torch.Tensor, interaction_tokens: torch.Tensor):
        interaction = self.proj(interaction_tokens.mean(dim=-2))
        interaction = F.normalize(interaction, dim=-1)
        eos_feature = F.normalize(eos_feature, dim=-1)
        return F.normalize(eos_feature + self.alpha.to(eos_feature.dtype) * interaction, dim=-1)
