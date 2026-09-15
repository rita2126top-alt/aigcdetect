"""Executable implementation of the supplied method, without editing PPM-CLIP."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .backbone import Backbone
from .probabilistic import ProbabilisticContext, mlp


class TokenCrossAttention(nn.Module):
    def __init__(self, text_dim, visual_dim, dim, heads, dropout):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.q = nn.Linear(text_dim, dim, bias=False)
        self.k = nn.Linear(visual_dim, dim, bias=False)
        self.v = nn.Linear(visual_dim, dim, bias=False)
        self.ln_q, self.ln_k, self.ln_v = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.out = nn.Linear(dim, dim, bias=False)
        self.ln_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4*dim, dim), nn.Dropout(dropout))
        self.dropout = dropout

    def forward(self, text, patches, return_attention=False):
        q0, k0, v0 = self.q(text), self.k(patches), self.v(patches)
        def heads(x):
            return x.reshape(x.size(0), x.size(1), self.heads, self.head_dim).transpose(1, 2)
        q, k, v = heads(self.ln_q(q0)), heads(self.ln_k(k0)), heads(self.ln_v(v0))
        weights = None
        if return_attention:
            weights = ((q.float() @ k.float().transpose(-2, -1)) / math.sqrt(self.head_dim)).softmax(-1)
            o = F.dropout(weights, self.dropout, self.training).to(v.dtype) @ v
        else:
            o = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        o = o.transpose(1, 2).reshape_as(q0)
        r1 = q0 + self.out(o)
        return r1 + self.ffn(self.ln_ffn(r1)), weights


class Detector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone(cfg)
        b, m = self.backbone, cfg['model']
        self.repositories = m['repositories']
        self.shared_count, self.private_count = m['shared_tokens'], m['private_tokens']
        self.lengths = list(m['lengths'])
        self.fixed_length = m['fixed_length']
        self.register_buffer('length_values', torch.tensor(self.lengths, dtype=torch.float32))
        self.prompt_shared = nn.Parameter(torch.randn(self.repositories, self.shared_count, b.text_dim) * 0.02)
        self.prompt_private = nn.Parameter(torch.randn(self.repositories, self.private_count, b.text_dim) * 0.02)
        embeddings = b.clip.token_embedding(b.template_ids).detach().clone()
        self.register_buffer('word_embeddings', embeddings)
        self.class_residual = nn.Parameter(torch.zeros(2, b.text_dim), requires_grad=m['anchor_trainable'])
        self.probabilistic = ProbabilisticContext(cfg, b.joint_dim, b.text_dim)
        self.length_router = mlp(b.joint_dim, m['hidden_dim'], len(self.lengths))
        if self.fixed_length is not None or len(self.lengths) == 1:
            self.length_router.requires_grad_(False)
        self.cross_attention = TokenCrossAttention(b.text_dim, b.visual_dim, m['attention_dim'],
                                                   m['attention_heads'], m['attention_dropout']) if m['cross_attention'] else None
        self.interaction_projection = nn.Linear(m['attention_dim'], b.joint_dim, bias=False) if m['cross_attention'] else None
        initial_fraction = m['alpha_init'] / m['alpha_max']
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(initial_fraction / (1-initial_fraction))), requires_grad=m['cross_attention'])
        self.alpha_max = m['alpha_max']
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    @property
    def alpha(self):
        return self.alpha_max * self.alpha_logit.sigmoid()

    def build_prompts(self, shared_bias, private_bias, r):
        """Return [B,S,K,2,77,dt], exact EOS/context indices; labels are not inputs."""
        b, s, _, d = private_bias.shape
        shared = self.prompt_shared[None, None] + shared_bias[None, :, None]
        shared = shared.expand(b, -1, -1, -1, -1)
        private = self.prompt_private[None, None, :, :r] + private_bias[:, :, None, :r]
        context = torch.cat((shared, private), dim=-2)
        context = context.unsqueeze(3).expand(-1, -1, -1, 2, -1, -1)
        w = self.word_embeddings
        anchors = w[2:4] + self.class_residual
        prefix = torch.cat((w[:2].expand(2, -1, -1), anchors[:, None], w[4:6].expand(2, -1, -1)), dim=1)
        prefix = prefix[None, None, None].expand(b, s, self.repositories, -1, -1, -1)
        eos = w[6:7].view(1,1,1,1,1,d).expand(b,s,self.repositories,2,1,d)
        remaining = 77 - (6 + self.shared_count + r)
        pad = w[7:8].view(1,1,1,1,1,d).expand(b,s,self.repositories,2,remaining,d)
        return torch.cat((prefix, context, eos, pad), dim=-2)

    def _branch(self, g, patches, bs, bp, r, return_attention=False):
        b, samples = bp.shape[:2]
        outputs, maps = [], []
        for q in range(samples):
            prompts = self.build_prompts(bs[q:q+1], bp[:, q:q+1], r)
            flattened = prompts.reshape(b*self.repositories*2, 77, self.backbone.text_dim)
            eos, ctx = self.backbone.encode_text_embeddings(flattened, self.shared_count + r)
            prototype = F.normalize(eos.float(), dim=-1)
            if self.cross_attention is not None:
                visual = patches[:, None, None].expand(-1, self.repositories, 2, -1, -1)
                visual = visual.reshape(b*self.repositories*2, patches.size(1), patches.size(2))
                interacted, attn = self.cross_attention(ctx, visual, return_attention)
                local = F.normalize(self.interaction_projection(interacted.mean(dim=1)).float(), dim=-1)
                prototype = F.normalize(prototype + self.alpha * local, dim=-1)
                if attn is not None:
                    maps.append(attn.reshape(b, self.repositories, 2, *attn.shape[1:]))
            prototype = prototype.reshape(b, self.repositories, 2, -1)
            logits = self.logit_scale.exp() * (F.normalize(g.float(), dim=-1)[:, None, None] * prototype).sum(-1)
            outputs.append(logits.softmax(-1))
        return torch.stack(outputs, dim=1), maps

    def forward(self, images, *, epoch=0, samples=None, return_attention=False):
        """No label argument by design. Losses are computed separately below."""
        g, patches, patch_loss = self.backbone.encode_image(images)
        route_logits = self.length_router(F.normalize(g.detach().float(), dim=-1))
        pi = route_logits.float().softmax(-1)
        if self.fixed_length is not None:
            chosen = torch.full((len(images),), self.lengths.index(self.fixed_length), device=images.device, dtype=torch.long)
            h = F.one_hot(chosen, len(self.lengths)).float()
            pi = h
        elif self.training:
            temperature = max(0.1, 1.0 - 0.9 * epoch / self.cfg['train']['epochs'])
            h = F.gumbel_softmax(route_logits.float(), tau=temperature, hard=True)
            chosen = h.detach().argmax(-1)
        else:
            chosen = pi.argmax(-1)
            h = F.one_hot(chosen, len(self.lengths)).float()
        count = (1 if self.training else self.cfg['model']['eval_samples']) if samples is None else int(samples)
        if count < 1 or (self.training and count != 1):
            raise ValueError('Training uses S=1; inference samples must be positive')
        bs, bp, auxiliary = self.probabilistic(g, count)
        maps = {}
        if self.training and self.fixed_length is None:
            # All candidate probabilities must be present for the ST router gradient.
            # Only one candidate contributes to each image's FORWARD prediction.
            branches = [self._branch(g, patches, bs, bp, r)[0] for r in self.lengths]
            pair = (torch.stack(branches, dim=1) * h[:, :, None, None, None]).sum(dim=1)
        else:
            pair = g.new_zeros((len(images), count, self.repositories, 2), dtype=torch.float32)
            for j, r in enumerate(self.lengths):
                ids = (chosen == j).nonzero(as_tuple=True)[0]
                if not len(ids):
                    continue
                values, attention = self._branch(g[ids], patches[ids], bs, bp[ids], r, return_attention)
                pair = pair.index_copy(0, ids, values)
                if return_attention:
                    maps[str(r)] = {'indices': ids, 'attention': attention}
        residual_norm = self.class_residual.square().sum(-1)
        anchor_norm = self.word_embeddings[2:4].square().sum(-1).clamp_min(1e-8)
        auxiliary.update(patch=patch_loss, anchor=(residual_norm/anchor_norm).mean(),
                         length=(pi * self.length_values).sum(-1).mean()/self.private_count)
        return {'probabilities': pair.mean(dim=(1, 2)), 'pair_probabilities': pair,
                'length_probabilities': pi, 'private_lengths': self.length_values[chosen].long(),
                'auxiliary': auxiliary, 'attention': maps}


def detection_loss(output, labels, weights):
    pair = output['pair_probabilities']
    if labels.ndim != 1 or labels.size(0) != pair.size(0) or not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError('Labels must be a [B] tensor containing only 0=real or 1=fake')
    target = labels[:, None, None, None].expand(*pair.shape[:-1], 1)
    cls = -pair.gather(-1, target).clamp_min(1e-12).log().mean()
    terms = {'cls': cls, **output['auxiliary']}
    weights = output.get('loss_weights', weights)
    total = cls + sum(float(weights[name]) * value for name, value in output['auxiliary'].items())
    return total, terms
