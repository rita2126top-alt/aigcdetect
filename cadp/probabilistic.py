"""Equations (46)-(74), (114)-(123): shared/conditional planar flows."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


def mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))


def planar_step(z, u, w, b):
    """Broadcastable invertible planar flow, with exact log determinant."""
    a = (w * u).sum(-1, keepdim=True)
    target = -1.0 + F.softplus(a)
    u_hat = u + (target - a) * w / w.square().sum(-1, keepdim=True).clamp_min(1e-12)
    h = torch.tanh((w * z).sum(-1, keepdim=True) + b)
    result = z + u_hat * h
    det = 1.0 + (u_hat * w).sum(-1, keepdim=True) * (1.0 - h.square())
    return result, det.abs().clamp_min(1e-12).log().squeeze(-1)


def log_normal(z, mu, logvar):
    return -0.5 * (math.log(2 * math.pi) + logvar + (z - mu).square() * (-logvar).exp()).sum(-1)


class ProbabilisticContext(nn.Module):
    def __init__(self, cfg, joint_dim, text_dim):
        super().__init__()
        m = cfg['model']
        self.dim = joint_dim
        self.flows = m['flows']
        self.shared_count, self.private_count = m['shared_tokens'], m['private_tokens']
        self.stochastic = m['stochastic']
        self.rho_s, self.rho_p = m['rho_shared'], m['rho_private']
        hidden = m['hidden_dim']
        self.shared_mu = nn.Parameter(torch.zeros(joint_dim))
        self.shared_logvar = nn.Parameter(torch.zeros(joint_dim))
        self.private_stats = mlp(joint_dim, hidden, 2 * joint_dim)
        # Small conditional-flow initialization avoids an enormous initial KL.
        self.shared_flow = nn.Parameter(torch.randn(self.flows, 2 * joint_dim + 1) * 0.01)
        self.private_flow = nn.Linear(joint_dim, self.flows * (2 * joint_dim + 1)) if self.flows else None
        if self.private_flow is not None:
            nn.init.normal_(self.private_flow.weight, std=0.001)
            nn.init.normal_(self.private_flow.bias, std=0.01)
        self.shared_generator = mlp(joint_dim, hidden, self.shared_count * text_dim) if self.shared_count else None
        self.private_generator = mlp(joint_dim, hidden, self.private_count * text_dim)
        self.decoder = mlp(joint_dim, hidden, joint_dim)
        self.text_dim = text_dim
        gen = torch.Generator(device='cpu').manual_seed(m['noise_seed'])
        self.register_buffer('eval_shared_noise', torch.randn(m['noise_capacity'], joint_dim, generator=gen))
        self.register_buffer('eval_private_noise', torch.randn(m['noise_capacity'], joint_dim, generator=gen))

    def _flow(self, z, params):
        logdet = torch.zeros(z.shape[:-1], device=z.device, dtype=z.dtype)
        for f in range(self.flows):
            p = params[..., f, :]
            u, w, b = p[..., :self.dim], p[..., self.dim:2*self.dim], p[..., -1:]
            z, ld = planar_step(z, u, w, b)
            logdet = logdet + ld
        return z, logdet

    def forward(self, global_feature, samples):
        # Density/flow arithmetic always float32 even under mixed-precision training.
        with torch.autocast(device_type=global_feature.device.type, enabled=False):
            g = global_feature.float()
            batch = g.size(0)
            mu_p, lv_p = self.private_stats(g).chunk(2, -1)
            lv_p = lv_p.clamp(-12, 8)
            lv_s = self.shared_logvar.clamp(-12, 8)
            if not self.stochastic:
                eps_s = g.new_zeros(samples, self.dim)
                eps_p = g.new_zeros(batch, samples, self.dim)
            elif self.training:
                eps_s = torch.randn(samples, self.dim, device=g.device)
                eps_p = torch.randn(batch, samples, self.dim, device=g.device)
            else:
                if samples > self.eval_shared_noise.size(0):
                    raise ValueError('Requested samples exceed the checkpoint fixed noise capacity')
                eps_s = self.eval_shared_noise[:samples]
                eps_p = self.eval_private_noise[:samples].unsqueeze(0).expand(batch, -1, -1)
            zs0 = self.shared_mu + (0.5 * lv_s).exp() * eps_s
            zp0 = mu_p[:, None] + (0.5 * lv_p[:, None]).exp() * eps_p
            ps = self.shared_flow.unsqueeze(0)
            pp = self.private_flow(g).view(batch, 1, self.flows, 2 * self.dim + 1) if self.flows else None
            zs, lds = self._flow(zs0, ps)
            zp, ldp = self._flow(zp0, pp)
            # Monte Carlo KL is not clamped to zero: a single-sample estimate can be negative.
            kls = (log_normal(zs0, self.shared_mu, lv_s) - lds - log_normal(zs, 0, torch.zeros_like(zs))).mean()
            klp = (log_normal(zp0, mu_p[:, None], lv_p[:, None]) - ldp - log_normal(zp, 0, torch.zeros_like(zp))).mean()
            rec = F.mse_loss(self.decoder(zp), g.detach()[:, None].expand_as(zp))
            bs = self.shared_generator(zs).view(samples, self.shared_count, self.text_dim) if self.shared_generator is not None else g.new_zeros(samples, 0, self.text_dim)
            bp = self.private_generator(zp).view(batch, samples, self.private_count, self.text_dim)
            return self.rho_s * bs, self.rho_p * bp, {'kl': (kls + klp) / self.dim, 'rec': rec}
