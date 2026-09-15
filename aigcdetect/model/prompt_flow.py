from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

LOG_2PI = math.log(2.0 * math.pi)


def gaussian_log_prob(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (LOG_2PI + logvar + (z - mu).pow(2) * torch.exp(-logvar)).sum(dim=-1)


def standard_normal_log_prob(z: torch.Tensor) -> torch.Tensor:
    return -0.5 * (LOG_2PI + z.pow(2)).sum(dim=-1)


def make_u_hat(u: torch.Tensor, w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = (w * u).sum(dim=-1, keepdim=True)
    m = -1.0 + F.softplus(a)
    return u + ((m - a) / (w.pow(2).sum(dim=-1, keepdim=True) + eps)) * w


def planar_step(z: torch.Tensor, u: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    u_hat = make_u_hat(u, w, eps)
    linear = (w * z).sum(dim=-1, keepdim=True) + b
    t = torch.tanh(linear)
    z_new = z + u_hat * t
    psi = (1.0 - t.pow(2)) * w
    det = 1.0 + (u_hat * psi).sum(dim=-1)
    logabsdet = torch.log(det.abs() + eps)
    return z_new, logabsdet


class SharedPlanarFlow(nn.Module):
    def __init__(self, dim: int, num_flows: int):
        super().__init__()
        self.u = nn.Parameter(torch.randn(num_flows, dim) * 0.01)
        self.w = nn.Parameter(torch.randn(num_flows, dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(num_flows, 1))
        self.num_flows = num_flows

    def forward(self, z: torch.Tensor):
        logdet = torch.zeros(z.shape[:-1], device=z.device, dtype=z.dtype)
        for i in range(self.num_flows):
            u = self.u[i].view(*([1] * (z.ndim - 1)), -1).expand_as(z)
            w = self.w[i].view(*([1] * (z.ndim - 1)), -1).expand_as(z)
            b = self.b[i].view(*([1] * (z.ndim - 1)), 1)
            z, ld = planar_step(z, u, w, b)
            logdet = logdet + ld
        return z, logdet


class ConditionalPlanarFlow(nn.Module):
    """Image-conditioned flow parameters for private latent variables."""
    def __init__(self, cond_dim: int, dim: int, num_flows: int):
        super().__init__()
        self.dim = dim
        self.num_flows = num_flows
        self.param_net = nn.Linear(cond_dim, num_flows * (2 * dim + 1))
        nn.init.zeros_(self.param_net.bias)
        nn.init.normal_(self.param_net.weight, std=0.001)

    def forward(self, z: torch.Tensor, cond: torch.Tensor):
        # z: [B,S,D], cond: [B,C]
        bsz, samples, dim = z.shape
        params = self.param_net(cond).view(bsz, self.num_flows, 2 * dim + 1)
        logdet = torch.zeros(bsz, samples, device=z.device, dtype=z.dtype)
        for i in range(self.num_flows):
            p = params[:, i]
            u, w, bias = p[:, :dim], p[:, dim:2 * dim], p[:, -1:]
            u = u[:, None, :].expand(-1, samples, -1)
            w = w[:, None, :].expand(-1, samples, -1)
            bias = bias[:, None, :].expand(-1, samples, -1)
            z, ld = planar_step(z, u, w, bias)
            logdet = logdet + ld
        return z, logdet


class ProbabilisticPromptFlow(nn.Module):
    def __init__(self, image_dim: int, latent_dim: int, text_dim: int,
                 shared_tokens: int, private_tokens: int, num_flows: int,
                 hidden_dim: int = 1024, rho_shared: float = 0.1, rho_private: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.shared_tokens = shared_tokens
        self.private_tokens = private_tokens
        self.rho_shared = rho_shared
        self.rho_private = rho_private

        self.shared_mu = nn.Parameter(torch.zeros(latent_dim))
        self.shared_logvar = nn.Parameter(torch.zeros(latent_dim))
        self.private_stats = nn.Sequential(
            nn.Linear(image_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.shared_flow = SharedPlanarFlow(latent_dim, num_flows)
        self.private_flow = ConditionalPlanarFlow(image_dim, latent_dim, num_flows)
        self.shared_generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, shared_tokens * text_dim),
        )
        self.private_generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, private_tokens * text_dim),
        )
        self.private_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, image_dim)
        )

    def forward(self, image_features: torch.Tensor, sample_num: int):
        bsz = image_features.shape[0]
        dtype, device = image_features.dtype, image_features.device

        smu = self.shared_mu.to(dtype=dtype)
        slv = self.shared_logvar.to(dtype=dtype)
        eps_s = torch.randn(sample_num, self.latent_dim, device=device, dtype=dtype)
        z0_s = smu[None] + torch.exp(0.5 * slv)[None] * eps_s
        z_s, ld_s = self.shared_flow(z0_s)
        logq0_s = gaussian_log_prob(z0_s, smu[None], slv[None])
        kl_s = (logq0_s - ld_s - standard_normal_log_prob(z_s)).mean()

        stats = self.private_stats(image_features)
        pmu, plv = stats.chunk(2, dim=-1)
        plv = plv.clamp(-10.0, 10.0)
        eps_p = torch.randn(bsz, sample_num, self.latent_dim, device=device, dtype=dtype)
        z0_p = pmu[:, None] + torch.exp(0.5 * plv)[:, None] * eps_p
        z_p, ld_p = self.private_flow(z0_p, image_features)
        logq0_p = gaussian_log_prob(z0_p, pmu[:, None], plv[:, None])
        kl_p = (logq0_p - ld_p - standard_normal_log_prob(z_p)).mean()

        shared_bias = self.shared_generator(z_s).view(sample_num, self.shared_tokens, -1)
        private_bias = self.private_generator(z_p).view(bsz, sample_num, self.private_tokens, -1)
        rec = self.private_decoder(z_p)
        rec_loss = F.mse_loss(rec, image_features.detach()[:, None].expand_as(rec))
        return {
            "shared_bias": self.rho_shared * shared_bias,
            "private_bias": self.rho_private * private_bias,
            "kl_loss": (kl_s + kl_p) / self.latent_dim,
            "rec_loss": rec_loss,
            "z_private": z_p,
        }
