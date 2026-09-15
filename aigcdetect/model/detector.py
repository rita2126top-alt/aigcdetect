from __future__ import annotations
import os
import sys
from pathlib import Path
from types import SimpleNamespace, MethodType
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..io_utils import sha256_file
from .prompt_flow import ProbabilisticPromptFlow
from .components import LengthRouter, TokenCrossAttention, InteractivePrototypeFusion

VISION_LORA_INDICES = {
    "ViT-L/14": {"half-up": list(range(12, 24)), "half-bottom": list(range(12)), "all": list(range(24))},
    "ViT-B/16": {"half-up": list(range(6, 12)), "half-bottom": list(range(6)), "all": list(range(12))},
    "ViT-B/32": {"half-up": list(range(6, 12)), "half-bottom": list(range(6)), "all": list(range(12))},
}


def _ensure_upstream(ppmclip_root: str):
    root = Path(ppmclip_root).expanduser().resolve()
    for name in ("clip/clip.py", "loralib/utils.py", "networks/DCT_score.py"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"Missing {root / name}; run git submodule update --init --recursive")
    sys.path.insert(0, str(root)) if str(root) not in sys.path else None
    import clip
    import loralib
    from networks import DCT_score
    for module in (clip, loralib, DCT_score):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError(f"Conflicting import {module.__file__}; start a fresh process using {root}")


def _lora_train_unmerged(self, mode=True):
    # The pinned upstream LinearLoRA forward always adds BA separately. Avoid its
    # eval-time merge/unmerge roundoff and keep checkpoint base weights canonical.
    nn.Module.train(self, mode)
    if self.merged:
        self.sub_lora_data()
        self.merged = False
    return self


def _load_local_clip(path):
    from clip.model import build_model
    # Separate opens: a failed JIT read must not leave torch.load at EOF.
    try:
        state = torch.jit.load(str(path), map_location="cpu").state_dict()
    except RuntimeError:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    return build_model(state).float().eval()


def _first_word_token(clip_module, word: str, device: torch.device) -> torch.Tensor:
    ids = clip_module.tokenize(word)[0]
    ids = ids[(ids != 0) & (ids != 49406) & (ids != 49407)]
    if len(ids) < 1:
        raise RuntimeError(f"No CLIP token for word: {word}")
    if len(ids) > 1:
        raise RuntimeError(f"Expected single CLIP token for '{word}', got token ids {ids.tolist()}")
    return ids[0].to(device=device, dtype=torch.long)


class ClassAnchoredDynamicPPMCLIP(nn.Module):
    """Implementation of the user-specified class-anchored dynamic probabilistic PPM-CLIP."""
    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        paths, model_cfg = cfg["paths"], cfg["model"]
        _ensure_upstream(paths["ppmclip_root"])
        import clip
        from loralib.utils import apply_lora, mark_only_lora_as_trainable
        from networks.DCT_score import DCTPatches

        self.clip_module = clip
        self.backbone_name = model_cfg.get("backbone", "ViT-L/14")
        if self.backbone_name not in VISION_LORA_INDICES:
            raise ValueError(f"Unsupported ViT backbone: {self.backbone_name}")
        model_path = paths["clip_model"]
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Local CLIP checkpoint not found: {model_path}. Run scripts/download_clip.sh or edit paths.clip_model."
            )
        self.clip_sha256 = sha256_file(model_path)
        self.clip_model = _load_local_clip(model_path)
        self.clip_model.float().eval()

        self.visual_width = int(self.clip_model.visual.transformer.width)
        self.text_width = int(self.clip_model.transformer.width)
        self.joint_dim = int(self.clip_model.text_projection.shape[1])
        self.context_length = int(self.clip_model.context_length)

        lora_cfg = model_cfg["lora"]
        args = SimpleNamespace(
            lora_encoder="vision",
            lora_position=lora_cfg.get("position", "half-up"),
            lora_alpha=float(lora_cfg.get("alpha", 0.5)),
            lora_r=int(lora_cfg.get("rank", 4)),
            lora_params=lora_cfg.get("target", "qkv"),
            lora_dropout_rate=float(lora_cfg.get("dropout", 0.0)),
            backbone=self.backbone_name,
        )
        self.lora_layers = apply_lora(args, self.clip_model)
        mark_only_lora_as_trainable(self.clip_model)
        self.lora_indices = VISION_LORA_INDICES[self.backbone_name][args.lora_position]
        if not self.lora_layers:
            raise ValueError("No vision LoRA blocks selected; check backbone and checkpoint")
        for module in self.clip_model.modules():
            if hasattr(module, "w_lora_A"):
                module.train = MethodType(_lora_train_unmerged, module)

        patch_size = int(self.clip_model.visual.conv1.kernel_size[0])
        self.dct = DCTPatches(window_size=patch_size, stride=patch_size, grade_N=6,
                              num_select_rate=float(model_cfg.get("dct_select_rate", 0.5)))
        self.patch_margin = float(model_cfg.get("dct_margin", 1.0))
        resolution = int(self.clip_model.visual.input_resolution)
        if int(cfg["data"].get("image_size", 224)) != resolution:
            raise ValueError(f"data.image_size must equal CLIP input resolution {resolution}")
        n_patches = (resolution // patch_size) ** 2
        if not 1 <= int(n_patches * self.dct.num_select_rate) < n_patches:
            raise ValueError("DCT selection must leave nonempty top and bottom patch groups")

        p = model_cfg["prompt"]
        self.num_repos = int(p.get("repositories", 2))
        self.shared_count = int(p.get("shared_tokens", 3))
        self.private_max = int(p.get("private_max", 7))
        self.length_candidates = tuple(p.get("private_candidates", [1, 3, 5, 7]))
        if not self.length_candidates or any(r < 1 or r > self.private_max for r in self.length_candidates):
            raise ValueError("private_candidates must lie in [1, private_max]")
        if self.shared_count + self.private_max + 6 > self.context_length:
            raise ValueError("Prompt including SOS/EOS exceeds CLIP context length")
        self.branch_image_chunk = int(p.get("image_chunk_size", 4))
        self.branch_sample_chunk = int(p.get("sample_chunk_size", 1))
        self.text_chunk = int(p.get("text_chunk_size", 32))
        self.checkpoint_prompts = bool(p.get("gradient_checkpointing", True))
        if min(self.branch_image_chunk, self.branch_sample_chunk, self.text_chunk) < 1:
            raise ValueError("Prompt chunk sizes must be positive")
        self.prompt_shared = nn.Parameter(torch.randn(self.num_repos, self.shared_count, self.text_width) * 0.02)
        self.prompt_private = nn.Parameter(torch.randn(self.num_repos, self.private_max, self.text_width) * 0.02)

        with torch.no_grad():
            real_id = _first_word_token(clip, "real", torch.device("cpu"))
            fake_id = _first_word_token(clip, "fake", torch.device("cpu"))
            a_id = _first_word_token(clip, "a", torch.device("cpu"))
            photo_id = _first_word_token(clip, "photo", torch.device("cpu"))
            of_id = _first_word_token(clip, "of", torch.device("cpu"))
            self.register_buffer("real_base", self.clip_model.token_embedding(real_id).detach().clone(), persistent=True)
            self.register_buffer("fake_base", self.clip_model.token_embedding(fake_id).detach().clone(), persistent=True)
            self.register_buffer("fixed_a", self.clip_model.token_embedding(a_id).detach().clone(), persistent=False)
            self.register_buffer("fixed_photo", self.clip_model.token_embedding(photo_id).detach().clone(), persistent=False)
            self.register_buffer("fixed_of", self.clip_model.token_embedding(of_id).detach().clone(), persistent=False)
            self.register_buffer("fixed_sos", self.clip_model.token_embedding(torch.tensor(49406)).detach().clone(), persistent=False)
            self.register_buffer("fixed_eos", self.clip_model.token_embedding(torch.tensor(49407)).detach().clone(), persistent=False)
        self.delta_real = nn.Parameter(torch.zeros(self.text_width))
        self.delta_fake = nn.Parameter(torch.zeros(self.text_width))

        flow = model_cfg["flow"]
        self.prompt_flow = ProbabilisticPromptFlow(
            image_dim=self.joint_dim,
            latent_dim=self.joint_dim,
            text_dim=self.text_width,
            shared_tokens=self.shared_count,
            private_tokens=self.private_max,
            num_flows=int(flow.get("layers", 10)),
            hidden_dim=int(flow.get("hidden_dim", 1024)),
            rho_shared=float(flow.get("rho_shared", 0.1)),
            rho_private=float(flow.get("rho_private", 0.1)),
        )
        self.length_router = LengthRouter(self.joint_dim, self.length_candidates,
                                          hidden_dim=int(p.get("router_hidden", 512)))

        ca = model_cfg["cross_attention"]
        self.cross_attention = TokenCrossAttention(
            self.text_width, self.visual_width,
            attn_dim=int(ca.get("dim", 256)), heads=int(ca.get("heads", 4)),
            dropout=float(ca.get("dropout", 0.1)), ffn_ratio=int(ca.get("ffn_ratio", 4)),
        )
        self.fusion = InteractivePrototypeFusion(
            int(ca.get("dim", 256)), self.joint_dim,
            alpha_max=float(ca.get("alpha_max", 0.5)), alpha_init=float(ca.get("alpha_init", 0.1)),
        )
        self.temperature_image = nn.Parameter(torch.tensor(np.log(1.0 / 0.07), dtype=torch.float32))

        # Freeze every original CLIP parameter except LoRA params; text transformer still participates in autograd.
        for name, param in self.clip_model.named_parameters():
            param.requires_grad = "lora_" in name

        ab = cfg.get("ablation", {})
        self.disable_cross_attention = bool(ab.get("disable_cross_attention", False))
        self.disable_dynamic_length = bool(ab.get("disable_dynamic_length", False))
        self.disable_probabilistic_prompt = bool(ab.get("disable_probabilistic_prompt", False))
        self.disable_dct_loss = bool(ab.get("disable_dct_loss", False))
        if ab.get("freeze_class_anchors", False):
            self.delta_real.requires_grad_(False)
            self.delta_fake.requires_grad_(False)
        if ab.get("disable_lora", False):
            for parameter in self.clip_model.parameters():
                parameter.requires_grad_(False)
        # Fixed base Gaussian samples, shared by all test images. Conditioning
        # still makes private prompts image-dependent. Independent of batch size.
        rng = torch.Generator(device="cpu").manual_seed(int(p.get("eval_seed", 0)))
        n_eval = int(p.get("test_samples", 10))
        self.register_buffer("eval_eps_shared", torch.randn(n_eval, self.joint_dim, generator=rng), persistent=False)
        self.register_buffer("eval_eps_private", torch.randn(n_eval, self.joint_dim, generator=rng), persistent=False)

    def patch_contrastive_loss(self, tokens_lnd, top_idx, bottom_idx):
        # tokens_lnd: [N+1,B,D]
        patches = tokens_lnd[1:].permute(1, 0, 2).float()
        dim = patches.size(-1)
        top = torch.gather(patches, 1, top_idx.unsqueeze(-1).expand(-1, -1, dim))
        bottom = torch.gather(patches, 1, bottom_idx.unsqueeze(-1).expand(-1, -1, dim))
        anchor = top[:, :1]
        pos = (top[:, 1:] - anchor).pow(2).sum(-1)
        neg = (bottom - anchor).pow(2).sum(-1)
        pos_loss = pos.sum(dim=1) if pos.shape[1] else torch.zeros_like(neg[:, 0])
        neg_loss = torch.clamp(self.patch_margin - neg, min=0).sum(dim=1)
        return (pos_loss + neg_loss).mean()

    def encode_image_lora(self, image: torch.Tensor, training: bool):
        visual = self.clip_model.visual
        x_in = image.float()
        top_idx = bottom_idx = None
        if training and not self.disable_dct_loss:
            with torch.autocast(device_type=image.device.type, enabled=False):
                top_idx, bottom_idx = self.dct(x_in.float())
        x = visual.conv1(x_in)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype)
        x = torch.cat([cls, x], dim=1)
        x = visual.ln_pre(x + visual.positional_embedding.to(x.dtype))
        x = x.permute(1, 0, 2)
        patch_loss = x.new_zeros(())
        count = 0
        for i, block in enumerate(visual.transformer.resblocks):
            x = block(x)
            if training and not self.disable_dct_loss and i in self.lora_indices:
                patch_loss = patch_loss + self.patch_contrastive_loss(x, top_idx, bottom_idx)
                count += 1
        if count:
            patch_loss = patch_loss / count
        x = x.permute(1, 0, 2)
        normed = visual.ln_post(x)
        global_feature = normed[:, 0] @ visual.proj
        patch_tokens = normed[:, 1:]
        return global_feature, patch_tokens, patch_loss

    def _context_for_length(self, flow_out, private_len: int):
        # returns [B,S,K,M,T]
        shared = self.prompt_shared[None, None] + flow_out["shared_bias"][None, :, None]
        # [1,S,K,Ms,T] -> B expansion
        bsz = flow_out["private_bias"].shape[0]
        shared = shared.expand(bsz, -1, -1, -1, -1)
        private = self.prompt_private[None, None, :, :private_len] + flow_out["private_bias"][:, :, None, :private_len]
        return torch.cat([shared, private], dim=-2)

    def _build_prompt_embeddings(self, context: torch.Tensor):
        # context [B,S,K,M,T] -> prompts [B,S,K,2,77,T]
        bsz, samp, repos, m, dim = context.shape
        device, dtype = context.device, context.dtype
        cls_tokens = torch.stack([self.real_base + self.delta_real, self.fake_base + self.delta_fake], dim=0).to(dtype=dtype)
        out = torch.zeros(bsz, samp, repos, 2, self.context_length, dim, device=device, dtype=dtype)
        out[..., 0, :] = self.fixed_sos.to(dtype=dtype)
        out[..., 1, :] = self.fixed_a.to(dtype=dtype)
        out[..., 2, :] = cls_tokens.view(1, 1, 1, 2, dim)
        out[..., 3, :] = self.fixed_photo.to(dtype=dtype)
        out[..., 4, :] = self.fixed_of.to(dtype=dtype)
        out[..., 5:5 + m, :] = context[:, :, :, None].expand(-1, -1, -1, 2, -1, -1)
        eos_index = 5 + m
        out[..., eos_index, :] = self.fixed_eos.to(dtype=dtype)
        return out, eos_index, slice(5, 5 + m)

    def _encode_prompts(self, prompt_embeddings: torch.Tensor, eos_index: int, ctx_slice: slice):
        shape = prompt_embeddings.shape[:-2]
        seq, dim = prompt_embeddings.shape[-2:]
        x = prompt_embeddings.reshape(-1, seq, dim)
        all_eos, all_ctx = [], []
        for chunk in x.split(self.text_chunk):
            chunk = chunk + self.clip_model.positional_embedding.to(dtype=chunk.dtype, device=chunk.device)
            chunk = self.clip_model.transformer(chunk.permute(1, 0, 2)).permute(1, 0, 2)
            chunk = self.clip_model.ln_final(chunk)
            all_ctx.append(chunk[:, ctx_slice, :])
            all_eos.append(chunk[:, eos_index, :] @ self.clip_model.text_projection)
        eos, ctx = torch.cat(all_eos), torch.cat(all_ctx)
        return eos.reshape(*shape, -1), ctx.reshape(*shape, ctx.shape[-2], ctx.shape[-1])

    def _branch_probabilities(self, image_norm, patch_tokens, context):
        # Bound both text and repeated visual KV memory without dropping any
        # repository or Monte Carlo sample. Recompute training activations.
        p_images, l_images = [], []
        for b in range(0, len(image_norm), self.branch_image_chunk):
            p_samples, l_samples = [], []
            for q in range(0, context.shape[1], self.branch_sample_chunk):
                args = (image_norm[b:b+self.branch_image_chunk],
                        patch_tokens[b:b+self.branch_image_chunk],
                        context[b:b+self.branch_image_chunk, q:q+self.branch_sample_chunk])
                if self.training and self.checkpoint_prompts and torch.is_grad_enabled():
                    p, l = checkpoint(self._branch_core, *args, use_reentrant=False)
                else:
                    p, l = self._branch_core(*args)
                p_samples.append(p)
                l_samples.append(l)
            p_images.append(torch.cat(p_samples, dim=1))
            l_images.append(torch.cat(l_samples, dim=1))
        return torch.cat(p_images, dim=0), torch.cat(l_images, dim=0)

    def _branch_core(self, image_norm, patch_tokens, context):
        prompts, eos_index, ctx_slice = self._build_prompt_embeddings(context)
        eos, ctx = self._encode_prompts(prompts, eos_index, ctx_slice)
        # eos [B,S,K,2,D], ctx [B,S,K,2,M,T]
        bsz, samp, repos, classes, m, _ = ctx.shape
        if self.disable_cross_attention:
            proto = F.normalize(eos, dim=-1)
        else:
            q = ctx.reshape(-1, m, self.text_width)
            visual = patch_tokens[:, None, None, None].expand(-1, samp, repos, classes, -1, -1)
            visual = visual.reshape(-1, patch_tokens.shape[1], self.visual_width)
            inter = self.cross_attention(q, visual)
            proto = self.fusion(eos.reshape(-1, self.joint_dim), inter).reshape(bsz, samp, repos, classes, self.joint_dim)
        with torch.autocast(device_type=image_norm.device.type, enabled=False):
            logits = self.temperature_image.float().exp() * torch.einsum("bd,bskcd->bskc", image_norm.float(), proto.float())
        return logits.softmax(dim=-1), logits

    def anchor_regularization(self):
        eps = 1e-8
        real = self.delta_real.pow(2).sum() / (self.real_base.pow(2).sum() + eps)
        fake = self.delta_fake.pow(2).sum() / (self.fake_base.pow(2).sum() + eps)
        return 0.5 * (real + fake)

    def gumbel_temperature(self, epoch: int, max_epochs: int):
        return max(0.1, 1.0 - 0.9 * float(epoch) / max(float(max_epochs), 1.0))

    def forward(self, image: torch.Tensor, labels: torch.Tensor | None = None,
                mode: str = "test", epoch: int = 0, max_epochs: int = 100):
        training = mode == "train"
        global_feature, patch_tokens, patch_loss = self.encode_image_lora(image, training=training)
        image_norm = F.normalize(global_feature, dim=-1)
        tau = self.gumbel_temperature(epoch, max_epochs)
        _, length_probs, length_hard = self.length_router(image_norm, training=training, temperature=tau)
        if self.disable_dynamic_length:
            fixed_idx = self.length_candidates.index(int(self.cfg["model"]["prompt"].get("fixed_private_length", max(self.length_candidates))))
            length_hard = F.one_hot(torch.full((image.shape[0],), fixed_idx, device=image.device), len(self.length_candidates)).to(image_norm.dtype)
            length_probs = length_hard

        sample_num = int(self.cfg["model"]["prompt"].get("train_samples", 1) if training else self.cfg["model"]["prompt"].get("test_samples", 10))
        if self.disable_probabilistic_prompt:
            sample_num = 1
        if mode not in ("train", "test"):
            raise ValueError("mode must be train or test")
        if sample_num < 1:
            raise ValueError("Prompt sample count must be positive")
        if self.disable_probabilistic_prompt:
            flow_out = {"shared_bias": global_feature.new_zeros(sample_num, self.shared_count, self.text_width),
                        "private_bias": global_feature.new_zeros(len(image), sample_num, self.private_max, self.text_width),
                        "kl_loss": global_feature.new_zeros(()), "rec_loss": global_feature.new_zeros(())}
        else:
            # Flow densities, invertibility corrections and KL stay FP32 under AMP.
            with torch.autocast(device_type=image.device.type, enabled=False):
                flow_out = self.prompt_flow(global_feature.float(), sample_num,
                    eps_shared=None if training else self.eval_eps_shared,
                    eps_private=None if training else self.eval_eps_private)

        if training:
            probs_by_len, logits_by_len = [], []
            for r in self.length_candidates:
                context = self._context_for_length(flow_out, int(r))
                probs, logits = self._branch_probabilities(image_norm, patch_tokens, context)
                probs_by_len.append(probs)
                logits_by_len.append(logits)
            selector = length_hard[:, None, None, :, None]
            selected_probs = (torch.stack(probs_by_len, dim=3) * selector).sum(dim=3)
            selected_logits = (torch.stack(logits_by_len, dim=3) * selector).sum(dim=3)
        else:
            # Test only the selected length, not four unused text branches.
            selected_probs = image_norm.new_empty(len(image), sample_num, self.num_repos, 2, dtype=torch.float32)
            selected_logits = torch.empty_like(selected_probs)
            choices = length_hard.argmax(dim=-1)
            for j, r in enumerate(self.length_candidates):
                idx = (choices == j).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                subset_flow = {"shared_bias": flow_out["shared_bias"], "private_bias": flow_out["private_bias"][idx]}
                context = self._context_for_length(subset_flow, int(r))
                probs, logits = self._branch_probabilities(image_norm[idx], patch_tokens[idx], context)
                selected_probs[idx], selected_logits[idx] = probs, logits
        final_probs = selected_probs.mean(dim=(1, 2))

        out = {
            "probabilities": final_probs,
            "pair_probabilities": selected_probs,
            "logits": selected_logits,
            "length_probabilities": length_probs,
            "length_choice": length_hard.argmax(dim=-1),
            "selected_private_length": image.new_tensor(self.length_candidates, dtype=torch.long)[length_hard.argmax(dim=-1)],
            "alpha": self.fusion.alpha.detach(),
        }
        if training:
            if labels is None:
                raise ValueError("labels are required in train mode")
            target = labels[:, None, None, None].expand(-1, sample_num, self.num_repos, 1)
            p_y = torch.gather(selected_probs.clamp_min(1e-8), -1, target).squeeze(-1)
            cls_loss = -torch.log(p_y).mean()
            len_loss = global_feature.new_zeros(()) if self.disable_dynamic_length else self.length_router.expected_length_loss(length_probs, self.private_max)
            out["losses"] = {
                "cls": cls_loss,
                "rec": flow_out["rec_loss"],
                "kl": flow_out["kl_loss"],
                "patch": patch_loss,
                "anchor": self.anchor_regularization(),
                "len": len_loss,
            }
        return out

    def trainable_parameter_groups(self):
        lora, other = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (lora if "clip_model" in name and "lora_" in name else other).append(p)
        return lora, other
