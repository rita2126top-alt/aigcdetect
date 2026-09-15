"""Non-invasive adapters around the original repository's CLIP modules."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


class QKVLoRA(nn.Module):
    """Frozen MHA + separate q/k/v low-rank residuals, alpha/sqrt(r) as upstream.

    No weight merging or .data writes: repeated train/eval transitions are exact.
    The upstream CLIP uses sequence-first self-attention and zero dropout.
    """
    def __init__(self, base: nn.MultiheadAttention, rank: int, alpha: float):
        super().__init__()
        if base.batch_first or not base._qkv_same_embed_dim or base.dropout != 0:
            raise ValueError('Expected the original CLIP sequence-first self-attention')
        self.base = base
        self.base.requires_grad_(False)
        width = base.embed_dim
        self.lora_A = nn.Parameter(torch.empty(3, rank, width))
        self.lora_B = nn.Parameter(torch.zeros(3, width, rank))
        for a in self.lora_A:
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.scale = float(alpha) / math.sqrt(rank)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        b = self.base
        update = torch.bmm(self.lora_B, self.lora_A).reshape(3 * b.embed_dim, b.embed_dim)
        return F.multi_head_attention_forward(
            query, key, value, b.embed_dim, b.num_heads,
            b.in_proj_weight + self.scale * update, b.in_proj_bias,
            b.bias_k, b.bias_v, b.add_zero_attn, b.dropout,
            b.out_proj.weight, b.out_proj.bias, training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, average_attn_weights=average_attn_weights,
            is_causal=is_causal)


class Backbone(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        from clip.model import CLIP
        import clip
        m = cfg['model']
        if m['tiny']:
            # A real, small upstream CLIP transformer, NOT a stub or pretrained detector.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(m['tiny_seed'])
                self.clip = CLIP(embed_dim=32, image_resolution=28, vision_layers=2,
                                 vision_width=64, vision_patch_size=7, context_length=77,
                                 vocab_size=49408, transformer_width=32,
                                 transformer_heads=4, transformer_layers=2).float()
            self.fingerprint = f'tiny-random-upstream-clip-v1-seed{m["tiny_seed"]}'
        else:
            path = Path(cfg['paths']['clip_checkpoint'])
            if not path.is_file():
                raise FileNotFoundError(f'Local CLIP checkpoint missing: {path}. Run download-model first.')
            # A filename, never a model name, is passed to clip.load: no implicit download.
            self.fingerprint = file_sha256(path)
            self.clip, _ = clip.load(str(path), device='cpu', jit=False)
            self.clip.float()
            v = self.clip.visual
            if (int(v.input_resolution) != 224 or v.conv1.kernel_size != (14, 14) or v.conv1.out_channels != 1024 or
                    len(v.transformer.resblocks) != 24 or self.clip.text_projection.shape != (768, 768)):
                raise ValueError('The local checkpoint is not the requested CLIP ViT-L/14')
        self.clip.requires_grad_(False)
        self.clip.eval()
        visual = self.clip.visual
        self.image_size = int(visual.input_resolution)
        self.patch_size = int(visual.conv1.kernel_size[0])
        self.visual_dim = visual.conv1.out_channels
        self.text_dim = self.clip.ln_final.weight.numel()
        self.joint_dim = self.clip.text_projection.shape[1]
        self.selected_layers = list(range(len(visual.transformer.resblocks)//2, len(visual.transformer.resblocks)))
        if m['lora_rank']:
            for i in self.selected_layers:
                block = visual.transformer.resblocks[i]
                block.attn = QKVLoRA(block.attn, m['lora_rank'], m['lora_alpha'])
        self.checkpoint_blocks = m['checkpoint_blocks']
        self.text_chunk_size = m['text_chunk_size']
        from networks.DCT_score import DCTPatches
        # Reuse original DCT score/selection, including six bands and complementary bottom set.
        self.dct = DCTPatches(self.patch_size, self.patch_size, 6, m['dct_rate'])
        self.dct.requires_grad_(False)
        self.compute_patch = cfg['loss']['patch'] > 0
        patch_count = (self.image_size // self.patch_size)**2
        if not 1 <= int(patch_count*m['dct_rate']) < patch_count:
            raise ValueError('DCT selection must leave at least one top and one bottom patch')
        # Verify the exact template's one-token assumption, rather than guessing BPE IDs.
        def word_id(word):
            ids = clip.tokenize(word)[0]
            end = int((ids == 49407).nonzero()[0])
            if end != 2:
                raise ValueError(f'Template word must be one BPE token: {word}')
            return int(ids[1])
        self.register_buffer('template_ids', torch.tensor([49406, word_id('a'), word_id('real'),
                                                         word_id('fake'), word_id('photo'), word_id('of'), 49407, 0]))

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip.eval()  # Freeze pretrained dropout/batch state, not prompt gradients.
        for module in self.clip.visual.modules():
            if isinstance(module, QKVLoRA):
                module.train(mode)
        return self

    def _block(self, block, x):
        if self.training and self.checkpoint_blocks and torch.is_grad_enabled() and x.requires_grad:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    @staticmethod
    def patch_loss(tokens, top, bottom):
        v = tokens[1:].permute(1, 0, 2).float()
        top_v = v.gather(1, top.unsqueeze(-1).expand(-1, -1, v.size(-1)))
        bottom_v = v.gather(1, bottom.unsqueeze(-1).expand(-1, -1, v.size(-1)))
        anchor = top_v[:, :1]
        positive = (top_v[:, 1:] - anchor).square().sum(-1).sum(-1)
        negative = (1.0 - (bottom_v - anchor).square().sum(-1)).clamp_min(0).sum(-1)
        return (positive + negative).mean()

    def encode_image(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, self.image_size, self.image_size):
            raise ValueError(f'Expected [B,3,{self.image_size},{self.image_size}], got {tuple(images.shape)}')
        top = bottom = None
        if self.training and self.compute_patch:
            with torch.no_grad(), torch.autocast(device_type=images.device.type, enabled=False):
                top, bottom = self.dct(images.float())
        v = self.clip.visual
        x = v.conv1(images).flatten(2).transpose(1, 2)
        cls = v.class_embedding.to(x.dtype).view(1, 1, -1).expand(x.size(0), 1, -1)
        x = v.ln_pre(torch.cat((cls, x), dim=1) + v.positional_embedding.to(x.dtype))
        x = x.permute(1, 0, 2)
        patch_loss = images.new_zeros((), dtype=torch.float32)
        for index, block in enumerate(v.transformer.resblocks):
            x = self._block(block, x)
            if top is not None and index in self.selected_layers:
                patch_loss = patch_loss + self.patch_loss(x, top, bottom)
        x = v.ln_post(x.permute(1, 0, 2))
        return x[:, 0] @ v.proj, x[:, 1:], patch_loss / len(self.selected_layers)

    def encode_text_embeddings(self, prompts, context_length: int):
        eos_index = 5 + context_length
        if prompts.shape[1] != 77:
            raise ValueError('Text inputs must be padded to 77 positions')
        eos_features, contexts = [], []
        for p in prompts.split(self.text_chunk_size):
            x = (p + self.clip.positional_embedding.to(p.dtype)).permute(1, 0, 2)
            # Deliberately NOT no_grad(): learned tokens must receive gradients through frozen text.
            for block in self.clip.transformer.resblocks:
                x = self._block(block, x)
            x = self.clip.ln_final(x.permute(1, 0, 2))
            eos_features.append(x[:, eos_index] @ self.clip.text_projection)
            contexts.append(x[:, 5:eos_index])
        return torch.cat(eos_features), torch.cat(contexts)
