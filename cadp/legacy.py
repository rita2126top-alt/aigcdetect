"""Controlled PPM-CLIP baseline using the ORIGINAL modules, without file edits.

Runtime-only compatibility changes: device-safe global flow, nonmerging LoRA,
local checkpoint loading and fixed-noise per-image evaluation. The detector
architecture and original train-time randomly selected prompt pair are retained.
This is a matched-harness baseline, NOT a claim to reproduce a published table.
"""
from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from .backbone import file_sha256


class LegacyDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        import clip
        from clip.model import CLIP
        import networks.Bayes as bayes
        from networks.PFL import PlanarPFL, PlanarPFL_learnable
        from networks.model_engine import PPM_clip
        import loralib.layers as layers
        m=cfg['model']; self.cfg=cfg
        locked={'repositories':2,'shared_tokens':3,'private_tokens':7,'flows':10,'lora_rank':4,'lora_alpha':.5,'dct_rate':.5}
        if any(m[k]!=v for k,v in locked.items()):
            raise ValueError('The matched PPM baseline fixes K=2, shared=3, private=7, flows=10, rank=4, alpha=.5, DCT=.5; use CADP for the supplied ablations.')
        # Original global and private forward bodies are otherwise identical.
        class DeviceSafeGlobalFlow(PlanarPFL_learnable):
            def forward(self, x, mode=None):
                return PlanarPFL.forward(self, self.state, mode=mode)
        class NonMergingLinearLoRA(layers.LinearLoRA):
            def train(self, mode=True):
                # Original forward already adds the low-rank residual explicitly.
                return nn.Linear.train(self, mode)
        original_load=clip.load
        def local_load(_name, device='cpu', **kwargs):
            if m['tiny']:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(m['tiny_seed'])
                    net=CLIP(embed_dim=32,image_resolution=28,vision_layers=24,vision_width=64,
                             vision_patch_size=14,context_length=77,vocab_size=49408,
                             transformer_width=32,transformer_heads=4,transformer_layers=2)
                return net.to(device), None
            from pathlib import Path
            path=Path(cfg['paths']['clip_checkpoint'])
            if not path.is_file(): raise FileNotFoundError(path)
            return original_load(str(path),device=device,jit=False)
        device=torch.device(cfg['train']['device'])
        if device.type=='cuda' and device.index is None:
            device=torch.device('cuda',torch.cuda.current_device())
        args=SimpleNamespace(gpu='-1' if device.type=='cpu' else str(device.index or 0),
                             backbone='ViT-L/14',lora_encoder='vision',lora_position='half-up',
                             lora_r=4,lora_alpha=.5,lora_dropout_rate=0.,lora_params='qkv',
                             num_select_rate=.5,prompt_share_len=3,prompt_private_len=7,
                             prompt_class_len=10,prompt_num=2,num_flows=10,sample_num=m['eval_samples'])
        with patch.object(clip,'load',local_load), patch.object(bayes,'PlanarPFL_learnable',DeviceSafeGlobalFlow), patch.object(layers,'LinearLoRA',NonMergingLinearLoRA):
            self.net=PPM_clip(args)
        self.backbone=SimpleNamespace(image_size=int(self.net.clip_model.visual.input_resolution),
                    fingerprint=(f'tiny-original-ppm-clip-depth24-seed{m["tiny_seed"]}' if m['tiny'] else file_sha256(cfg['paths']['clip_checkpoint'])))
        self.lengths=[7]

    def forward(self, images, *, epoch=0, samples=None, return_attention=False):
        if return_attention:
            raise ValueError('The original PPM baseline has no context-to-patch cross-attention to export.')
        if samples is not None and samples!=self.cfg['model']['eval_samples']:
            raise ValueError('For the legacy baseline, set model.eval_samples before model construction.')
        if self.training:
            logits, losses=self.net(images,None,mode='train')
            pair=logits.softmax(-1)[:,None,None,:]
            auxiliary={'rec':losses['rec'],'kl':losses['kl'],'patch':losses['contrastive'],'ort':losses['ort']}
        else:
            values=[]
            for x in images:
                devices=[images.device] if images.device.type=='cuda' else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(self.cfg['model']['noise_seed'])
                    self.net.context_prompting.zk_real=None; self.net.context_prompting.zk_fake=None
                    probs,_=self.net(x[None],None,mode='test');values.append(probs)
            pair=torch.cat(values)[:,None,None,:];auxiliary={}
        return {'pair_probabilities':pair,'probabilities':pair.mean((1,2)),
                'private_lengths':torch.full((len(images),),7,device=images.device,dtype=torch.long),
                'auxiliary':auxiliary,'attention':{},'uncertainty_supported':False,
                'loss_weights':{'rec':self.cfg['loss']['rec'],'kl':self.cfg['loss']['kl'],
                                'patch':self.cfg['loss']['patch'],'ort':self.cfg['model']['legacy_orthogonal_weight']}}
