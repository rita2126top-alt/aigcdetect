import torch
from aigcdetect.model.components import LengthRouter, TokenCrossAttention, InteractivePrototypeFusion
from aigcdetect.model.prompt_flow import ProbabilisticPromptFlow

def test_components_shapes():
    b,s,d,t,v=2,3,64,32,48
    flow=ProbabilisticPromptFlow(d,d,t,3,7,2,hidden_dim=64)
    o=flow(torch.randn(b,d),s)
    assert o["shared_bias"].shape==(s,3,t)
    assert o["private_bias"].shape==(b,s,7,t)
    r=LengthRouter(d); _,p,h=r(torch.randn(b,d),True,1.0); assert p.shape==(b,4) and h.shape==(b,4)
    ca=TokenCrossAttention(t,v,attn_dim=32,heads=4); z=ca(torch.randn(b,6,t),torch.randn(b,16,v)); assert z.shape==(b,6,32)
    f=InteractivePrototypeFusion(32,d); q=f(torch.randn(b,d),z); assert q.shape==(b,d)
