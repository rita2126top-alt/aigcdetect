import copy
import pytest
import torch
from aigcdetect.model import ClassAnchoredDynamicPPMCLIP as Detector
from aigcdetect.engine.train import weighted_loss, seed_all
from aigcdetect.model.prompt_flow import planar_step


def test_full_gradient_and_frozen_backbone(cfg):
    seed_all(2)
    m = Detector(cfg).train()
    frozen = {k: v.clone() for k, v in m.clip_model.named_parameters() if not v.requires_grad}
    x = torch.randn(2, 3, 28, 28)
    out = m(x, torch.tensor([0, 1]), mode="train")
    assert set(out["losses"]) == {"cls", "rec", "kl", "patch", "anchor", "len"}
    weighted_loss(out["losses"], cfg["loss"]).backward()
    grads = dict(m.named_parameters())
    for pattern in ("delta_real", "prompt_shared", "prompt_private", "length_router", "prompt_flow", "cross_attention", "lora_B"):
        selected = [p.grad for n, p in grads.items() if pattern in n and p.grad is not None]
        assert selected and all(torch.isfinite(g).all() for g in selected), pattern
        assert any(g.abs().sum() > 0 for g in selected), pattern
    optimizer = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    optimizer.step()
    m.eval(); m.train(); m.eval()
    for n, p in m.clip_model.named_parameters():
        if n in frozen:
            assert p.grad is None and torch.equal(p, frozen[n]), n


def test_context_shared_labels_not_used_and_fixed_eval(cfg):
    m = Detector(cfg)
    context = torch.randn(2, 1, 2, 6, m.text_width)
    prompts, eos, span = m._build_prompt_embeddings(context)
    assert eos == 11 and prompts.shape == (2, 1, 2, 2, 77, 64)
    assert torch.equal(prompts[:, :, :, 0, span], prompts[:, :, :, 1, span])
    assert not torch.equal(prompts[:, :, :, 0, 2], prompts[:, :, :, 1, 2])
    x = torch.randn(2, 3, 28, 28)
    m.train()
    seed_all(3); a = m(x, torch.tensor([0, 1]), mode="train")["probabilities"]
    seed_all(3); b = m(x, torch.tensor([1, 0]), mode="train")["probabilities"]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    m.eval()
    with torch.no_grad():
        p = m(x)["probabilities"]
        seed_all(777)
        q = torch.cat([m(x[i:i+1])["probabilities"] for i in range(2)])
    torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(p.sum(-1), torch.ones(2))


@pytest.mark.parametrize("switch", ["disable_cross_attention", "disable_dynamic_length", "disable_probabilistic_prompt", "disable_dct_loss", "freeze_class_anchors", "disable_lora"])
def test_ablation_forward_backward(cfg, switch):
    cfg["ablation"][switch] = True
    m = Detector(cfg).train()
    out = m(torch.randn(1, 3, 28, 28), torch.tensor([1]), mode="train")
    weighted_loss(out["losses"], cfg["loss"]).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    if switch == "disable_dynamic_length":
        assert out["selected_private_length"].item() == 7
        assert out["losses"]["len"] == 0
    if switch == "disable_probabilistic_prompt":
        assert out["losses"]["kl"] == out["losses"]["rec"] == 0
    if switch == "freeze_class_anchors":
        assert m.delta_real.grad is None
    if switch == "disable_lora":
        assert not m.trainable_parameter_groups()[0]


def test_planar_jacobian():
    torch.manual_seed(5)
    z, u, w = (torch.randn(4, dtype=torch.float64) for _ in range(3))
    b = torch.randn(1, dtype=torch.float64)
    jac = torch.autograd.functional.jacobian(lambda a: planar_step(a, u, w, b)[0], z)
    _, logdet = planar_step(z, u, w, b)
    assert torch.linalg.det(jac) > 0
    torch.testing.assert_close(logdet, torch.linalg.slogdet(jac)[1], atol=1e-7, rtol=1e-6)


def test_chunked_predictions_equal(cfg):
    m = Detector(cfg).eval()
    x = torch.randn(2, 3, 28, 28)
    with torch.no_grad():
        a = m(x)["probabilities"]
        m.branch_image_chunk = 10
        m.branch_sample_chunk = 10
        m.text_chunk = 100
        b = m(x)["probabilities"]
    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
