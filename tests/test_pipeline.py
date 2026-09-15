import copy
import json
from pathlib import Path
import torch
import pytest
import yaml
from aigcdetect.data.folder_dataset import build_loaders
from aigcdetect.model import ClassAnchoredDynamicPPMCLIP as Detector
from aigcdetect.engine.train import train, seed_all, read_checkpoint, load_checkpoint
from aigcdetect.cli import eval_main, infer_main, robustness_main, check_main


def execute(c, stop=None):
    seed_all(c["training"]["seed"])
    model = Detector(c)
    tr, val, _ = build_loaders(c, stage="train")
    path = train(c, model, tr, val, torch.device("cpu"), stop_after_epoch=stop)
    return model, path


def test_epoch_resume_exact(cfg, tmp_path):
    full = copy.deepcopy(cfg)
    full["paths"]["output_dir"] = str(tmp_path / "full")
    m1, _ = execute(full)
    interrupted = copy.deepcopy(cfg)
    interrupted["paths"]["output_dir"] = str(tmp_path / "resume")
    execute(interrupted, stop=0)
    ckpt = tmp_path / "resume/last.pt"
    interrupted["training"]["resume"] = str(ckpt)
    m2, best = execute(interrupted)
    for (n, a), (n2, b) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert n == n2
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=n)
    ck = read_checkpoint(ckpt)
    assert ck["format"] == "adapter_v1" and ck["epoch"] == 1
    assert all(not k.startswith("clip_model.") or "lora_" in k for k in ck["model"])
    assert ck["scheduler"] and ck["rng"] and Path(best).exists()


def test_real_cli_pipeline(cfg, tmp_path):
    cfg["training"]["epochs"] = 1
    cfg["evaluation"].update(jpeg_qualities=[75], blur_radii=[1])
    _, best = execute(cfg)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(cfg))
    args = ["--config", str(config_file), "--checkpoint", best]
    eval_main(args)
    summary = json.loads((Path(cfg["paths"]["output_dir"]) / "eval/summary.json").read_text())
    assert summary["domains"]["domain"]["n"] == 2
    robustness_main(args)
    robust = json.loads((Path(cfg["paths"]["output_dir"]) / "eval/robustness_summary.json").read_text())
    assert set(robust) == {"clean", "jpeg75", "blur1"}
    image = next(Path(cfg["data"]["val_root"]).rglob("*.png"))
    infer_main([str(image), *args])
    check_main(["--config", str(config_file), "--stage", "infer", "--forward", "--backward"])
    wrong = Detector(cfg)
    wrong.clip_sha256 = "different"
    with pytest.raises(ValueError, match="SHA256"):
        load_checkpoint(best, wrong)


def test_independent_train_and_eval_data(cfg):
    train_cfg = copy.deepcopy(cfg)
    train_cfg["data"]["test_root"] = "/definitely/not/a/dataset"
    tr, va, te = build_loaders(train_cfg, stage="train")
    assert len(tr) == 2 and not te
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg["data"]["train_root"] = eval_cfg["data"]["val_root"] = "/not/a/dataset"
    tr, va, te = build_loaders(eval_cfg, stage="eval")
    assert tr is None and va is None and len(te) == 1
