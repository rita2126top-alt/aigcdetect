"""Import-safe train/evaluate/infer/check entry points."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import torch
from PIL import Image
from .config import load_config, apply_overrides
from .runtime import prepare_config_paths, choose_device
from .io_utils import atomic_json
from .data.folder_dataset import build_loaders, build_transform, make_loader, test_directories
from .model import ClassAnchoredDynamicPPMCLIP
from .engine.train import read_checkpoint, load_checkpoint, seed_all, train, weighted_loss
from .engine.evaluate import evaluate_loader, save_eval, summarize_domains

ROOT = Path(__file__).resolve().parents[1]


def parser(description, checkpoint=False):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    if checkpoint:
        p.add_argument("--checkpoint", required=True)
    return p


def config_for(args, checkpoint=False):
    cfg = load_config(args.config)
    if checkpoint:
        saved = read_checkpoint(args.checkpoint)["config"]
        # Crucial for evaluating ablations: restore the trained architecture and
        # ablation switches; keep local paths/device from the user's current YAML.
        cfg["model"] = copy.deepcopy(saved["model"])
        cfg["ablation"] = copy.deepcopy(saved.get("ablation", {}))
    return prepare_config_paths(apply_overrides(cfg, args.set), ROOT)


def model_for(cfg, checkpoint_path=None):
    device = choose_device(cfg["runtime"].get("device", "cuda"))
    model = ClassAnchoredDynamicPPMCLIP(cfg).to(device)
    if checkpoint_path:
        load_checkpoint(checkpoint_path, model)
    return model, device


def train_main(argv=None):
    args = parser("Train the full class-anchored dynamic PPM-CLIP detector").parse_args(argv)
    cfg = config_for(args)
    seed_all(int(cfg["training"].get("seed", 0)))
    tr, val, _ = build_loaders(cfg, stage="train")
    model, device = model_for(cfg)
    print(json.dumps({"trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                      "train_images": len(tr.dataset), "val_images": len(val.dataset)}, indent=2))
    print("best checkpoint:", train(cfg, model, tr, val, device))


def eval_main(argv=None):
    p = parser("Evaluate local domains without requiring training data", checkpoint=True)
    p.add_argument("--include-val", action="store_true")
    args = p.parse_args(argv)
    cfg = config_for(args, checkpoint=True)
    _, val, tests = build_loaders(cfg, stage="eval", include_val=args.include_val)
    model, device = model_for(cfg, args.checkpoint)
    all_metrics = {}
    for name, loader in tests.items():
        metrics, predictions = evaluate_loader(model, loader, device, name)
        save_eval(cfg["paths"]["output_dir"], name, metrics, predictions)
        all_metrics[name] = metrics
    summary = summarize_domains(all_metrics)
    if val is not None:
        metrics, predictions = evaluate_loader(model, val, device, "val")
        save_eval(cfg["paths"]["output_dir"], "val", metrics, predictions)
        summary["validation"] = metrics  # never part of the cross-domain macro
    summary["checkpoint"] = str(Path(args.checkpoint).resolve())
    summary["config"] = cfg
    atomic_json(Path(cfg["paths"]["output_dir"]) / "eval/summary.json", summary)
    print(json.dumps(summary, indent=2))


def robustness_main(argv=None):
    args = parser("Full local JPEG/blur robustness evaluation", checkpoint=True).parse_args(argv)
    cfg = config_for(args, checkpoint=True)
    model, device = model_for(cfg, args.checkpoint)
    d, t = cfg["data"], cfg["training"]
    common = dict(batch_size=int(d.get("eval_batch_size", t["batch_size"])),
                  workers=int(d.get("num_workers", 8)), load_size=int(d["load_size"]),
                  image_size=int(d["image_size"]), train=False, seed=int(t["seed"]))
    conditions = [("clean", {})]
    conditions += [(f"jpeg{q}", {"jpeg_quality": int(q)}) for q in cfg["evaluation"]["jpeg_qualities"]]
    conditions += [(f"blur{float(r):g}", {"blur_radius": float(r)}) for r in cfg["evaluation"]["blur_radii"]]
    report = {}
    for condition, transform_args in conditions:
        results = {}
        for name, root in test_directories(cfg):
            loader = make_loader(root, **common, **transform_args)
            metrics, predictions = evaluate_loader(model, loader, device, f"{name}__{condition}")
            save_eval(cfg["paths"]["output_dir"], f"{name}__{condition}", metrics, predictions)
            results[name] = metrics
        report[condition] = summarize_domains(results)
    atomic_json(Path(cfg["paths"]["output_dir"]) / "eval/robustness_summary.json", report)
    print(json.dumps(report, indent=2))


def infer_main(argv=None):
    p = parser("Single-image local inference", checkpoint=True)
    p.add_argument("image")
    args = p.parse_args(argv)
    cfg = config_for(args, checkpoint=True)
    model, device = model_for(cfg, args.checkpoint)
    model.eval()
    transform = build_transform(False, cfg["data"]["load_size"], cfg["data"]["image_size"])
    with Image.open(args.image) as image:
        x = transform(image.convert("RGB"))[None].to(device)
    with torch.no_grad():
        out = model(x, mode="test")
    prob = out["probabilities"][0].cpu()
    print(json.dumps({"real": float(prob[0]), "fake": float(prob[1]),
                      "prediction": "fake" if prob[1] >= 0.5 else "real",
                      "private_length": int(out["selected_private_length"][0])}))


def check_main(argv=None):
    p = parser("Fail-fast local environment/data checks; optional actual model step")
    p.add_argument("--stage", choices=["train", "eval", "infer", "all"], default="all")
    p.add_argument("--forward", action="store_true")
    p.add_argument("--backward", action="store_true")
    args = p.parse_args(argv)
    cfg = config_for(args)
    device = choose_device(cfg["runtime"]["device"])
    from .model.detector import _ensure_upstream
    _ensure_upstream(cfg["paths"]["ppmclip_root"])
    if not Path(cfg["paths"]["clip_model"]).is_file():
        raise FileNotFoundError(cfg["paths"]["clip_model"])
    counts = {}
    if args.stage != "infer":
        tr, val, tests = build_loaders(cfg, stage=args.stage)
        counts = {k: len(v.dataset) for k, v in {"train": tr, "val": val, **tests}.items() if v is not None}
    report = {"torch": str(torch.__version__), "cuda_available": torch.cuda.is_available(),
              "device": str(device), "dataset_counts": counts, "status": "passed"}
    if device.type == "cuda":
        report["gpu"] = torch.cuda.get_device_name(device)
    if args.forward or args.backward:
        model, device = model_for(cfg)
        size = int(cfg["data"]["image_size"])
        x = torch.randn(1, 3, size, size, device=device)
        model.eval()
        with torch.no_grad():
            result = model(x)
        assert torch.isfinite(result["probabilities"]).all()
        report["forward"] = "passed (random image, local CLIP checkpoint)"
        if args.backward:
            model.train()
            loss = weighted_loss(model(x, torch.zeros(1, dtype=torch.long, device=device), mode="train")["losses"], cfg["loss"])
            loss.backward()
            grads = [v.grad for v in model.parameters() if v.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            report["backward"] = "passed (FP32)"
    print(json.dumps(report, indent=2))
