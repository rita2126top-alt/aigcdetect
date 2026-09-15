from __future__ import annotations
import json
import math
import os
import random
import warnings
from pathlib import Path
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from .evaluate import evaluate_loader
from ..io_utils import atomic_json


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def save_checkpoint(path, model, optimizer, epoch, cfg, best_metric, scheduler=None, scaler=None):
    # Save new modules and LoRA only. The original CLIP is loaded locally and
    # checked by SHA256 on restore, avoiding a frozen ~GB backbone every epoch.
    state = {k: v for k, v in model.state_dict().items()
             if not k.startswith("clip_model.") or "lora_" in k}
    obj = {"format": "adapter_v1", "model": state, "epoch": epoch, "config": cfg,
           "best_metric": best_metric, "optimizer": optimizer.state_dict(), "rng": rng_state(),
           "clip_sha256": getattr(model, "clip_sha256", None),
           "scheduler": scheduler.state_dict() if scheduler else None,
           "scaler": scaler.state_dict() if scaler else None}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def read_checkpoint(path):
    # Plain tensors and primitive containers only. Never fall back to unsafe pickle.
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ck, dict) or "model" not in ck or "config" not in ck:
        raise ValueError(f"Not an AIGCDetect training checkpoint: {path}")
    return ck


def load_checkpoint(path, model, optimizer=None, map_location="cpu", scheduler=None, scaler=None, restore_random=False):
    ck = read_checkpoint(path)
    expected = ck.get("clip_sha256")
    if expected and expected != getattr(model, "clip_sha256", None):
        raise ValueError("Local CLIP checkpoint SHA256 differs from the training backbone")
    if ck.get("format") == "adapter_v1":
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        bad = [k for k in missing if not k.startswith("clip_model.") or "lora_" in k]
        if bad or unexpected:
            raise ValueError(f"Incompatible adapter checkpoint: missing={bad}, unexpected={unexpected}")
    else:
        model.load_state_dict(ck["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(ck["optimizer"])
        if not ck.get("scheduler"):
            warnings.warn("Legacy checkpoint lacks scheduler/RNG state; resume is not exact")
    if scheduler is not None and ck.get("scheduler"):
        scheduler.load_state_dict(ck["scheduler"])
    if scaler is not None and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    if restore_random and ck.get("rng"):
        restore_rng(ck["rng"])
    return ck


def weighted_loss(losses, weights):
    return losses["cls"] + sum(float(weights[k]) * losses[k] for k in ("rec", "kl", "patch", "anchor", "len"))


def train(cfg, model, train_loader, val_loader, device, stop_after_epoch=None):
    t = cfg["training"]
    lora, other = model.trainable_parameter_groups()
    optimizer = AdamW([{"params": other, "lr": float(t["lr_new"])},
                       {"params": lora, "lr": float(t["lr_lora"])}],
                      weight_decay=float(t.get("weight_decay", 1e-4)))
    epochs, accumulation = int(t["epochs"]), int(t.get("accumulate_steps", 1))
    if epochs < 1 or accumulation < 1 or len(train_loader) == 0:
        raise ValueError("Training requires positive epochs/accumulation and a nonempty loader")
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    amp = bool(t.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "last.pt").exists() and not t.get("resume"):
        raise FileExistsError(f"Existing run in {out_dir}; resume it or choose a new output_dir")
    atomic_json(out_dir / "resolved_config.json", cfg)
    start_epoch, best = 0, -1.0
    if t.get("resume"):
        ck = load_checkpoint(t["resume"], model, optimizer, scheduler=scheduler, scaler=scaler, restore_random=True)
        old = ck["config"]
        for key in ("epochs", "lr_new", "lr_lora", "weight_decay", "batch_size", "accumulate_steps"):
            if old["training"].get(key, 1 if key == "accumulate_steps" else None) != t.get(key, 1 if key == "accumulate_steps" else None):
                raise ValueError(f"Exact resume requires unchanged training.{key}; use a separate run for a new schedule")
        if old["model"] != cfg["model"] or old.get("ablation", {}) != cfg.get("ablation", {}):
            raise ValueError("Resume model/ablation config differs from checkpoint")
        start_epoch, best = int(ck["epoch"]) + 1, float(ck.get("best_metric", -1))
    if start_epoch >= epochs:
        return str(out_dir / "best.pt")
    for epoch in range(start_epoch, epochs):
        if train_loader.generator is not None:
            train_loader.generator.manual_seed(int(t.get("seed", 0)) + epoch)
        model.train()
        running, count = {}, 0
        optimizer.zero_grad(set_to_none=True)
        bar = tqdm(train_loader, desc=f"train {epoch+1}/{epochs}")
        for step, (images, labels, _) in enumerate(bar):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            # Last partial accumulation group is normalized by actual samples.
            group_start = (step // accumulation) * accumulation
            group_end = min(group_start + accumulation, len(train_loader))
            group_samples = min(group_end * train_loader.batch_size, len(train_loader.dataset)) - group_start * train_loader.batch_size
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                losses = model(images, labels, mode="train", epoch=epoch, max_epochs=epochs)["losses"]
                total = weighted_loss(losses, cfg["loss"])
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch}, step={step}; aborting without saving bad weights")
            scaler.scale(total * (len(images) / group_samples)).backward()
            if step + 1 == group_end:
                scaler.unscale_(optimizer)
                if t.get("grad_clip"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(t["grad_clip"]), error_if_nonfinite=not amp)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            vals = {"total": float(total.detach()), **{k: float(v.detach()) for k, v in losses.items()}}
            for k, value in vals.items():
                running[k] = running.get(k, 0.0) + value * len(images)
            count += len(images)
            bar.set_postfix(loss=f"{vals['total']:.4f}")
        scheduler.step()
        metrics, _ = evaluate_loader(model, val_loader, device, "val")
        score = metrics.get(t.get("selection_metric", "ap"))
        if score is None or not math.isfinite(score):
            raise ValueError("Selection metric is unavailable/non-finite; validation needs both classes")
        improved = score > best
        best = max(best, score)
        with open(out_dir / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "val": metrics, "train": {k: v / count for k, v in running.items()},
                                "lr_next": scheduler.get_last_lr(), "best_metric": best}, allow_nan=False) + "\n")
        args = (model, optimizer, epoch, cfg, best, scheduler, scaler)
        save_checkpoint(out_dir / "last.pt", *args)
        if improved:
            save_checkpoint(out_dir / "best.pt", *args)
        print(f"epoch={epoch+1} val={metrics} best={best:.6f}")
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            break
    return str(out_dir / "best.pt")
