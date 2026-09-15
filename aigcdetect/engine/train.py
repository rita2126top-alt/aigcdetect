from __future__ import annotations
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from .evaluate import evaluate_loader


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, model, optimizer, epoch, cfg, best_metric):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                "config": cfg, "best_metric": best_metric}, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def train(cfg, model, train_loader, val_loader, device):
    t = cfg["training"]; losses_w = cfg["loss"]
    lora, other = model.trainable_parameter_groups()
    optimizer = AdamW([
        {"params": other, "lr": float(t["lr_new"])},
        {"params": lora, "lr": float(t["lr_lora"])},
    ], weight_decay=float(t.get("weight_decay", 1e-4)))
    epochs = int(t["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    amp = bool(t.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    out_dir = Path(cfg["paths"]["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, best = 0, -1.0
    resume = t.get("resume")
    if resume:
        ck = load_checkpoint(resume, model, optimizer, map_location=device)
        start_epoch = int(ck["epoch"]) + 1; best = float(ck.get("best_metric", -1))

    for epoch in range(start_epoch, epochs):
        model.train(); running = {}
        bar = tqdm(train_loader, desc=f"train {epoch+1}/{epochs}")
        for images, labels, _ in bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                out = model(images, labels, mode="train", epoch=epoch, max_epochs=epochs)
                ls = out["losses"]
                total = (ls["cls"] + float(losses_w["rec"]) * ls["rec"] + float(losses_w["kl"]) * ls["kl"]
                         + float(losses_w["patch"]) * ls["patch"] + float(losses_w["anchor"]) * ls["anchor"]
                         + float(losses_w["len"]) * ls["len"])
            scaler.scale(total).backward()
            if t.get("grad_clip"):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), float(t["grad_clip"]))
            scaler.step(optimizer); scaler.update()
            vals = {"total": float(total.detach()), **{k: float(v.detach()) for k, v in ls.items()}}
            for k, v in vals.items(): running[k] = running.get(k, 0.0) + v
            bar.set_postfix(loss=f"{vals['total']:.4f}")
        scheduler.step()
        val_metrics, _ = evaluate_loader(model, val_loader, device, desc="val")
        score = val_metrics.get(t.get("selection_metric", "ap"), val_metrics["ap"])
        with open(out_dir / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "val": val_metrics,
                                "train": {k: v / max(len(train_loader),1) for k,v in running.items()}}) + "\n")
        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, cfg, best)
        if score > best:
            best = score; save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, cfg, best)
        print(f"epoch={epoch+1} val={val_metrics} best={best:.6f}")
    return str(out_dir / "best.pt")
