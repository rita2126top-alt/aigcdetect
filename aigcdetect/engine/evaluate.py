from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from .metrics import binary_metrics

@torch.no_grad()
def evaluate_loader(model, loader, device, desc="eval"):
    model.eval()
    labels, scores, paths, lengths = [], [], [], []
    for images, y, p in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        out = model(images, mode="test")
        labels.extend(y.numpy().tolist())
        scores.extend(out["probabilities"][:, 1].detach().cpu().numpy().tolist())
        lengths.extend(out["selected_private_length"].detach().cpu().numpy().tolist())
        paths.extend(list(p))
    m = binary_metrics(labels, scores)
    m["mean_private_length"] = float(np.mean(lengths)) if lengths else float("nan")
    return m, {"label": labels, "fake_probability": scores, "private_length": lengths, "path": paths}


def save_eval(output_dir, name, metrics, predictions):
    out = Path(output_dir) / "eval"; out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}_metrics.json", "w", encoding="utf-8") as f: json.dump(metrics, f, indent=2)
    import csv
    with open(out / f"{name}_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(predictions.keys())); w.writeheader()
        for row in zip(*predictions.values()): w.writerow(dict(zip(predictions.keys(), row)))
