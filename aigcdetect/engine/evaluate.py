from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from .metrics import binary_metrics
from ..io_utils import atomic_json


@torch.no_grad()
def evaluate_loader(model, loader, device, desc="eval"):
    model.eval()
    labels, scores, paths, lengths = [], [], [], []
    for images, y, p in tqdm(loader, desc=desc, leave=False):
        out = model(images.to(device, non_blocking=True), mode="test")
        labels.extend(y.tolist())
        scores.extend(out["probabilities"][:, 1].float().cpu().tolist())
        lengths.extend(out["selected_private_length"].cpu().tolist())
        paths.extend(p)
    metrics = binary_metrics(labels, scores)
    metrics["mean_private_length"] = float(np.mean(lengths))
    metrics["length_counts"] = {str(x): lengths.count(x) for x in sorted(set(lengths))}
    return metrics, {"label": labels, "fake_probability": scores, "private_length": lengths, "path": paths}


def save_eval(output_dir, name, metrics, predictions):
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Evaluation name must be a single safe path component")
    out = Path(output_dir) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / f"{name}_metrics.json", metrics)
    with open(out / f"{name}_predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions))
        writer.writeheader()
        for row in zip(*predictions.values()):
            writer.writerow(dict(zip(predictions, row)))


def summarize_domains(results):
    keys = ("acc", "ap", "auc", "f1", "real_acc", "fake_acc", "balanced_acc", "mean_private_length")
    macro = {key: float(np.mean([m[key] for m in results.values() if m.get(key) is not None]))
             for key in keys if any(m.get(key) is not None for m in results.values())}
    return {"domains": results, "macro": macro, "num_domains": len(results),
            "aggregation": "unweighted mean over domains; not a pooled-image metric"}
