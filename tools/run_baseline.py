"""Launch untouched upstream code with process-local paths. Not a controlled baseline protocol."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os
import runpy
from aigcdetect.cli import parser, config_for


def main():
    p = parser(__doc__)
    p.add_argument("--action", choices=["train", "eval"], default="train")
    p.add_argument("--dataset", choices=["genimage", "ojha"], default="genimage")
    p.add_argument("--checkpoint", help="Upstream state-dict checkpoint, not the new method adapter")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    c = config_for(args)
    d, t = c["data"], c["training"]
    if isinstance(d["test_sets"], dict):
        raise ValueError("The original baseline requires list-valued test_sets")
    if d.get("train_classes") and args.action == "train":
        raise ValueError("Original baseline cannot filter classes. Prepare a separate 4-class directory and set data.train_classes=[] explicitly.")
    if int(d["num_workers"]) < 1:
        raise ValueError("The original baseline requires num_workers >= 1")
    if args.action == "eval" and not args.checkpoint:
        raise ValueError("--checkpoint is required for baseline evaluation")
    device = c["runtime"]["device"]
    gpu = "-1" if device == "cpu" else device.split(":")[-1] if ":" in device else "0"
    forwarded = ["--dataset", args.dataset, "--gpu", gpu, "--backbone", c["model"]["backbone"],
        "--num_workers", str(d["num_workers"]), "--loadSize", str(d["load_size"]),
        "--epochs", str(t["epochs"]), "--batch_size", str(t["batch_size"]), "--lr", str(t["lr_new"])]
    if args.checkpoint:
        forwarded += ["--ckpt_path", str(Path(args.checkpoint).resolve())]
    output = Path(c["paths"]["output_dir"]) / "original_baseline"
    print("WARNING: unchanged upstream training uses test scores for early stopping after val_acc>=0.9. Do not present it as a leakage-free controlled comparison.")
    print("Original entry:", args.action, "arguments:", forwarded, "output:", output)
    if args.dry_run:
        return
    from aigcdetect.model.detector import _ensure_upstream, _load_local_clip
    import torch
    _ensure_upstream(c["paths"]["ppmclip_root"])
    import clip
    import data_loading
    data_loading.DATASETS[args.dataset] = {"train_root": d["train_root"], "val_root": d["val_root"],
                                          "test_root": d["test_root"], "vals": d["test_sets"]}
    # The baseline asks for a named model; route it strictly to the specified local file.
    def local_load(name, device="cpu", jit=False, **kwargs):
        return _load_local_clip(c["paths"]["clip_model"]).to(device), None
    clip.load = local_load
    # PyTorch 2.6+ removed verbose; drop only this cosmetic deprecated argument.
    original = torch.optim.lr_scheduler.ReduceLROnPlateau
    def plateau(*a, **kw):
        kw.pop("verbose", None)
        return original(*a, **kw)
    torch.optim.lr_scheduler.ReduceLROnPlateau = plateau
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    entry = Path(c["paths"]["ppmclip_root"]) / ("main.py" if args.action == "train" else "test.py")
    sys.argv = [str(entry), *forwarded]
    runpy.run_path(str(entry), run_name="__main__")

if __name__ == "__main__":
    main()
