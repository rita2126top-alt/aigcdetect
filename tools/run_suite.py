"""Execute full-length experiment grids; never substitute smoke runs for results."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import copy
import json
import shlex
import subprocess
import yaml
from aigcdetect.cli import parser, config_for
from aigcdetect.io_utils import atomic_json

ABLATIONS = {
    "full": {},
    "no_cross_attention": {"disable_cross_attention": True},
    "fixed_length": {"disable_dynamic_length": True},
    "no_probability": {"disable_probabilistic_prompt": True},
    "no_dct": {"disable_dct_loss": True},
    "fixed_anchors": {"freeze_class_anchors": True},
    "no_lora": {"disable_lora": True},
}


def plan_runs(cfg, suite, seeds, output=None):
    if suite not in ("full", "ablations", "all"):
        raise ValueError("suite must be full, ablations or all")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    base = Path(output or cfg["paths"]["suite_output"]).resolve()
    variants = ABLATIONS if suite in ("ablations", "all") else {"full": {}}
    tasks = []
    for seed in seeds:
        for name, switches in variants.items():
            c = copy.deepcopy(cfg)
            c["ablation"] = {k: False for k in cfg["ablation"]}
            c["ablation"].update(switches)
            c["training"].update(seed=seed, resume=None)
            directory = base / f"{name}_seed{seed}"
            c["paths"]["output_dir"] = str(directory)
            tasks.append((name, seed, directory, c))
    return tasks


def main(argv=None):
    p = parser("Run full 100-epoch defaults / 7-variant ablations / repeated seeds")
    p.add_argument("--suite", choices=["full", "ablations", "all"], default="full")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--output", help="Suite output directory; default paths.suite_output in YAML")
    p.add_argument("--stage", choices=["all", "train", "eval"], default="all")
    p.add_argument("--resume", action="store_true", help="Resume last.pt without changing the original epoch budget")
    p.add_argument("--dry-run", action="store_true", help="Print exact plan only; does not generate results or launch jobs")
    args = p.parse_args(argv)
    cfg = config_for(args)
    seeds = [int(v.strip()) for v in args.seeds.split(",") if v.strip()]
    tasks = plan_runs(cfg, args.suite, seeds, args.output)
    print(json.dumps({"suite": args.suite, "runs": len(tasks), "epochs_per_run": cfg["training"]["epochs"],
                      "seeds": seeds, "robustness": args.suite == "all"}, indent=2))
    for name, seed, directory, c in tasks:
        config_path = directory / "run.yaml"
        if args.resume and (directory / "last.pt").exists():
            c["training"]["resume"] = str(directory / "last.pt")
        commands = []
        if args.stage in ("all", "train"):
            commands.append([sys.executable, str(ROOT / "tools/train.py"), "--config", str(config_path)])
        if args.stage in ("all", "eval"):
            common = ["--config", str(config_path), "--checkpoint", str(directory / "best.pt")]
            commands.append([sys.executable, str(ROOT / "tools/eval.py"), *common])
            if args.suite == "all":
                commands.append([sys.executable, str(ROOT / "tools/eval_robustness.py"), *common])
        for cmd in commands:
            print(shlex.join(cmd), flush=True)
        if args.dry_run:
            continue
        directory.mkdir(parents=True, exist_ok=True)
        if config_path.exists():
            previous = yaml.safe_load(config_path.read_text())
            previous["training"]["resume"] = c["training"]["resume"]
            if previous != c:
                raise ValueError(f"Refusing to overwrite a different experiment: {directory}")
        config_path.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
        for cmd in commands:
            subprocess.run(cmd, check=True, cwd=ROOT)
    if not args.dry_run and args.stage in ("all", "eval"):
        from collect_results import collect
        base = Path(args.output or cfg["paths"]["suite_output"]).resolve()
        atomic_json(base / "aggregate.json", collect(base))

if __name__ == "__main__":
    main()
