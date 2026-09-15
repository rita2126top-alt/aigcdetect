"""Reproducible syntax, configuration, upstream-integrity and CPU integration audit."""
import argparse
import ast
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="validation")
    p.add_argument("--static-only", action="store_true")
    args = p.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = [x for folder in ("aigcdetect", "tools", "tests") for x in (ROOT / folder).rglob("*.py")]
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in (ROOT / "scripts").glob("*.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)
    from aigcdetect.config import load_config, validate_config
    for path in (ROOT / "configs").glob("*.yaml"):
        validate_config(load_config(path))
    import yaml
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        yaml.safe_load(path.read_text())
    subprocess.run([sys.executable, str(ROOT / "tools/verify_upstream.py")], check=True)
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "parsed_python_files": len(files), "static_checks": "passed", "gpu_training_tested": False,
              "scope": "Actual upstream tiny random CLIP CPU integration; not pretrained detection accuracy"}
    if not args.static_only:
        import torch
        report.update(torch=str(torch.__version__), cuda_available=torch.cuda.is_available())
        cmd = [sys.executable, "-m", "pytest", "-q", "tests", f"--junitxml={output / 'pytest.xml'}"]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
        run = subprocess.run(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (output / "pytest.log").write_text(run.stdout)
        print(run.stdout)
        if run.returncode:
            raise SystemExit(run.returncode)
        from xml.etree import ElementTree as ET
        suite = ET.parse(output / "pytest.xml").getroot()
        report["tests"] = sum(int(x.get("tests", 0)) for x in suite.iter("testsuite"))
        report["cpu_tests"] = "passed"
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
