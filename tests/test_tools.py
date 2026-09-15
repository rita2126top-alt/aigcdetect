import importlib.util
import json
import subprocess
import sys
import runpy
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_suite_plan(cfg, tmp_path):
    suite = module("run_suite")
    tasks = suite.plan_runs(cfg, "all", [0, 1, 2], tmp_path)
    assert len(tasks) == 21
    assert {n for n, _, _, _ in tasks} == set(suite.ABLATIONS)
    assert len({str(p) for _, _, p, _ in tasks}) == 21
    for name, _, _, c in tasks:
        assert sum(c["ablation"].values()) == (0 if name == "full" else 1)
        assert c["training"]["epochs"] == cfg["training"]["epochs"]
    with pytest.raises(ValueError):
        suite.plan_runs(cfg, "full", [0, 0])


def test_aggregation(tmp_path):
    for seed in (0, 1):
        p = tmp_path / f"full_seed{seed}/eval/summary.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"macro": {"ap": .8 + seed * .1}}))
    result = module("collect_results").collect(tmp_path)
    assert result["variants"]["full"]["metrics"]["ap"]["mean"] == pytest.approx(.85)
    with pytest.raises(FileNotFoundError):
        module("collect_results").collect(tmp_path / "absent")


@pytest.mark.parametrize("name", ["train", "eval", "infer", "eval_robustness", "check_env", "export_genimage_arrow", "prepare_genimage", "audit_data", "download_assets", "download_ojha", "run_suite", "collect_results", "run_baseline"])
def test_cli_help(name, monkeypatch, capsys):
    monkeypatch.chdir("/tmp")
    monkeypatch.setattr(sys, "argv", [str(ROOT / f"tools/{name}.py"), "--help"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(ROOT / f"tools/{name}.py"), run_name="__main__")
    assert exit_info.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_shell_syntax():
    for p in (ROOT / "scripts").glob("*.sh"):
        subprocess.run(["bash", "-n", str(p)], check=True)
