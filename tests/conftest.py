"""Small random CLIP fixtures: actual upstream code, not pretrained accuracy tests."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pytest
import torch
import numpy as np
from PIL import Image
from aigcdetect.config import load_config
from aigcdetect.runtime import prepare_config_paths

@pytest.fixture(scope="session")
def clip_path(tmp_path_factory):
    torch.set_num_threads(2)
    from aigcdetect.model.detector import _ensure_upstream
    _ensure_upstream(str(ROOT / "PPM_CLIP"))
    from clip.model import CLIP
    torch.manual_seed(17)
    model = CLIP(embed_dim=32, image_resolution=28, vision_layers=24, vision_width=64,
                 vision_patch_size=14, context_length=77, vocab_size=49408,
                 transformer_width=64, transformer_heads=1, transformer_layers=2)
    path = tmp_path_factory.mktemp("clip") / "random-test-only.pt"
    torch.save(model.state_dict(), path)
    return path

@pytest.fixture
def cfg(tmp_path, clip_path):
    c = load_config(ROOT / "configs/default.yaml")
    c["paths"].update(ppmclip_root=str(ROOT / "PPM_CLIP"), clip_model=str(clip_path), output_dir=str(tmp_path / "out"))
    c["runtime"]["device"] = "cpu"
    c["data"].update(image_size=28, load_size=28, num_workers=0, eval_batch_size=2, test_sets=["domain"])
    c["model"]["flow"]["hidden_dim"] = 32
    c["model"]["cross_attention"]["dim"] = 32
    c["model"]["prompt"].update(router_hidden=32, image_chunk_size=1, text_chunk_size=8)
    c["training"].update(epochs=2, batch_size=3, accumulate_steps=2, amp=False)
    rng = np.random.default_rng(53)
    for name, count in (("train", 3), ("val", 1), ("test/domain", 1)):
        for label in ("0_real", "1_fake"):
            base = tmp_path / "data" / name / label
            base.mkdir(parents=True)
            for i in range(count):
                Image.fromarray(rng.integers(0, 256, (28, 28, 3), dtype=np.uint8)).save(base / f"{i}.png")
    c["data"].update(train_root=str(tmp_path / "data/train"), val_root=str(tmp_path / "data/val"),
                      test_root=str(tmp_path / "data/test"))
    return prepare_config_paths(c, ROOT)
