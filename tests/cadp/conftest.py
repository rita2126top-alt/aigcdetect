import os
import sys
from pathlib import Path
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import pytest
from cadp.config import load_config

torch.set_num_threads(2)

@pytest.fixture
def cfg():
    return load_config(overrides=['model.tiny=true','model.attention_dim=32','model.hidden_dim=32',
        'model.text_chunk_size=8','train.device=cpu','train.amp=off','data.workers=0',
        'data.min_size=28','train.batch_size=2','train.accumulation_steps=3','train.epochs=2'])
