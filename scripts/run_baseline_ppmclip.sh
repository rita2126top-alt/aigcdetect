#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${CLIP_MODEL:-$ROOT/weights/clip/ViT-L-14.pt}"
if [[ ! -f "$MODEL" ]]; then
  echo "Missing local CLIP model: $MODEL" >&2
  echo "Run: bash scripts/download_clip.sh" >&2
  exit 1
fi
mkdir -p "$HOME/.cache/clip" "$ROOT/PPM_CLIP/_runtime"
ln -sf "$MODEL" "$HOME/.cache/clip/ViT-L-14.pt"
cd "$ROOT/PPM_CLIP/_runtime"
python ../main.py "$@"
