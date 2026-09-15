#!/usr/bin/env bash
# Apply this delivery to a clean clone; optional explicit non-force push to main.
set -euo pipefail
usage() { echo "Usage: $0 /absolute/delivery.patch /absolute/aigcdetect-clone [--check|--apply|--push]" >&2; }
[[ $# -ge 2 && $# -le 3 ]] || { usage; exit 2; }
PATCH="$(realpath "$1")"
REPO="$(realpath "$2")"
MODE="${3:---check}"
case "$MODE" in --check|--apply|--push) ;; *) usage; exit 2;; esac
[[ -f "$PATCH" ]] || { echo 'Patch file missing' >&2; exit 2; }
cd "$REPO"
[[ "$(git rev-parse --show-toplevel)" == "$REPO" ]] || { echo 'Use the repository root' >&2; exit 2; }
ORIGIN="$(git remote get-url origin)"
case "$ORIGIN" in
  https://github.com/rita2126top-alt/aigcdetect|https://github.com/rita2126top-alt/aigcdetect.git|git@github.com:rita2126top-alt/aigcdetect.git|ssh://git@github.com/rita2126top-alt/aigcdetect.git) ;;
  *) echo 'Refusing an origin other than rita2126top-alt/aigcdetect' >&2; exit 2;;
esac
[[ -z "$(git status --porcelain)" ]] || { echo 'Working tree/index is not clean; use a fresh clone. Nothing was overwritten.' >&2; exit 2; }
[[ "$(git branch --show-current)" == main ]] || { echo 'Use a clean main branch clone' >&2; exit 2; }
if [[ "$MODE" == --push ]]; then
  git var GIT_AUTHOR_IDENT >/dev/null
  git var GIT_COMMITTER_IDENT >/dev/null
  git fetch origin main
  git merge --ff-only origin/main
fi
git apply --check --index "$PATCH"
if [[ "$MODE" == --check ]]; then
  echo 'Patch applies cleanly. No source changes, commit or push performed.'
  exit 0
fi
# The repository already pins the upstream SHA; do not update it to upstream latest.
git submodule update --init --recursive
git apply --index "$PATCH"
python - <<'PY'
import hashlib, json
from pathlib import Path
p=json.loads(Path('provenance/upstream.json').read_text())
root=Path('PPM_CLIP')
bad=[name for name,sha in p['sha256'].items() if not (root/name).is_file() or hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha]
if bad: raise SystemExit('Upstream integrity failed: '+repr(bad))
print('Original upstream files verified:',len(p['sha256']))
PY
git diff --cached --check
git diff --cached --stat
if [[ "$MODE" == --push ]]; then
  git commit -m "Complete CADP-CLIP implementation, full experiments, documentation and verified CPU tests"
  git push origin HEAD:main
  git rev-parse HEAD
else
  echo 'Patch applied and staged, not committed or pushed. Review git diff --cached.'
fi
