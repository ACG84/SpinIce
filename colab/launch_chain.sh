#!/usr/bin/env bash
# Create a fresh Colab GPU session, install mumax+, and start the next analysis chain detached.
#   colab/launch_chain.sh [T4|A100|L4]        (exit 2 if Colab refuses to hand out the GPU)
# The chain is ordered by value because free T4s are reclaimed after ~2 h; every table checkpoints
# after each state row, so a reclaimed VM still leaves a usable partial transitions.json.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
GPU=${1:-T4}
export COLAB_BIN=${COLAB_BIN:-$HOME/.local/bin/colab} COLAB_SESSION=${COLAB_SESSION:-gpu_$GPU}
if ! timeout 300 "$COLAB_BIN" new -s "$COLAB_SESSION" --gpu "$GPU" </dev/null 2>&1 | tee /dev/stderr | grep -q "Session READY"; then
  echo "no $GPU available"; exit 2
fi
tar -C "$ROOT" -czf /tmp/spinice.tar.gz --exclude=.git --exclude=runs --exclude='__pycache__' .
"$COLAB_BIN" upload -s "$COLAB_SESSION" /tmp/spinice.tar.gz /content/spinice.tar.gz </dev/null 2>&1 | tail -1
timeout 600 "$COLAB_BIN" exec -s "$COLAB_SESSION" --timeout 86400 -f "$ROOT/colab/remote_setup.py" </dev/null 2>&1 | grep -E "GPU|mumaxplus|SMOKE|Error" || true
C="mumaxplus/state_catalogue.py --unit single --cell-xy 5e-9 --hysteresis"
T="mumaxplus/state_catalogue.py --unit cell --cell-xy 10e-9 --fast --closure-only --transitions --trans-angle 45 --trans-amplitudes 28e-3 56e-3 4e-3"
timeout 100 "$ROOT/colab/run_remote.sh" start chain3 colab/run_batch.py \
  "$T --spacer 50e-9 --out runs/trans_cell_10_45deg_sp50" \
  "$C --spacer 70e-9 --out runs/scan_sp70" \
  "$T --out runs/trans_cell_10_45deg_4mT" \
  "$C --width 180e-9 --out runs/scan_w180" \
  "$C --width 220e-9 --out runs/scan_w220" \
  "mumaxplus/state_catalogue.py --unit cell --cell-xy 5e-9 --fast --out runs/cat_cell_5" </dev/null 2>&1 | tail -1
date -u
