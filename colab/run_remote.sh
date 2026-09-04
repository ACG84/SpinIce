#!/usr/bin/env bash
# Drive the micromagnetic runs on a Google Colab GPU from your terminal with the
# Colab CLI (pip/uv install google-colab-cli; `colab sessions` to log in).
#
#   colab/run_remote.sh setup   [T4]         # new session + upload repo + install mumax+ + smoke test
#   colab/run_remote.sh sweep   [args...]    # mumaxplus/run_field_sweep.py on the VM, results -> runs/colab/sweep
#   colab/run_remote.sh reservoir [args...]  # mumaxplus/run_reservoir.py on the VM, results -> runs/colab/<out>
#   colab/run_remote.sh shell "<python>"     # arbitrary python in the kernel
#   colab/run_remote.sh stop                 # release the VM (it is billed until you do!)
set -euo pipefail
SESSION=${COLAB_SESSION:-spinice}
COLAB=${COLAB_BIN:-colab}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cmd=${1:-help}; shift || true

remote_py() {   # run a python snippet in the persistent kernel, streaming output
  printf '%s\n' "$1" | "$COLAB" exec -s "$SESSION"
}
run_script() {  # run one of our scripts on the VM as a subprocess (output streams back)
  local script=$1; shift
  local argjson; argjson=$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1:]))' "$@")
  remote_py "import subprocess, sys
r = subprocess.run([sys.executable, '$script', *$argjson], cwd='/content/SpinIce')
print('remote exit code', r.returncode)
if r.returncode: raise RuntimeError('remote script failed')"
}

case "$cmd" in
  setup)
    GPU=${1:-T4}
    "$COLAB" new -s "$SESSION" --gpu "$GPU"
    tar -C "$ROOT" -czf /tmp/spinice.tar.gz --exclude=.git --exclude=runs --exclude='__pycache__' .
    "$COLAB" upload -s "$SESSION" /tmp/spinice.tar.gz /content/spinice.tar.gz
    "$COLAB" exec -s "$SESSION" -f "$ROOT/colab/remote_setup.py"
    ;;
  sync)   # re-upload the repo after local edits
    tar -C "$ROOT" -czf /tmp/spinice.tar.gz --exclude=.git --exclude=runs --exclude='__pycache__' .
    "$COLAB" upload -s "$SESSION" /tmp/spinice.tar.gz /content/spinice.tar.gz
    remote_py "import tarfile; tarfile.open('/content/spinice.tar.gz').extractall('/content/SpinIce'); print('synced')"
    ;;
  sweep)
    OUT=runs/sweep; for a in "$@"; do case $a in --out=*) OUT=${a#--out=};; esac; done
    run_script mumaxplus/run_field_sweep.py "$@"
    mkdir -p "$ROOT/runs/colab"; "$COLAB" download -s "$SESSION" "/content/SpinIce/$OUT/sweep.npz" "$ROOT/runs/colab/$(basename "$OUT")_sweep.npz"
    ;;
  reservoir)
    OUT=runs/reservoir; for a in "$@"; do case $a in --out=*) OUT=${a#--out=};; esac; done
    run_script mumaxplus/run_reservoir.py "$@"
    mkdir -p "$ROOT/runs/colab"; "$COLAB" download -s "$SESSION" "/content/SpinIce/$OUT/spectra.npz" "$ROOT/runs/colab/$(basename "$OUT")_spectra.npz"
    ;;
  download) "$COLAB" download -s "$SESSION" "$1" "$2" ;;
  shell) remote_py "$1" ;;
  status) "$COLAB" status -s "$SESSION" ;;
  stop) "$COLAB" stop -s "$SESSION" ;;
  *) sed -n 2,12p "$0" ;;
esac
