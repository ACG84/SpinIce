#!/usr/bin/env bash
# Drive the micromagnetic runs on a Google Colab GPU from your terminal with the
# Colab CLI (uv tool install google-colab-cli; `colab sessions` once to log in).
#
#   colab/run_remote.sh setup [T4]                 # new session, upload repo, install mumax+, smoke test
#   colab/run_remote.sh sync                       # re-upload the repo after local edits
#   colab/run_remote.sh start NAME SCRIPT [args]   # launch a script detached on the VM (nohup), log -> /content/logs/NAME.log
#   colab/run_remote.sh log NAME [N]               # tail the last N lines of that log
#   colab/run_remote.sh wait NAME                  # block until the job exits, printing new log lines
#   colab/run_remote.sh fetch REMOTE LOCAL         # download a result file
#   colab/run_remote.sh py "<python>"              # run python in the kernel (streams output)
#   colab/run_remote.sh stop                       # release the VM (it is billed until you do!)
#   colab/run_remote.sh adopt NAME=ENDPOINT_SUBSTR # re-attach orphaned VMs ("[?]" in `colab sessions`)
#
# Example:
#   colab/run_remote.sh setup T4
#   colab/run_remote.sh start sweep mumaxplus/run_field_sweep.py --b-start=-30e-3 --b-stop 60e-3 --out runs/sweep
#   colab/run_remote.sh wait sweep && colab/run_remote.sh fetch /content/SpinIce/runs/sweep/sweep.npz runs/colab/sweep.npz
#   colab/run_remote.sh start mg10 mumaxplus/run_reservoir.py --task mackey_glass_10 --n 200 --out runs/mg10
#
# Long jobs are started with nohup because `colab exec` holds a WebSocket for
# the duration of a cell and the kernel executes cells serially.
set -euo pipefail
SESSION=${COLAB_SESSION:-spinice}
COLAB=${COLAB_BIN:-colab}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cmd=${1:-help}; shift || true

py() { printf '%s\n' "$1" | "$COLAB" exec -s "$SESSION" --timeout "${COLAB_TIMEOUT:-86400}"; }
upload_repo() {
  tar -C "$ROOT" -czf /tmp/spinice.tar.gz --exclude=.git --exclude=runs --exclude='__pycache__' .
  "$COLAB" upload -s "$SESSION" /tmp/spinice.tar.gz /content/spinice.tar.gz
}

case "$cmd" in
  setup)
    "$COLAB" new -s "$SESSION" --gpu "${1:-T4}"
    upload_repo
    "$COLAB" exec -s "$SESSION" -f "$ROOT/colab/remote_setup.py"
    ;;
  sync)
    upload_repo
    py "import tarfile; tarfile.open('/content/spinice.tar.gz').extractall('/content/SpinIce'); print('synced')"
    ;;
  start)
    NAME=$1; SCRIPT=$2; shift 2
    ARGJSON=$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1:]))' "$@")
    py "import subprocess, sys, os
os.makedirs('/content/logs', exist_ok=True)
log = open('/content/logs/$NAME.log', 'w')
p = subprocess.Popen([sys.executable, '-u', '$SCRIPT', *$ARGJSON], cwd='/content/SpinIce',
                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
open('/content/logs/$NAME.pid', 'w').write(str(p.pid))
print('started $NAME pid', p.pid)"
    ;;
  log)
    NAME=$1; N=${2:-30}
    py "import os, subprocess
pid = int(open('/content/logs/$NAME.pid').read())
alive = os.path.exists(f'/proc/{pid}') and 'zombie' not in open(f'/proc/{pid}/status').read()
print('=== $NAME', 'RUNNING' if alive else 'FINISHED', '===')
print(subprocess.run(['tail', '-n', '$N', '/content/logs/$NAME.log'], capture_output=True, text=True).stdout)"
    ;;
  wait)
    NAME=$1
    py "import os, time, subprocess
pid = int(open('/content/logs/$NAME.pid').read())
seen = 0
def alive():
    try: return 'zombie' not in open(f'/proc/{pid}/status').read()
    except FileNotFoundError: return False
while True:
    lines = open('/content/logs/$NAME.log').read().splitlines()
    for l in lines[seen:]: print(l, flush=True)
    seen = len(lines)
    if not alive(): break
    time.sleep(20)
print('=== $NAME finished ===')"
    ;;
  fetch) "$COLAB" download -s "$SESSION" "$1" "$2" ;;
  adopt)  # the CLI prunes sessions whose 1 h runtime token expired; the VM is usually still there
    PY=$(dirname "$(readlink -f "$(command -v "$COLAB")")")/python
    "$PY" "$ROOT/colab/adopt_sessions.py" "$@" ;;
  py) py "$1" ;;
  status) "$COLAB" status -s "$SESSION" ;;
  stop) "$COLAB" stop -s "$SESSION" ;;
  *) sed -n 2,20p "$0" ;;
esac
