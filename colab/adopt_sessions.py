"""Re-adopt orphaned Colab runtimes into the Colab CLI's local session store.

The CLI prunes a session from its local state when a request to the runtime
fails with 404/401 - which also happens when its cached runtime-proxy token
(1 h lifetime) has expired, e.g. after the keep-alive daemon died.  The VM
itself is usually still assigned (``colab sessions`` shows it as ``[?]``).
This script fetches the current assignments with a fresh token, recreates
the local records (re-attaching to the existing kernel) and restarts the
keep-alive daemons.

    <colab-cli venv python> colab/adopt_sessions.py NAME=ENDPOINT_SUBSTRING [...]
    e.g.  adopt_sessions.py spinice=gpu-t4 a100=gpu-a100
"""
import sys

import requests

from colab_cli.common import state
from colab_cli.state import SessionState
from colab_cli.commands.session import spawn_keep_alive

wanted = dict(arg.split("=", 1) for arg in sys.argv[1:])
existing = state.store.list()
for a in state.client.list_assignments():
    d = a.model_dump()
    ep = d["endpoint"]
    name = next((n for n, sub in wanted.items() if sub in ep), None)
    if name is None:
        print("skip", ep)
        continue
    info = d.get("runtime_proxy_info") or d.get("runtimeProxyInfo")
    url, token = info["url"], info["token"]
    r = requests.get(f"{url}/api/kernels", headers={"X-Colab-Runtime-Proxy-Token": token},
                     params={"colab-runtime-proxy-token": token}, timeout=30)
    r.raise_for_status()
    kernels = r.json()
    kid = kernels[0]["id"] if kernels else None
    variant = str(getattr(d.get("variant"), "name", d.get("variant")) or "DEFAULT")
    variant = {"1": "GPU", "2": "TPU"}.get(variant, variant)
    accel = str(getattr(d.get("accelerator"), "name", d.get("accelerator")) or "NONE")
    s = SessionState(name=name, token=token, url=url, endpoint=ep, variant=variant,
                     accelerator=accel, kernel_id=kid)
    if name in existing:
        state.store.remove(name)
    state.store.add(s)
    s.keep_alive_pid = spawn_keep_alive(ep, name, config_path=None)
    state.store.remove(name); state.store.add(s)
    print(f"adopted {name}: {ep} accel {s.accelerator} kernel {kid} keep-alive pid {s.keep_alive_pid}")
