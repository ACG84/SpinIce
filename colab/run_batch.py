"""Run several of our scripts sequentially on the VM (use with run_remote.sh start).

    python colab/run_batch.py "mumaxplus/run_field_sweep.py --seed 0 ..." "mumaxplus/run_field_sweep.py --seed 1 ..."
"""
import os, shlex, subprocess, sys, time
cmds = sys.argv[1:]
if cmds and cmds[0] == "--after":          # wait for another detached job (its .pid file) to exit
    pidfile, cmds = cmds[1], cmds[2:]
    try:
        pid = int(open(pidfile).read())
        while os.path.exists(f"/proc/{pid}") and "zombie" not in open(f"/proc/{pid}/status").read():
            time.sleep(30)
        print(f"##### {pidfile} (pid {pid}) finished, starting batch", flush=True)
    except FileNotFoundError:
        print(f"##### {pidfile} not found, starting batch now", flush=True)
for i, cmd in enumerate(cmds):
    argv = [sys.executable, "-u", *shlex.split(cmd)]
    print(f"\n##### batch job {i + 1}/{len(cmds)}: {cmd}", flush=True)
    t0 = time.time()
    rc = subprocess.call(argv)
    print(f"##### job {i + 1} exit {rc} after {(time.time() - t0) / 60:.1f} min", flush=True)
    if rc:
        sys.exit(rc)
