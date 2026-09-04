"""Run several of our scripts sequentially on the VM (use with run_remote.sh start).

    python colab/run_batch.py "mumaxplus/run_field_sweep.py --seed 0 ..." "mumaxplus/run_field_sweep.py --seed 1 ..."
"""
import shlex, subprocess, sys, time
for i, cmd in enumerate(sys.argv[1:]):
    argv = [sys.executable, "-u", *shlex.split(cmd)]
    print(f"\n##### batch job {i + 1}/{len(sys.argv) - 1}: {cmd}", flush=True)
    t0 = time.time()
    rc = subprocess.call(argv)
    print(f"##### job {i + 1} exit {rc} after {(time.time() - t0) / 60:.1f} min", flush=True)
    if rc:
        sys.exit(rc)
