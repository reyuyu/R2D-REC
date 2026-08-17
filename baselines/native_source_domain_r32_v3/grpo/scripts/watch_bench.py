# -*- coding: utf-8 -*-
"""Wait for bench_fixed process to finish; then re-run it redirecting to a log."""
import time
import paramiko

def ssh():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("103.102.203.0", port=1058, username="root", password="Yhjyhj@+0519",
              timeout=30, look_for_keys=False, allow_agent=False, banner_timeout=30, auth_timeout=30)
    return c

def run(cmd, t=90):
    c = ssh()
    try:
        stdin, stdout, stderr = c.exec_command(cmd, timeout=t)
        return stdout.read().decode("utf-8", "replace")
    finally:
        c.close()

def log(msg):
    print(time.strftime("%H:%M:%S") + " " + msg, flush=True)

log("waiting for bench_fixed to finish...")
deadline = time.time() + 2400
while time.time() < deadline:
    alive = run("pgrep -f 'bench_fixed.p[y]' | head -1").strip()
    if not alive:
        log("process finished")
        break
    time.sleep(60)
# re-run with output redirect to capture results (idempotent; re-run is fine)
log("re-running bench_fixed with log capture...")
print(run("cd /data/GRPO/scripts && CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 nohup /data/venvs/llamafactory-01398eb-liger081/bin/python bench_fixed.py > /data/GRPO/logs/bench_fixed.log 2>&1 < /dev/null & echo STARTED", t=30))
time.sleep(20)
log("started")
