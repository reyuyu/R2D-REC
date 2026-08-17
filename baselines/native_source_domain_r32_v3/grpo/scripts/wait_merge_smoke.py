# -*- coding: utf-8 -*-
"""Wait for 4 GPU smoke summaries, then merge them."""
import time
import paramiko

def ssh():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("103.102.203.0", port=1058, username="root", password="Yhjyhj@+0519",
              timeout=30, look_for_keys=False, allow_agent=False, banner_timeout=30, auth_timeout=30)
    return c

def run(cmd, t=120):
    c = ssh()
    try:
        stdin, stdout, stderr = c.exec_command(cmd, timeout=t)
        return stdout.read().decode("utf-8", "replace")
    finally:
        c.close()

def log(msg):
    print(time.strftime("%H:%M:%S") + " " + msg, flush=True)

FILES = [f"/data/GRPO/logs/smoke_v2_gpu{i}.json" for i in range(4)]
log("waiting for 4 smoke summaries...")
while True:
    present = [f for f in FILES if run(f"test -f {f} && echo Y || echo N").strip() == "Y"]
    log(f"ready {len(present)}/4")
    if len(present) == 4:
        break
    # also check for process death (crash)
    alive = run("pgrep -f run_smoke.py | wc -l").strip()
    if alive == "0" and len(present) < 4:
        log("WARNING: all run_smoke processes died but not all summaries ready")
        for f in FILES:
            log(run(f"tail -c 600 {f.replace('.json','.log')} 2>/dev/null"))
        break
    time.sleep(90)

log("=== merging ===")
print(run("cd /data/GRPO/scripts && PYTHONIOENCODING=utf-8 /data/venvs/llamafactory-01398eb-liger081/bin/python merge_smoke.py 2>&1", t=120))
log("done")
