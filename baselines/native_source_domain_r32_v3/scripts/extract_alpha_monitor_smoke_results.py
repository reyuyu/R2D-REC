import json
import statistics

for name in ("off", "on"):
    path = f"/data/logs/alpha_monitor_{name}_smoke.log"
    durations, peaks = [], []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "PACK_RATIO_SMOKE_TIMING=" in line:
                payload, _ = json.JSONDecoder().raw_decode(line.split("PACK_RATIO_SMOKE_TIMING=", 1)[1])
                durations.extend(payload["step_durations_sec"])
                peaks.append(payload["peak_cuda_gib"])
    print(json.dumps({
        "run": name,
        "steps": len(durations),
        "mean_sec": statistics.mean(durations),
        "median_sec": statistics.median(durations),
        "min_sec": min(durations),
        "max_sec": max(durations),
        "peak_gib_per_rank": peaks,
    }, sort_keys=True))
