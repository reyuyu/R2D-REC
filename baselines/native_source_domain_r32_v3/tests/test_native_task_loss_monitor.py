"""Verify native task loss accumulation is a sample mean across a GA window."""
import importlib.util
import math
from pathlib import Path

import torch

PATH = Path(__file__).parents[1] / 'scripts' / 'train_native_source_domain_r32_v3.py'
spec = importlib.util.spec_from_file_location('native_source', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeTrainer:
    pass


trainer = FakeTrainer()
all_loss = [[] for _ in module.TASK_NAMES]
# Simulate 16 compute_loss calls with uneven segment counts exactly as packed
# batches create in practice.  Domain factors are already applied at the same
# point as production code.
for micro in range(16):
    task_ids = torch.tensor([(micro + k) % 4 for k in range((micro % 3) + 1)])
    raw = torch.tensor([0.25 + micro * 0.1 + k * 0.01 for k in range(len(task_ids))])
    domain = torch.tensor([1.0 + 0.05 * k for k in range(len(task_ids))])
    observed = raw * domain
    module._accumulate_task_loss_metrics(trainer, observed, task_ids)
    for value, task in zip(observed.tolist(), task_ids.tolist()):
        all_loss[task].append(value)

stats = trainer._native_task_loss_stats.cpu()
logged = [float(stats[i, 0] / stats[i, 1]) for i in range(len(module.TASK_NAMES))]
manual = [sum(values) / len(values) for values in all_loss]
report = {name: {'logged': logged[i], 'manual': manual[i], 'difference': logged[i] - manual[i], 'count': len(all_loss[i])}
          for i, name in enumerate(module.TASK_NAMES)}
print('TASK_LOSS_MONITOR_AUDIT=' + repr(report))
assert all(abs(logged[i] - manual[i]) < 1e-12 for i in range(4))
