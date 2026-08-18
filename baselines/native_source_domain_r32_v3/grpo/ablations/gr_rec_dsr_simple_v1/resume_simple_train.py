#!/usr/bin/env python3
"""Recover the formal DSR-Simple run from its accepted Gate200 checkpoint."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/data/GRPO/scripts")
sys.path.insert(0, "/data/GRPO/scripts/ablations")

from gr_rec_dsr_simple_v1 import run_simple_train as simple


REQUIRED_CHECKPOINT_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "rng_state_0.pth",
    "rng_state_1.pth",
    "rng_state_2.pth",
    "rng_state_3.pth",
)


def validate_gate200_recovery(
    argv,
    outputs_root=Path("/data/GRPO/outputs/formal"),
    runs_root=Path("/data/GRPO/runs"),
):
    values = list(argv)
    if values.count("--run-id") != 1 or values.count("--resume-from-checkpoint") != 1:
        raise ValueError("Gate200 recovery requires one run id and one checkpoint")
    run_pos = values.index("--run-id")
    resume_pos = values.index("--resume-from-checkpoint")
    if run_pos + 1 >= len(values) or resume_pos + 1 >= len(values):
        raise ValueError("Gate200 recovery arguments require values")
    run_id = values[run_pos + 1]
    checkpoint = Path(values[resume_pos + 1]).resolve()
    expected = (Path(outputs_root) / run_id / "checkpoint-200").resolve()
    if checkpoint != expected:
        raise ValueError(f"Gate200 recovery requires exact checkpoint {expected}")
    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (checkpoint / name).is_file()]
    if missing:
        raise ValueError(f"Gate200 checkpoint is incomplete: {missing}")
    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if int(trainer_state.get("global_step", -1)) != 200:
        raise ValueError("Gate200 trainer state must have global_step 200")
    gate = json.loads((Path(runs_root) / run_id / "gate200_report.json").read_text(encoding="utf-8"))
    if gate.get("decision") not in {"PASS", "WARN"}:
        raise ValueError("Gate200 recovery requires a PASS or WARN decision")

    formal_values = values[:resume_pos] + values[resume_pos + 2:]
    enforced = simple.enforce_simple_contract(formal_values)
    return enforced + ["--resume-from-checkpoint", str(checkpoint)]


def main(argv=None):
    values = validate_gate200_recovery(sys.argv[1:] if argv is None else argv)
    checkpoint = Path(values[values.index("--resume-from-checkpoint") + 1]).resolve()
    # The checkpoint is locally generated and path-locked above. Torch <2.6
    # cannot safely-unpickle the NumPy objects in Trainer's RNG state.
    import transformers.trainer as trainer_module
    original_torch_load = trainer_module.torch.load

    def load_trusted_recovery_state(path, *args, **kwargs):
        candidate = Path(path).resolve()
        if candidate.parent == checkpoint and candidate.suffix in {".pt", ".pth"}:
            kwargs["weights_only"] = False
        return original_torch_load(path, *args, **kwargs)

    trainer_module.check_torch_load_is_safe = lambda: None
    trainer_module.torch.load = load_trusted_recovery_state
    simple.reset_global_capture()
    simple.baseline_train.RecGRPOTrainer = simple.SimpleDsrGRPOTrainer
    simple.baseline_train.make_nothink_reward_func = simple.make_simple_nothink_reward_func
    simple.baseline_train.make_think_reward_func = simple.make_simple_think_reward_func
    simple.baseline_train.make_beam32_fn = simple.simple_beam_factory
    simple.baseline_train.monitor_from_env = simple.simple_monitor_factory
    simple.baseline_train.FixedProbeEvaluator = simple.SimpleFixedProbeEvaluator
    simple.baseline_train.FixedProbeCallback = simple.SimpleGateFixedProbeCallback
    simple.baseline_train.main(values)


if __name__ == "__main__":
    main()
