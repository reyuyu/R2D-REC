"""Runner isolation and frozen schedule tests."""
import json
from pathlib import Path
import tempfile

from .resume_simple_train import REQUIRED_CHECKPOINT_FILES, validate_gate200_recovery
from .simple_contract import (
    EXPECTED_SCHEDULE,
    OPTIMIZER_SCHEDULE_SHA256,
    PROBE_GROUP_IDS,
    enforce_simple_contract,
)


def test_contract_injects_frozen_fairness_values():
    values = enforce_simple_contract(["--run-id", "GR-REC-DSR-SIMPLE-V1-TEST"])
    assert values[values.index("--seed") + 1] == "20260816"
    assert values[values.index("--probe-seed") + 1] == "20260818"
    assert [values[index + 1] for index, value in enumerate(values) if value == "--probe-group-id"] == list(PROBE_GROUP_IDS)
    assert EXPECTED_SCHEDULE["optimizer_steps"] == 2316
    assert OPTIMIZER_SCHEDULE_SHA256 == "ac86490e5660e365fe4adedb6ac9321cffb95db1636b2f4bafad99841ba508b7"


def test_contract_rejects_resume_and_wrong_prefix():
    for args in (
        ["--run-id", "GR-REC-DSR-SIMPLE-V1-X", "--resume-from-checkpoint", "/tmp/x"],
        ["--run-id", "GR-REC-DSR-V1-WRONG"],
    ):
        try:
            enforce_simple_contract(args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"contract accepted {args}")


def test_gate200_recovery_accepts_only_complete_accepted_exact_checkpoint():
    root = Path(tempfile.mkdtemp())
    outputs = root / "outputs"
    runs = root / "runs"
    run_id = "GR-REC-DSR-SIMPLE-V1-RECOVERY-TEST"
    checkpoint = outputs / run_id / "checkpoint-200"
    checkpoint.mkdir(parents=True)
    for name in REQUIRED_CHECKPOINT_FILES:
        (checkpoint / name).write_bytes(b"x")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 200}), encoding="utf-8"
    )
    run_dir = runs / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "gate200_report.json").write_text(
        json.dumps({"decision": "WARN"}), encoding="utf-8"
    )
    values = validate_gate200_recovery(
        ["--run-id", run_id, "--resume-from-checkpoint", str(checkpoint)],
        outputs_root=outputs,
        runs_root=runs,
    )
    assert values[-2:] == ["--resume-from-checkpoint", str(checkpoint.resolve())]

    (run_dir / "gate200_report.json").write_text(
        json.dumps({"decision": "STOP"}), encoding="utf-8"
    )
    try:
        validate_gate200_recovery(
            ["--run-id", run_id, "--resume-from-checkpoint", str(checkpoint)],
            outputs_root=outputs,
            runs_root=runs,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("STOP gate must reject recovery")
