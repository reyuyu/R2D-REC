"""Runner isolation and frozen schedule tests."""
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
