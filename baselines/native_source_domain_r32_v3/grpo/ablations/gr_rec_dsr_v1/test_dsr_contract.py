from __future__ import annotations

from .dsr_contract import (
    PROBE_EVERY_STEPS,
    PROBE_GROUP_IDS,
    PROBE_SEED,
    TRAIN_SEED,
    enforce_formal_contract,
)


def _values(argv, option):
    return [argv[index + 1] for index, value in enumerate(argv) if value == option]


def _assert_rejected(argv):
    try:
        enforce_formal_contract(argv)
    except ValueError:
        return
    raise AssertionError(f"contract accepted invalid arguments: {argv}")


def test_contract_injects_frozen_probe_and_data_order():
    result = enforce_formal_contract(["--run-id", "GR-REC-DSR-V1-PILOT", "--max-steps", "400"])
    assert _values(result, "--seed") == [TRAIN_SEED]
    assert _values(result, "--n-groups") == ["all"]
    assert _values(result, "--probe-groups") == ["4"]
    assert _values(result, "--probe-seed") == [PROBE_SEED]
    assert _values(result, "--probe-every-steps") == [PROBE_EVERY_STEPS]
    assert _values(result, "--probe-group-id") == list(PROBE_GROUP_IDS)


def test_contract_accepts_exact_explicit_values():
    argv = [
        "--seed", TRAIN_SEED,
        "--n-groups", "all",
        "--probe-groups", "4",
        "--probe-seed", PROBE_SEED,
        "--probe-every-steps", PROBE_EVERY_STEPS,
    ]
    for group_id in PROBE_GROUP_IDS:
        argv.extend(["--probe-group-id", group_id])
    assert enforce_formal_contract(argv) == argv


def test_contract_rejects_resume_or_fairness_drift():
    for argv in (
        ["--resume-from-checkpoint", "/tmp/checkpoint-2"],
        ["--seed", "7"],
        ["--n-groups", "100"],
        ["--probe-groups", "0"],
        ["--probe-seed", "7"],
        ["--probe-every-steps", "100"],
        ["--probe-group-id", PROBE_GROUP_IDS[1]],
    ):
        _assert_rejected(argv)
