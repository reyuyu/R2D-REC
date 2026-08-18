"""Synthetic coverage for compact records and the preregistered gate."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from gr_rec_dsr_v1.dsr_objectives import build_nothink_rescue_plan

from .simple_forensic import evaluate_gate_data, nothink_forensic_rows, think_forensic_rows
from .simple_probe import SimpleGateFixedProbeCallback


def _think(raw=3, invalid=0, zero=False, step=180):
    rewards = [0.0] * 4 if zero else [8.0, 2.0, 0.5, 0.0]
    return [{
        "type": "think_candidate", "step": step, "Raw_N": raw,
        "Coverage": 1.0, "group_primary_zero_std": zero,
        "group_primary_all_zero": zero,
        "Beam": {"invalid_count_for_this_Beam32_task": invalid},
    } for _ in rewards]


def _no(rewards=None, zero=False, wrong=False, step=180):
    values = rewards or ([0.0] * 8 if zero else [8.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    if wrong:
        values = [-0.25] * 8
    return [{
        "type": "nothink_group", "step": step, "candidate_rewards": values,
        "zero_std": zero or wrong, "all_zero": zero, "rescue_active": zero,
        "A_concentration": 0.25,
    }]


def _policy(kl=0.0, clip=0.0):
    return [{"step": step, "approx_kl": kl, "clip_fraction": clip} for step in range(180, 200)]


def _healthy():
    return _think(), _no(), _policy(), []


def test_compact_forensic_shapes_and_fields():
    records = []
    for candidate_id, reward in enumerate((8.0, 2.0, 0.5, 0.0)):
        records.append({
            "group_id": "g", "target_domain": "prod", "gold_count": 5,
            "unique_gold_a": 3, "primary_reward": reward, "raw_interest_n": 4,
            "s_n": 1.0, "unique_valid_target_a": 2, "d_a": 0.25,
            "simple_s_aux": 0.0, "simple_a_aux": 0.0, "simple_branch": "primary_only",
            "completion_length": 10 + candidate_id, "closed": True, "beam_invalid": 0,
            "parsed": {"parser_success": True},
            "diagnostic_only": {"grounded_n": 3, "exact": 1, "ab": 2, "a": 3},
        })
    rows = think_forensic_rows(records, 10, 5)
    assert len(rows) == 4 and rows[0]["Coverage"] == 0.75
    assert rows[0]["group_primary_std"] > 0 and rows[0]["candidate_id"] == 0

    no_records = [{
        "group_id": "n", "target_domain": "prod", "gold_count": 2,
        "unique_gold_a": 2, "primary_reward": 0.0, "predicted_a": 7,
        "sa_position": 1, "gold_as": [1, 2],
    } for _ in range(8)]
    plan = build_nothink_rescue_plan([0.0] * 8, [7] * 8, [1] * 8, [1, 2])
    no_rows = nothink_forensic_rows(no_records, [plan], 10, 5)
    assert len(no_rows) == 1 and no_rows[0]["rescue_active"]
    assert no_rows[0]["frequency_weights"] == [1.0] * 8


def test_gate_pass_and_each_catastrophic_stop():
    assert evaluate_gate_data(*_healthy())["decision"] == "PASS"
    cases = []
    think, no, policy, probes = _healthy(); cases.append(("beam", _think(invalid=1), no, policy, probes))
    think, no, policy, probes = _healthy(); cases.append(("raw", _think(raw=1), no, policy, probes))
    think, no, policy, probes = _healthy(); cases.append(("valid", think, _no(rewards=[-1.0] * 8), policy, probes))
    think, no, policy, probes = _healthy(); cases.append(("wrong", think, _no(wrong=True), policy, probes))
    think, no, policy, probes = _healthy(); cases.append(("kl", think, no, _policy(kl=0.031), probes))
    think, no, policy, probes = _healthy(); cases.append(("clip", think, no, _policy(clip=0.201), probes))
    for name, think, no, policy, probes in cases:
        report = evaluate_gate_data(think, no, policy, probes)
        assert report["decision"] == "STOP", (name, report)


def test_gate_combination_stop_and_single_warning_continues():
    think = _think(zero=True)
    no = _no(zero=False)
    report = evaluate_gate_data(think, no, _policy(), [])
    assert report["decision"] == "WARN"
    no = _no(zero=True)
    report = evaluate_gate_data(think, no, _policy(), [])
    assert report["decision"] == "STOP"
    assert "combined_serious_outcomes_at_least_2" in report["reasons"]


def test_gate_callback_synthetic_pass_and_stop(tmp_path=None):
    import tempfile
    root = Path(tempfile.mkdtemp()) if tmp_path is None else Path(tmp_path)
    evaluator = SimpleNamespace(
        every_steps=200,
        evaluate=lambda step, reason: None,
        monitor=SimpleNamespace(run_dir=root),
    )
    callback = SimpleGateFixedProbeCallback(evaluator)
    state = SimpleNamespace(global_step=200)
    for decision, expected_stop in (("PASS", False), ("WARN", False), ("STOP", True)):
        (root / "gate200_report.json").write_text(json.dumps({"decision": decision}), encoding="utf-8")
        control = SimpleNamespace(should_save=False, should_training_stop=False)
        callback.on_step_end(SimpleNamespace(), state, control)
        assert control.should_save and control.should_training_stop is expected_stop


def test_gate_checkpoint_persistence_is_checked_only_on_world_process_zero(tmp_path=None):
    import tempfile
    root = Path(tempfile.mkdtemp()) if tmp_path is None else Path(tmp_path)
    evaluator = SimpleNamespace(
        every_steps=200,
        evaluate=lambda step, reason: None,
        monitor=SimpleNamespace(run_dir=root),
    )
    callback = SimpleGateFixedProbeCallback(evaluator)
    args = SimpleNamespace(output_dir=str(root))
    control = SimpleNamespace()

    worker_state = SimpleNamespace(global_step=200, is_world_process_zero=False)
    callback.on_save(args, worker_state, control)

    main_state = SimpleNamespace(global_step=200, is_world_process_zero=True)
    try:
        callback.on_save(args, main_state, control)
    except RuntimeError as exc:
        assert str(exc) == "Gate200 checkpoint persistence failed"
    else:
        raise AssertionError("rank 0 must reject a missing Gate200 adapter")

    checkpoint = root / "checkpoint-200"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    callback.on_save(args, main_state, control)
