from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

try:
    from .dual_beam8_adapter import adapt_events
    from .export_nonzero_samples import export_available
    from .server import create_app, monitor_advantage_formula
except ImportError:
    from dual_beam8_adapter import adapt_events
    from export_nonzero_samples import export_available
    from server import create_app, monitor_advantage_formula


def captured_event(step: int = 2) -> dict:
    cots = []
    for cot_index in range(4):
        rewards = [0.0] * 8
        advantages = [0.0] * 8
        levels = ["domain"] * 8
        if cot_index == 1:
            rewards[0], rewards[1] = 2.0, -0.25
            advantages[0], advantages[1] = 2.1, -0.7
            levels[0], levels[1] = "ab", "invalid"
        cots.append({
            "cot_text": f"<think>cot-{cot_index}</think>",
            "cot_length": 10 + cot_index,
            "closed": True,
            "cot_reward": 2.0 if cot_index == 1 else 0.0,
            "beam_candidate_ids": [[100 + i, 200 + i, 300 + i] for i in range(8)],
            "beam_sids": [["video", i, i + 1, i + 2] for i in range(8)],
            "sid_rewards": rewards,
            "sid_advantages": advantages,
            "sid_reward_levels": levels,
            "sid_population_std": 0.75 if cot_index == 1 else 0.0,
            "sid_zero_std": cot_index != 1,
            "domain_prefix": "<|video_begin|>",
            "exact": 0,
            "ab": 1 if cot_index == 1 else 0,
            "a": 0,
            "invalid": 1 if cot_index == 1 else 0,
        })
    return {
        "type": "dual_beam8",
        "step": step,
        "rollout_id": 1,
        "rollout_fingerprint": f"fingerprint-{step}",
        "recommendation_group_id": "group-1",
        "target_domain": "video",
        "cot_rewards": [0.0, 2.0, 0.0, 0.0],
        "cot_advantages": [-0.5, 1.5, -0.5, -0.5],
        "cot_population_std": 0.866,
        "cot_zero_std": False,
        "normalization_topology": "G4 + 4 independent G8; never G32",
        "cots": cots,
    }


def captured_sample8_event() -> dict:
    event = captured_event()
    event["type"] = "sample8_fullsid"
    for cot in event["cots"]:
        cot["sample_candidate_ids"] = [
            [10, 11, 12, 13, 14, 15, 16, 17] for _ in range(8)
        ]
        cot["sample_candidate_texts"] = ["prefix <SID> suffix"] * 8
        cot["sample_sids"] = cot.pop("beam_sids")
        cot.pop("beam_candidate_ids")
        cot["all_parsed_sids"] = [[sid] for sid in cot["sample_sids"]]
        cot["sid_action_spans"] = [[2, 6]] * 8
        cot["sid_counts"] = [1] * 7 + [2]
        cot["multi_sid_outputs"] = [False] * 7 + [True]
        cot["parser_statuses"] = ["ok"] * 7 + ["multiple_sids"]
        cot["sample8_wall_sec"] = 3.2
    return event


def test_adapter_preserves_two_level_captured_credit_and_input():
    groups = adapt_events(
        [captured_event()],
        source_index={"group-1": {
            "prompt": "model input",
            "all_gold_sids": ["<|video_begin|><s_a_0><s_b_1><s_c_2>"],
        }},
    )
    assert len(groups) == 1
    group = groups[0]
    assert group["valid"] is True
    assert group["prompt"] == "model input"
    assert len(group["candidates"]) == 4
    assert group["candidates"][1]["reward"] == 2.0
    assert group["candidates"][1]["final_advantage"] == 1.5
    assert len(group["candidates"][1]["sid_candidates"]) == 8
    sid = group["candidates"][1]["sid_candidates"][0]
    assert sid["reward"] == 2.0
    assert sid["advantage"] == 2.1
    assert sid["parsed_sid_text"] == "<|video_begin|><s_a_0><s_b_1><s_c_2>"


def test_api_capability_and_advantages_are_captured():
    with tempfile.TemporaryDirectory() as temporary:
        run = Path(temporary)
        (run / "manifest.json").write_text(json.dumps({
            "run_id": run.name,
            "experiment": "GR_REC_ThinkDualBeam8_v2",
            "dataset_path": "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl",
        }), encoding="utf-8")
        (run / "dual_beam8.jsonl").write_text(
            json.dumps(captured_event()) + "\n", encoding="utf-8"
        )
        client = TestClient(create_app(run_dir=run))
        capability = client.get("/api/capabilities").json()
        assert capability["dual_beam8"] is True
        assert capability["advantage_formula"] == "dual_beam8_v2"
        assert capability["advantage_source"] == "captured"
        payload = client.get("/api/advantages").json()
        assert payload["supported"] is True
        assert payload["formula"] == "dual_beam8_v2"
        assert payload["provenance"]["mode"] == "captured"
        assert len(payload["groups"][0]["candidates"][0]["sid_candidates"]) == 8
        direct = client.get("/api/dual-beam8").json()
        assert direct["groups"][0]["normalization_topology"].endswith("never G32")
        assert monitor_advantage_formula({
            "experiment": "GR_REC_ThinkDualBeam8_v2"
        }) == "dual_beam8_v2"


def test_sample8_api_uses_its_stream_and_preserves_first_sid_metadata():
    with tempfile.TemporaryDirectory() as temporary:
        run = Path(temporary)
        (run / "manifest.json").write_text(json.dumps({
            "run_id": run.name,
            "experiment": "GR_REC_ThinkSample8_FullSID_v3",
            "dataset_path": "/data/GRPO/data/rec_mp_grpo_v2/train.jsonl",
        }), encoding="utf-8")
        (run / "sample8_fullsid.jsonl").write_text(
            json.dumps(captured_sample8_event()) + "\n", encoding="utf-8"
        )
        client = TestClient(create_app(run_dir=run))
        capability = client.get("/api/capabilities").json()
        assert capability["dual_beam8"] is True
        assert capability["advantage_formula"] == "sample8_fullsid_v3"
        payload = client.get("/api/advantages").json()
        assert payload["formula"] == "sample8_fullsid_v3"
        group = payload["groups"][0]
        assert group["sample8_fullsid"] is True
        sid = group["candidates"][0]["sid_candidates"][7]
        assert sid["sid_action_span"] == [2, 6]
        assert sid["sid_count"] == 2
        assert sid["multi_sid_output"] is True
        assert sid["parser_status"] == "multiple_sids"


def test_nonzero_export_is_incremental_deduplicated_and_keeps_prompt():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run = root / "run"
        run.mkdir()
        dataset = root / "train.jsonl"
        dataset.write_text(json.dumps({
            "recommendation_group_id": "group-1",
            "prompt": "full prompt",
            "all_gold_sids": ["gold"],
        }) + "\n", encoding="utf-8")
        (run / "manifest.json").write_text(json.dumps({
            "dataset_path": str(dataset),
        }), encoding="utf-8")
        (run / "dual_beam8.jsonl").write_text(
            json.dumps(captured_event()) + "\n", encoding="utf-8"
        )
        output = run / "training_sample_exports"
        first = export_available(run, output)
        second = export_available(run, output)
        assert first["added_nonzero"] == 1
        assert first["added_positive"] == 1
        assert second["added_nonzero"] == 0
        row = json.loads((output / "positive_samples.jsonl").read_text(encoding="utf-8"))
        assert row["prompt"] == "full prompt"
        assert row["cot_reward"] == 2.0
        assert len(row["sid_rewards"]) == 8
        assert row["provenance"] == "training_capture_no_recalculation"


def test_sample8_export_keeps_continuation_and_multiple_sid_flag():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run = root / "run"
        run.mkdir()
        dataset = root / "train.jsonl"
        dataset.write_text(json.dumps({
            "recommendation_group_id": "group-1", "prompt": "prompt",
            "all_gold_sids": ["gold"],
        }) + "\n", encoding="utf-8")
        (run / "manifest.json").write_text(json.dumps({
            "experiment": "GR_REC_ThinkSample8_FullSID_v3",
            "dataset_path": str(dataset),
        }), encoding="utf-8")
        (run / "sample8_fullsid.jsonl").write_text(
            json.dumps(captured_sample8_event()) + "\n", encoding="utf-8"
        )
        result = export_available(run, run / "training_sample_exports")
        assert result["added_positive"] == 1
        row = json.loads((run / "training_sample_exports" / "positive_samples.jsonl").read_text())
        assert row["experiment"] == "GR_REC_ThinkSample8_FullSID_v3"
        assert row["sample_candidate_texts"][0] == "prefix <SID> suffix"
        assert row["multi_sid_outputs"][-1] is True


def test_frontend_contract_shows_cot_and_sid_credit_without_recalculation():
    static = Path(__file__).with_name("static")
    source = (static / "dual_beam8_dashboard.js").read_text(encoding="utf-8")
    index = (static / "index.html").read_text(encoding="utf-8")
    for text in ("CoT 分数", "CoT 优势", "答案分数", "答案优势", "模型输入",
                 "Reward-only Reference · NOT MODEL INPUT", "实采", "多 SID · 训练取第一个",
                 "查看完整 sampled continuation"):
        assert text in source
    assert "8 个 SID 独立做 G8 归一化" in source
    assert "前端未重算" in source
    assert "state.capabilities.dual_beam8||heavyDue" in index
    assert "dual_beam8_dashboard.js" in index
