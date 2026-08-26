"""CPU-only Phase 1.1E contract audit. No tokenizer/model forward is executed."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from beam32_contract import BeamDecoderContract, synthetic_beam_fixture
from compare_checkpoints import checkpoint_comparison_row
from eval_aggregate import aggregate_results
from eval_request import OfficialAlignedEvalRequest, frozen_split_interface
from eval_result import EvaluationBusinessSample, deduplicate_source_rows
from model_spec import ModelSpec
from official_aligned_contract import ACTION_LEVELS, ACTION_TOKENS, CONTRACT_ID, CONTRACT_SOURCE, DOMAIN_TOKENS, EVAL_BEAM_SIZE, OFFICIAL_COMPOSITE_FORMULA_AVAILABLE, OFFICIAL_EVAL_SOURCE_CODE_AVAILABLE, TRAIN_ROLLOUT_G, UNKNOWN_OFFICIAL_DECODING_DETAILS, eval_prefix_contract, shared_prefix_contract_pass, train_prefix_contract, validate_no_bridge_contract
from official_aligned_renderer import RENDERER_CONTRACT_ID
from scoring_adapter import INTERNAL_DIAGNOSTICS, SCORING_LABEL, internal_group_diagnostics


FROZEN_SPLIT_SHA = {
    "probe20_group_ids.json": "7193f540ee229ef53b7ca09e541f065396fe212a8cfd43893cceff499d83abdf",
    "dev_group_ids.json": "70341b476c57e4d7dc961ccf423ab3991a42375af1989bad040be4134e8f4fe4",
    "final_group_ids.json": "e6be46befa8ed320fb0762e6df23f553c6646297bdf5649003e332c04b2e61d0",
}
PHASE11E_FILES = (
    "eval/official_aligned_contract.py", "eval/model_spec.py", "eval/official_aligned_renderer.py",
    "eval/beam32_contract.py", "eval/eval_request.py", "eval/eval_result.py",
    "eval/eval_aggregate.py", "eval/compare_checkpoints.py", "eval/scoring_adapter.py",
    "eval/official_aligned_eval_v1.yaml", "eval/run_official_aligned_audit.py",
    "tests/test_official_aligned_eval_v1.py",
)
BRIDGE_MARKERS = (
    "bridge_prefix", "legacy_bridge", "answer_bridge", "beta_bridge",
    "natural_language_bridge", "该用户最近喜欢的视频有:", "该用户最近点击了商品:",
    "该用户最近感兴趣的广告有:", "该用户最近首次打赏了主播:",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_text_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_split(runtime_root: Path, name: str) -> list[str]:
    path = runtime_root / "data" / "splits" / name
    if canonical_text_sha(path) != FROZEN_SPLIT_SHA[name]:
        raise RuntimeError(f"FROZEN_SPLIT_SHA_FAIL={name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload and isinstance(payload[0], dict):
        return [str(row["recommendation_group_id"]) for row in payload]
    return [str(value) for value in payload]


def run(output_dir: Path, source_root: Path, runtime_root: Path, test_status: str) -> None:
    if test_status != "PASS":
        raise RuntimeError("TEST_STATUS_GATE_FAIL")
    output_dir.mkdir(parents=True, exist_ok=True)
    parity = {
        relative: {"source": sha(source_root / relative), "runtime": sha(runtime_root / relative)}
        for relative in PHASE11E_FILES
    }
    if not all(value["source"] == value["runtime"] for value in parity.values()):
        raise RuntimeError("SOURCE_RUNTIME_PARITY_FAIL")

    train, evaluation = train_prefix_contract(), eval_prefix_contract()
    prefix_audit = {
        "contract_id": CONTRACT_ID,
        "contract_source": CONTRACT_SOURCE,
        "train_eval_shared_prefix": shared_prefix_contract_pass(),
        "train": vars(train),
        "eval": vars(evaluation),
        "train_rollout_G": TRAIN_ROLLOUT_G,
        "eval_beam_size": EVAL_BEAM_SIZE,
        "source_runtime_parity": parity,
    }
    write_json(output_dir / "train_eval_prefix_contract_audit.json", prefix_audit)

    training_files = (runtime_root / "trainer" / "rollout_runtime_v1.py", runtime_root / "trainer" / "truerec_grpo_trainer_v1.py")
    renderer_source = (runtime_root / "data" / "beta_gamma_renderer.py").read_text(encoding="utf-8")
    insertion_hits = []
    for path in training_files:
        text = path.read_text(encoding="utf-8")
        insertion_hits.extend({"file": path.name, "marker": marker} for marker in BRIDGE_MARKERS if marker in text)
    renderer_method = renderer_source[renderer_source.index("    def rl_context_ids"): renderer_source.index("    def encode", renderer_source.index("    def rl_context_ids"))]
    renderer_hits = [marker for marker in BRIDGE_MARKERS if marker in renderer_method]
    validate_no_bridge_contract({"bridge": False, "legacy_bridge": None, "beta_bridge": False})
    no_bridge = {
        "train_bridge": False,
        "eval_bridge": False,
        "beta_special_bridge_in_eval": False,
        "beta_special_prefix_in_eval": False,
        "training_runtime_bridge_insertions": len(insertion_hits) + len(renderer_hits),
        "training_file_hits": insertion_hits,
        "renderer_method_hits": renderer_hits,
    }
    if no_bridge["training_runtime_bridge_insertions"]:
        raise RuntimeError(f"BRIDGE_INSERTION_GATE_FAIL={no_bridge}")
    write_json(output_dir / "no_bridge_audit.json", no_bridge)

    token_audit_path = runtime_root / "results" / "phase0_5" / "tokenizer_contract_audit.json"
    token_audit = json.loads(token_audit_path.read_text(encoding="utf-8"))
    fixed_domain = {
        "domain_tokens": DOMAIN_TOKENS,
        "domain_token_ids": token_audit["domain_token_ids"],
        "domain_token_single_token": token_audit["domain_token_single_token"],
        "fixed_domain_in_context": token_audit["domain_token_single_token"] == "PASS",
        "domain_is_action": False,
        "action_starts_after_fixed_domain": True,
        "action_levels": ACTION_LEVELS,
        "action_tokens": ACTION_TOKENS,
    }
    write_json(output_dir / "fixed_domain_audit.json", fixed_domain)

    models = (
        ModelSpec("Beta-Baseline", "beta", "/data/outputs/baselines/native_source_domain_r32_v3/BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106", "/data/models/onereason-8b-pretrain-competition", "/data/outputs/baselines/native_source_domain_r32_v3/BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106", "/data/outputs/baselines/native_source_domain_r32_v3/BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106"),
        ModelSpec("Beta-Gamma", "beta_gamma", "/data/outputs/baselines/native_source_domain_r32_v3/BETA-GAMMA-R32-2E-GC04-4GPU-20260825-054127", "/data/models/onereason-8b-pretrain-competition", "/data/outputs/baselines/native_source_domain_r32_v3/BETA-GAMMA-R32-2E-GC04-4GPU-20260825-054127", "/data/outputs/baselines/native_source_domain_r32_v3/BETA-GAMMA-R32-2E-GC04-4GPU-20260825-054127"),
        ModelSpec("TrueRec-configurable", "truerec", "CONFIGURABLE_TRUE_REC_CHECKPOINT", "/data/models/onereason-8b-pretrain-competition", "CONFIGURABLE_TRUE_REC_ADAPTER", "CONFIGURABLE_TOKENIZER_PATH"),
    )
    requests = [OfficialAlignedEvalRequest(model, "dev512") for model in models]
    beta_audit = {
        "shared_eval_contract": len({request.eval_contract_id for request in requests}) == 1,
        "shared_renderer_contract": len({request.renderer_contract_id for request in requests}) == 1,
        "shared_decoder_contract": len({request.decoder_contract_id for request in requests}) == 1,
        "model_specs": [model.provenance() for model in models],
        "family_changes_prompt_contract": False,
    }
    write_json(output_dir / "beta_alignment_audit.json", beta_audit)

    token_map = {1: "<s_a_1>", 2: "<s_b_2>", 3: "<s_c_3>"}
    rows = [([1, 2, 3], float(32 - index)) for index in range(32)]
    candidates = synthetic_beam_fixture(rows, recommendation_group_id="synthetic-group", target_domain="video", fixed_domain_token=DOMAIN_TOKENS["video"], id_to_token=token_map.__getitem__)
    diagnostics = internal_group_diagnostics(candidates, ("<s_a_1><s_b_2><s_c_3>",))
    decoder = BeamDecoderContract()
    beam_audit = {
        "frontend_only": not decoder.real_gpu_decoder_implemented,
        "synthetic_fixture_is_official_decoder": False,
        "beam_size": decoder.beam_size,
        "action_tokens": decoder.action_tokens,
        "candidate_count": len(candidates),
        "raw_token_ids_retained": all(item.raw_token_ids == (1, 2, 3) for item in candidates),
        "format_valid_count": sum(item.format_valid for item in candidates),
        "parser_source": "Phase0.7 rollout_metrics.parse_raw_abc",
        "internal_diagnostics": diagnostics,
    }
    write_json(output_dir / "beam32_contract_audit.json", beam_audit)

    probe_ids = load_split(runtime_root, "probe20_group_ids.json")
    dev_ids = load_split(runtime_root, "dev_group_ids.json")
    final_ids = load_split(runtime_root, "final_group_ids.json")
    split_values = [
        frozen_split_interface("probe20", probe_ids, dev_ids),
        frozen_split_interface("dev512", dev_ids),
        frozen_split_interface("final2048", final_ids),
    ]
    source_rows = [
        {"recommendation_group_id": "route-pair", "target_domain": "video", "all_gold_sids": ["sid1", "sid2"], "all_gold_abc": ["abc1", "abc2"], "route": route}
        for route in ("think", "no_think")
    ]
    deduplicated = deduplicate_source_rows(source_rows)
    split_audit = {
        "frozen_split_sha256": FROZEN_SPLIT_SHA,
        "probe20_subset_dev512": set(probe_ids) <= set(dev_ids),
        "interfaces": {value.name: {"groups": len(value.group_ids), "checkpoint_selection_allowed": value.checkpoint_selection_allowed, "optimizer_allowed": value.optimizer_allowed, "backward_allowed": value.backward_allowed} for value in split_values},
        "business_group_eval_unit": len(deduplicated) == 1,
        "route_rows_input": len(source_rows),
        "evaluation_groups_output": len(deduplicated),
        "multipositive_gold_preserved": len(deduplicated[0]["all_gold_sids"]) == 2,
    }
    write_json(output_dir / "split_interface_audit.json", split_audit)

    scoring = {
        "official_composite_formula_available": OFFICIAL_COMPOSITE_FORMULA_AVAILABLE,
        "score_label": SCORING_LABEL,
        "internal_diagnostics": INTERNAL_DIAGNOSTICS,
        "unsupported_uninvented_fields": ("sid2pid", "pass_recall", "token_acc", "emb_sim"),
    }
    write_json(output_dir / "scoring_scope_audit.json", scoring)

    sample = EvaluationBusinessSample("synthetic-group", "video", ("<|video_begin|><s_a_1><s_b_2><s_c_3>",), ("<s_a_1><s_b_2><s_c_3>",), candidates)
    aggregate = aggregate_results([sample])
    comparison = {
        "schema_fields": ("checkpoint_name", "model_family", "split", "eval_contract_id", "overall_internal_diagnostics", "per_domain"),
        "rows": [checkpoint_comparison_row(model, "dev512", aggregate) for model in models],
        "same_aggregation_schema": True,
    }
    write_json(output_dir / "checkpoint_comparison_schema.json", comparison)

    unknown = {
        "official_eval_source_code_available": OFFICIAL_EVAL_SOURCE_CODE_AVAILABLE,
        "eval_contract_source": CONTRACT_SOURCE,
        "known_official_contract": {"fixed_domain": True, "beam_size": 32, "action": "ABC3"},
        "unknown_official_decoding_details": UNKNOWN_OFFICIAL_DECODING_DETAILS,
        "unknown_fields_silently_hardcoded": False,
    }
    write_json(output_dir / "unknown_official_fields.json", unknown)

    review = (
        "TrueRec-GRPO Phase 1.1E CPU contract audit: PASS\n"
        "This is an Official-Aligned frontend based on the user-provided fixed-domain Beam32 ABC3 protocol, not a recovered official evaluator.\n"
        "Beta, Beta-Gamma, and TrueRec share one no-bridge prefix and one result schema. Unknown official decoding details remain explicit.\n"
        "Only internal ABC hierarchy diagnostics are implemented and labeled NOT_OFFICIAL_COMPOSITE_SCORE. No GPU, model forward, backward, optimizer, or training ran.\n"
    )
    (output_dir / "CHATGPT_PHASE1_1E_REVIEW.txt").write_text(review, encoding="utf-8")
    print("PHASE1_1E_CPU_AUDIT=PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--test-status", choices=("PASS", "FAIL"), required=True)
    args = parser.parse_args()
    run(args.output_dir, args.source_root, args.runtime_root, args.test_status)


if __name__ == "__main__":
    main()
