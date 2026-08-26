"""Real tokenizer-only Phase-2 Teacher-CoT renderer and parity audit."""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
for path in (HERE, DATA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from beta_gamma_renderer import BetaGammaRenderer  # noqa: E402
from build_paired_curriculum_v1 import (  # noqa: E402
    assert_no_context_overflow, audit_identity, build_paired_record, publish_if_pass,
)
from phase0_teacher_cot_audit import (  # noqa: E402
    extract_teacher_cot, file_sha256, iter_lineage_teacher_rows, load_ids, load_jsonl,
)
from teacher_cot_renderer_v1 import (  # noqa: E402
    CANONICAL_SEPARATOR, assert_action_boundary, audit_sft_prefix_parity, discover_canonical_separator,
    think_rl_context_ids, transform_terminal_route, validate_teacher_cot,
)


EXPECTED_BETA_GAMMA_SHA = "72170e142a3db1ee7d0dd5b76ffb884f143a9fbf074906ea9fc91c9e8883ad28"
SOURCE = Path("/data/lf_data/onereason_recommendation_cot.jsonl")
LINEAGE = Path("/data/lf_data_versions/task_pools/懂推荐/beta版/recommendation_beta.jsonl")
BETA_GAMMA = Path("/data/lf_data_versions/alltrain/beta_gamma_v1/onereason_beta_gamma.jsonl")
DATA_ROOT = Path("/data/GRPO/truerec_grpo/data")
CURRICULUM = DATA_ROOT / "curriculum2048_v2/records.jsonl"
ORDER = DATA_ROOT / "curriculum2048_v2/epoch1_order.json"
SPLITS = DATA_ROOT / "splits"
MODEL_CONFIG = Path("/data/models/onereason-8b-pretrain-competition/config.json")
DEFAULT_OUTPUT = HERE / "results/phase2_teacher_cot_renderer"
PAIRED_OUTPUT = DATA_ROOT / "mixed_fix_v1"


def percentile(values: list[int], q: float) -> float:
    ordered = sorted(values); position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return float(ordered[low]) if low == high else ordered[low] * (high - position) + ordered[high] * (position - low)


def distribution(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values), "p50": percentile(values, .50), "p90": percentile(values, .90),
        "p95": percentile(values, .95), "p99": percentile(values, .99), "max": max(values),
        "mean": statistics.fmean(values),
    }


def teacher_map(wanted: set[str]) -> dict[str, str]:
    variants: dict[str, set[str]] = defaultdict(set)
    for row in iter_lineage_teacher_rows(SOURCE, LINEAGE):
        metadata = json.loads(row["aux_metadata_json"])
        group_id = str(metadata["recommendation_group_id"])
        if group_id in wanted:
            cot = extract_teacher_cot(str(row["output"]))
            if cot is None:
                raise ValueError(f"Teacher COT close missing: {group_id}")
            variants[group_id].add(cot)
    invalid = {group_id: len(items) for group_id, items in variants.items() if len(items) != 1}
    if invalid or set(variants) != wanted:
        raise ValueError(f"Phase0 Teacher COT contract changed: invalid={invalid} missing={len(wanted-set(variants))}")
    return {group_id: next(iter(items)) for group_id, items in variants.items()}


def beta_gamma_references(wanted: set[str]) -> dict[str, list[dict[str, Any]]]:
    if file_sha256(BETA_GAMMA) != EXPECTED_BETA_GAMMA_SHA:
        raise ValueError("Beta-Gamma source SHA drift")
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with BETA_GAMMA.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("source_segment") != "recommendation_cot":
                continue
            metadata = json.loads(row.get("aux_metadata_json") or "{}")
            group_id = str(metadata.get("recommendation_group_id", ""))
            if group_id in wanted:
                result[group_id].append(row)
    return result


def candidate_jsonl_sha(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update((json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode())
    return digest.hexdigest()


def external_reference_summary(
    records: list[dict[str, Any]], reference_ids: set[str], prefix_pass: set[str],
    prefix_fail: set[str], token_pass: set[str], token_fail: set[str],
) -> dict[str, Any]:
    all_ids = {str(row["recommendation_group_id"]) for row in records}
    if not reference_ids <= all_ids:
        raise ValueError("external reference contains unknown group")
    if prefix_pass | prefix_fail != reference_ids or token_pass | token_fail != reference_ids:
        raise ValueError("observed parity denominator must equal covered reference groups")
    if prefix_pass & prefix_fail or token_pass & token_fail:
        raise ValueError("observed parity statuses overlap")
    by_id = {str(row["recommendation_group_id"]): row for row in records}
    missing = all_ids - reference_ids

    def counts(ids: set[str]) -> dict[str, dict[str, int]]:
        return {
            key: dict(sorted(Counter(str(by_id[group_id][field]) for group_id in ids).items()))
            for key, field in (("domain", "target_domain"), ("stage", "stage"), ("hierarchy_class", "hierarchy_class"))
        }

    domain_parity = {}
    for domain in ("video", "prod", "ad", "living"):
        covered = {group_id for group_id in reference_ids if by_id[group_id]["target_domain"] == domain}
        domain_parity[domain] = {
            "reference_groups": len(covered),
            "prefix_pass": len(covered & prefix_pass), "prefix_fail": len(covered & prefix_fail),
            "token_pass": len(covered & token_pass), "token_fail": len(covered & token_fail),
        }
    return {
        "reference_groups": len(reference_ids), "reference_missing": len(missing),
        "coverage_rate": len(reference_ids) / len(all_ids),
        "observed_prefix_pass": len(prefix_pass), "observed_prefix_fail": len(prefix_fail),
        "observed_token_pass": len(token_pass), "observed_token_fail": len(token_fail),
        "covered_distribution": counts(reference_ids), "missing_distribution": counts(missing),
        "domain_observed_parity": domain_parity,
    }


def phase2b_hard_gate(
    external: dict[str, Any], canonical_separator: str, *,
    constructed_context_pass: int, constructed_context_fail: int,
    self_consistency_pass: int, self_consistency_fail: int,
    fixed_domain_pass: int, route_pass: int, action_contract: bool,
    overflow_count: int, split_failure_count: int,
) -> bool:
    return (
        external["reference_groups"] > 0
        and external["observed_prefix_fail"] == 0
        and external["observed_token_fail"] == 0
        and external["observed_prefix_pass"] == external["reference_groups"]
        and external["observed_token_pass"] == external["reference_groups"]
        and canonical_separator == CANONICAL_SEPARATOR
        and constructed_context_pass == 2048 and constructed_context_fail == 0
        and self_consistency_pass == 2048 and self_consistency_fail == 0
        and fixed_domain_pass == 2048 and route_pass == 2048 and action_contract
        and overflow_count == 0 and split_failure_count == 0
    )


def run(source_commit: str, output_dir: Path) -> dict[str, Any]:
    records = load_jsonl(CURRICULUM)
    order = list(map(str, json.loads(ORDER.read_text(encoding="utf-8"))))
    by_id = {str(row["recommendation_group_id"]): row for row in records}
    if len(records) != 2048 or len(by_id) != 2048:
        raise ValueError("Curriculum2048 identity changed")
    teachers = teacher_map(set(by_id))
    references = beta_gamma_references(set(by_id))

    separator_inputs, prefix_pass = [], set()
    prefix_fail = set()
    for group_id, rows in references.items():
        teacher = teachers[group_id]; domain = by_id[group_id]["fixed_domain_token"]
        group_ok = True
        for row in rows:
            prefix = extract_teacher_cot(str(row["output"]))
            group_ok &= prefix == teacher
            if prefix == teacher:
                separator_inputs.append((str(row["output"]), teacher, domain))
        (prefix_pass if group_ok else prefix_fail).add(group_id)
    missing_reference = set(by_id) - set(references)
    canonical_separator = discover_canonical_separator(separator_inputs)

    renderer = BetaGammaRenderer()
    max_positions = int(json.loads(MODEL_CONFIG.read_text())["max_position_embeddings"])
    paired, group_token_pass, token_failed = [], set(), set()
    fixed_domain_pass = 0
    route_pass = 0
    action_contract = True
    constructed_context_pass = 0
    constructed_context_fail = 0
    self_consistency_pass = 0
    self_consistency_fail = 0
    parity_details: dict[str, dict[str, Any]] = {}
    for record in records:
        group_id = str(record["recommendation_group_id"]); teacher = teachers[group_id]
        validate_teacher_cot(teacher)
        think_user = transform_terminal_route(str(record["user_content_nothink"]))
        route_pass += int(think_user[:-len("/think")] == record["user_content_nothink"][:-len("/no_think")])
        no_ids = renderer.rl_context_ids(record["system"], record["user_content_nothink"], record["fixed_domain_token"])
        think_ids = think_rl_context_ids(
            renderer, record["system"], think_user, teacher,
            record["fixed_domain_token"], canonical_separator,
        )
        domain_id = renderer.encode(record["fixed_domain_token"])[0]
        fixed_domain_pass += int(think_ids[-1] == domain_id)
        for gold in record["all_gold_abc"]:
            try:
                assert_action_boundary(renderer, think_ids, record["fixed_domain_token"], gold)
            except ValueError:
                action_contract = False
        if think_ids[-1:] == renderer.encode(record["fixed_domain_token"]):
            constructed_context_pass += 1
        else:
            constructed_context_fail += 1
        paired.append(build_paired_record(record, think_user, teacher, len(no_ids), len(think_ids)))
        row_results = [
            audit_sft_prefix_parity(renderer, record["system"], think_user, str(row["output"]), think_ids)
            for row in references.get(group_id, [])
        ]
        if row_results:
            if all(item.passed for item in row_results):
                group_token_pass.add(group_id)
            else:
                token_failed.add(group_id)
        synthetic_response = teacher + canonical_separator + record["fixed_domain_token"] + record["all_gold_abc"][0]
        self_result = audit_sft_prefix_parity(
            renderer, record["system"], think_user, synthetic_response, think_ids
        )
        if self_result.passed:
            self_consistency_pass += 1
        else:
            self_consistency_fail += 1
        parity_details[group_id] = {
            "no_think_context_length": len(no_ids), "think_context_length": len(think_ids),
            "delta": len(think_ids) - len(no_ids), "domain_token_id": domain_id,
            "final_context_token_id": think_ids[-1],
            "reference_prefix_length": row_results[0].reference_prefix_length if row_results else None,
            "candidate_context_length": len(think_ids),
            "token_parity": bool(row_results and all(item.passed for item in row_results)),
            "first_mismatch_index": next((item.first_mismatch_index for item in row_results if not item.passed), None),
        }

    audit_identity(records, paired, order)
    overflow = [row["recommendation_group_id"] for row in paired if row["think_context_token_count"] + 3 > max_positions]
    if not overflow:
        assert_no_context_overflow(paired, max_positions)
    split_sets = {
        "train": load_ids(SPLITS / "train_pool_group_ids.json"),
        "dev": load_ids(SPLITS / "dev_group_ids.json"),
        "final": load_ids(SPLITS / "final_group_ids.json"),
        "probe": load_ids(SPLITS / "probe20_group_ids.json"),
    }
    ids = set(by_id)
    split_audit = {
        "train_missing": len(ids - split_sets["train"]),
        "train_dev_overlap": len(ids & split_sets["dev"]),
        "train_final_overlap": len(ids & split_sets["final"]),
        "train_probe_overlap": len(ids & split_sets["probe"]),
    }
    no_lengths = [row["no_think_context_token_count"] for row in paired]
    think_lengths = [row["think_context_token_count"] for row in paired]
    deltas = [row["think_minus_nothink_token_count"] for row in paired]
    length_audit = {
        "no_think": distribution(no_lengths), "think": distribution(think_lengths),
        "think_minus_nothink": distribution(deltas), "max_position_embeddings": max_positions,
        "requires_truncation": len(overflow), "overflow_group_ids": overflow,
    }
    examples = []
    for domain in ("video", "prod", "ad", "living"):
        for group_id in sorted(g for g, row in by_id.items() if row["target_domain"] == domain)[:2]:
            row, detail = by_id[group_id], parity_details[group_id]
            examples.append({
                "recommendation_group_id": group_id, "domain": domain, "stage": row["stage"],
                "hierarchy_class": row["hierarchy_class"], **detail,
                "teacher_cot_character_count": len(teachers[group_id]),
                "canonical_separator_repr": repr(canonical_separator),
            })
    external = external_reference_summary(
        records, set(references), prefix_pass, prefix_fail, group_token_pass, token_failed
    )
    renderer_provenance = renderer.audit()
    hard_pass = phase2b_hard_gate(
        external, canonical_separator,
        constructed_context_pass=constructed_context_pass,
        constructed_context_fail=constructed_context_fail,
        self_consistency_pass=self_consistency_pass,
        self_consistency_fail=self_consistency_fail,
        fixed_domain_pass=fixed_domain_pass, route_pass=route_pass,
        action_contract=action_contract, overflow_count=len(overflow),
        split_failure_count=sum(split_audit.values()),
    )
    candidate_sha = candidate_jsonl_sha(paired)
    publish_manifest = {
        "contract": "TRUEREC-MIXED-FIX-V1-CURRICULUM-PHASE2B",
        "source_main_commit": source_commit,
        "source_curriculum": {"path": str(CURRICULUM), "sha256": file_sha256(CURRICULUM)},
        "epoch1_order": {"path": str(ORDER), "sha256": file_sha256(ORDER)},
        "teacher_cot_source": {"path": str(SOURCE), "sha256": file_sha256(SOURCE)},
        "lineage_source": {"path": str(LINEAGE), "sha256": file_sha256(LINEAGE)},
        "beta_gamma_source": {"path": str(BETA_GAMMA), "sha256": file_sha256(BETA_GAMMA)},
        "teacher_cot_authoritative_source": "PHASE0_LINEAGE",
        "beta_gamma_reference_role": "EXTERNAL_SERIALIZATION_VALIDATION",
        "canonical_separator": canonical_separator,
        "separator_source": "5,070 real Beta-Gamma recommendation_cot direct-domain response audit",
        "external_reference": external,
        "constructed_validation": {
            "context_valid_pass": constructed_context_pass,
            "context_valid_fail": constructed_context_fail,
            "sft_prefix_self_consistency_pass": self_consistency_pass,
            "sft_prefix_self_consistency_fail": self_consistency_fail,
        },
        "renderer_provenance": renderer_provenance,
    }
    published, published_sha = publish_if_pass(hard_pass, PAIRED_OUTPUT, paired, publish_manifest)
    if published and published_sha != candidate_sha:
        raise ValueError(f"published paired SHA drift: candidate={candidate_sha} published={published_sha}")
    contract = {
        "source_main_commit": source_commit,
        "curriculum2048_groups": 2048,
        "paired_groups_built_in_memory": len(paired),
        "paired_unique_groups": len({row["recommendation_group_id"] for row in paired}),
        "paired_data_published": published,
        "teacher_cot_ready": len(teachers),
        "teacher_cot_authoritative_source": "PHASE0_LINEAGE",
        "beta_gamma_reference_role": "EXTERNAL_SERIALIZATION_VALIDATION",
        "external_reference": external,
        "observed_cot_prefix_parity_pass": len(prefix_pass),
        "observed_cot_prefix_parity_fail": len(prefix_fail),
        "beta_gamma_reference_cot_groups": len(references),
        "beta_gamma_reference_missing_groups": len(missing_reference),
        "beta_gamma_reference_missing_group_ids": sorted(missing_reference),
        "canonical_separator": canonical_separator,
        "canonical_separator_repr": repr(canonical_separator),
        "canonical_separator_variants": 1,
        "separator_source": "5,070 real Beta-Gamma recommendation_cot direct-domain response audit",
        "think_route_marker_parity_pass": route_pass,
        "think_route_marker_parity_fail": 2048 - route_pass,
        "observed_sft_token_parity_pass": len(group_token_pass),
        "observed_sft_token_parity_fail": len(token_failed),
        "constructed_think_context_valid_pass": constructed_context_pass,
        "constructed_think_context_valid_fail": constructed_context_fail,
        "constructed_sft_prefix_self_consistency_pass": self_consistency_pass,
        "constructed_sft_prefix_self_consistency_fail": self_consistency_fail,
        "fixed_domain_last_token_parity_pass": fixed_domain_pass,
        "abc_three_token_contract": "PASS" if action_contract else "FAIL",
        "candidate_paired_records_sha256": candidate_sha,
        "paired_records_sha256": published_sha,
        "paired_publish_blocker": None if published else "PHASE2B_HARD_GATE_FAILED",
        "sources": {
            "teacher_cot": {"path": str(SOURCE), "sha256": file_sha256(SOURCE)},
            "lineage": {"path": str(LINEAGE), "sha256": file_sha256(LINEAGE)},
            "beta_gamma": {"path": str(BETA_GAMMA), "sha256": file_sha256(BETA_GAMMA)},
            "curriculum": {"path": str(CURRICULUM), "sha256": file_sha256(CURRICULUM)},
            "epoch1_order": {"path": str(ORDER), "sha256": file_sha256(ORDER)},
        },
        "renderer_provenance": renderer_provenance,
        "split_audit": split_audit,
        "length_audit": length_audit,
        "model_loaded": False, "gpu_started": False, "generation_started": False,
        "forward_started": False, "backward_started": False, "optimizer_steps": 0,
        "phase2_renderer_contract": "PASS" if hard_pass else "FAIL",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (("contract.json", contract), ("context_length_audit.json", length_audit), ("parity_examples.json", examples)):
        (output_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "REVIEW.txt").write_text(render_review(contract), encoding="utf-8")
    return contract


def render_review(c: dict[str, Any]) -> str:
    length, split = c["length_audit"], c["split_audit"]
    fields = {
        "SOURCE_MAIN_COMMIT": c["source_main_commit"], "CURRICULUM2048_GROUPS": 2048,
        "PAIRED_GROUPS": c["paired_groups_built_in_memory"], "PAIRED_UNIQUE_GROUPS": c["paired_unique_groups"],
        "TEACHER_COT_AUTHORITATIVE_SOURCE": c["teacher_cot_authoritative_source"],
        "TEACHER_COT_READY": c["teacher_cot_ready"],
        "BETA_GAMMA_REFERENCE_ROLE": c["beta_gamma_reference_role"],
        "BETA_GAMMA_REFERENCE_GROUPS": c["external_reference"]["reference_groups"],
        "BETA_GAMMA_REFERENCE_MISSING": c["external_reference"]["reference_missing"],
        "BETA_GAMMA_REFERENCE_COVERAGE_RATE": c["external_reference"]["coverage_rate"],
        "OBSERVED_COT_PREFIX_PARITY_PASS": c["observed_cot_prefix_parity_pass"],
        "OBSERVED_COT_PREFIX_PARITY_FAIL": c["observed_cot_prefix_parity_fail"],
        "OBSERVED_SFT_TOKEN_PARITY_PASS": c["observed_sft_token_parity_pass"],
        "OBSERVED_SFT_TOKEN_PARITY_FAIL": c["observed_sft_token_parity_fail"],
        "CANONICAL_SEPARATOR_REPR": c["canonical_separator_repr"],
        "CANONICAL_SEPARATOR_VARIANTS": c["canonical_separator_variants"],
        "CONSTRUCTED_THINK_CONTEXT_VALID_PASS": c["constructed_think_context_valid_pass"],
        "CONSTRUCTED_THINK_CONTEXT_VALID_FAIL": c["constructed_think_context_valid_fail"],
        "CONSTRUCTED_SFT_PREFIX_SELF_CONSISTENCY_PASS": c["constructed_sft_prefix_self_consistency_pass"],
        "CONSTRUCTED_SFT_PREFIX_SELF_CONSISTENCY_FAIL": c["constructed_sft_prefix_self_consistency_fail"],
        "THINK_ROUTE_MARKER_PARITY_PASS": c["think_route_marker_parity_pass"],
        "THINK_ROUTE_MARKER_PARITY_FAIL": c["think_route_marker_parity_fail"],
        "FIXED_DOMAIN_LAST_TOKEN_PARITY_PASS": c["fixed_domain_last_token_parity_pass"],
        "ABC_THREE_TOKEN_CONTRACT": c["abc_three_token_contract"],
        "NOTHINK_CONTEXT_P50": length["no_think"]["p50"], "NOTHINK_CONTEXT_P99": length["no_think"]["p99"],
        "NOTHINK_CONTEXT_MAX": length["no_think"]["max"], "THINK_CONTEXT_P50": length["think"]["p50"],
        "THINK_CONTEXT_P99": length["think"]["p99"], "THINK_CONTEXT_MAX": length["think"]["max"],
        "THINK_MINUS_NOTHINK_P50": length["think_minus_nothink"]["p50"],
        "THINK_MINUS_NOTHINK_P99": length["think_minus_nothink"]["p99"],
        "THINK_MINUS_NOTHINK_MAX": length["think_minus_nothink"]["max"],
        "THINK_CONTEXT_REQUIRES_TRUNCATION": length["requires_truncation"],
        "TRAIN_DEV_OVERLAP": split["train_dev_overlap"], "TRAIN_FINAL_OVERLAP": split["train_final_overlap"],
        "TRAIN_PROBE_OVERLAP": split["train_probe_overlap"],
        "PAIRED_DATA_PUBLISHED": "YES" if c["paired_data_published"] else "NO",
        "PAIRED_RECORDS_SHA256": c["paired_records_sha256"] or "",
        "MODEL_LOADED": "NO", "GPU_STARTED": "NO", "GENERATION_STARTED": "NO", "FORWARD_STARTED": "NO",
        "BACKWARD_STARTED": "NO", "OPTIMIZER_STEPS": 0,
        "PHASE2_RENDERER_CONTRACT": c["phase2_renderer_contract"],
    }
    return "\n".join(f"{key}={value}" for key, value in fields.items()) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    contract = run(args.source_commit, args.output_dir); print(render_review(contract), end="")
    if contract["phase2_renderer_contract"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
