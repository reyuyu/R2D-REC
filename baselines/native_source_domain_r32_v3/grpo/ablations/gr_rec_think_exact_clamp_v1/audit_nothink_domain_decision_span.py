#!/usr/bin/env python3
"""CPU/tokenizer-only audit of the NoThink domain decision span."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DOMAINS = ("video", "prod", "ad", "living")
DOMAIN_DECLARATIONS = {
    "video": {"span": "视频", "declaration": "该用户最近喜欢的视频有: "},
    "prod": {"span": "商品", "declaration": "该用户最近点击了商品: "},
    "ad": {"span": "广告", "declaration": "该用户最近感兴趣的广告有: "},
    "living": {"span": "主播", "declaration": "该用户最近首次打赏了主播: "},
}
SID_RE = re.compile(
    r"<\|(?P<domain>video|prod|ad|living)_begin\|>"
    r"<s_a_(?P<a>\d+)><s_b_(?P<b>\d+)><s_c_(?P<c>\d+)>"
)
REAL_GROUP_RE = re.compile(r"[0-9a-f]{64}")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object at {path}:{line_number}")
                yield value


def final_sid(text: str) -> tuple[str, int, int, int] | None:
    tail = text
    close = text.rfind("</think>")
    if close >= 0:
        tail = text[close + len("</think>") :]
    matches = list(SID_RE.finditer(tail)) or list(SID_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    return (
        match.group("domain"),
        int(match.group("a")),
        int(match.group("b")),
        int(match.group("c")),
    )


def text_domain_before_final_sid(text: str) -> tuple[str | None, str | None]:
    matches = list(SID_RE.finditer(text))
    prefix = text[: matches[-1].start()] if matches else text
    close = prefix.rfind("</think>")
    decision_prefix = prefix[close + len("</think>") :] if close >= 0 else prefix
    found = []
    for domain, definition in DOMAIN_DECLARATIONS.items():
        offset = decision_prefix.rfind(definition["span"])
        if offset >= 0:
            found.append((offset, domain))
    if not found:
        return None, None
    _offset, domain = max(found)
    return domain, decision_prefix.strip()


def ratio(values: Iterable[bool]) -> dict[str, int | float | None]:
    values = list(values)
    numerator = sum(values)
    return {
        "numerator": numerator,
        "denominator": len(values),
        "rate": numerator / len(values) if values else None,
    }


def empty_confusion() -> Counter[str]:
    return Counter({
        "text_correct__sid_correct": 0,
        "text_correct__sid_wrong": 0,
        "text_wrong__sid_correct": 0,
        "text_wrong__sid_wrong": 0,
    })


def confusion(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = empty_confusion()
    for record in records:
        text = record["text_domain"]
        sid = record["sid_domain"]
        target = record["target_domain"]
        if text is None or sid is None or target is None:
            continue
        text_state = "correct" if text == target else "wrong"
        sid_state = "correct" if sid == target else "wrong"
        counts[f"text_{text_state}__sid_{sid_state}"] += 1
    return dict(counts)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    comparable_text_sid = [r for r in records if r["text_domain"] and r["sid_domain"]]
    comparable_text_target = [r for r in records if r["text_domain"] and r["target_domain"]]
    comparable_sid_target = [r for r in records if r["sid_domain"] and r["target_domain"]]
    return {
        "candidate_count": len(records),
        "text_domain_defined": ratio(r["text_domain"] is not None for r in records),
        "sid_domain_defined": ratio(r["sid_domain"] is not None for r in records),
        "text_domain_equals_sid_domain": ratio(
            r["text_domain"] == r["sid_domain"] for r in comparable_text_sid
        ),
        "text_domain_equals_target_domain": ratio(
            r["text_domain"] == r["target_domain"] for r in comparable_text_target
        ),
        "sid_domain_equals_target_domain": ratio(
            r["sid_domain"] == r["target_domain"] for r in comparable_sid_target
        ),
        "confusion": confusion(records),
        "undefined_for_confusion": sum(
            r["text_domain"] is None or r["sid_domain"] is None or r["target_domain"] is None
            for r in records
        ),
    }


def audit_sft(path: Path) -> dict[str, Any]:
    templates: dict[str, Counter[str]] = defaultdict(Counter)
    declarations: Counter[str] = Counter()
    total = direct_sid = missing_sid = missing_declaration = 0
    for row in read_jsonl(path):
        if row.get("source_segment") != "recommendation_nocot":
            continue
        total += 1
        output = str(row.get("output", ""))
        matches = list(SID_RE.finditer(output))
        if not matches:
            missing_sid += 1
            continue
        match = matches[-1]
        domain = match.group("domain")
        prefix = output[: match.start()]
        text_domain, declaration = text_domain_before_final_sid(output)
        if not prefix.strip():
            direct_sid += 1
        if text_domain is None:
            missing_declaration += 1
        else:
            declarations[text_domain] += 1
        templates[domain][prefix] += 1

    per_domain = {}
    for domain in DOMAINS:
        domain_templates = templates[domain]
        per_domain[domain] = {
            "sample_count": sum(domain_templates.values()),
            "natural_language_span": DOMAIN_DECLARATIONS[domain]["span"],
            "expected_declaration": DOMAIN_DECLARATIONS[domain]["declaration"],
            "template_count": len(domain_templates),
            "fixed_within_domain": len(domain_templates) == 1,
            "templates": [
                {"count": count, "template": prefix + "{SID}"}
                for prefix, count in domain_templates.most_common()
            ],
        }
    return {
        "path": str(path),
        "selection": "source_segment == recommendation_nocot",
        "total_samples": total,
        "samples_with_sid": total - missing_sid,
        "samples_with_sid_preceded_by_known_domain_declaration": sum(declarations.values()),
        "domain_declaration_rate": sum(declarations.values()) / total if total else None,
        "missing_sid_count": missing_sid,
        "missing_known_declaration_count": missing_declaration,
        "direct_sid_without_natural_language_count": direct_sid,
        "direct_sid_without_natural_language_rate": direct_sid / total if total else None,
        "distinct_templates_overall": sum(len(value) for value in templates.values()),
        "multiple_synonymous_expressions_observed": any(len(value) > 1 for value in templates.values()),
        "per_domain": per_domain,
    }


def target_domain_from_golds(golds: Any) -> str | None:
    if not isinstance(golds, list) or not golds or not all(isinstance(x, str) for x in golds):
        return None
    matches = [SID_RE.fullmatch(value) for value in golds]
    if not all(matches):
        return None
    domains = {match.group("domain") for match in matches if match is not None}
    return next(iter(domains)) if len(domains) == 1 else None


def source_family(run_name: str) -> str:
    if "DSR-SIMPLE" in run_name:
        return "DSR-Simple"
    if "DSR" in run_name:
        return "DSR"
    return "GR_REC"


def audit_rollouts(runs_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for path in sorted(runs_root.glob("*/traces/traces.jsonl")):
        accepted = 0
        run_name = path.parents[1].name
        for event in read_jsonl(path):
            if event.get("route") != "no_think":
                continue
            group_id = event.get("group_id")
            target = target_domain_from_golds(event.get("gold_sids"))
            if not isinstance(group_id, str) or not REAL_GROUP_RE.fullmatch(group_id) or target is None:
                continue
            for candidate in event.get("candidates", []):
                completion = candidate.get("completion")
                if not isinstance(completion, str):
                    continue
                sid = final_sid(completion)
                text, declaration = text_domain_before_final_sid(completion)
                reward = candidate.get("reward")
                records.append({
                    "source_family": source_family(run_name),
                    "run": run_name,
                    "trace": str(path),
                    "rollout_id": event.get("rollout_id"),
                    "step": event.get("step"),
                    "group_id": group_id,
                    "candidate_id": candidate.get("candidate_id"),
                    "reward": float(reward) if isinstance(reward, (int, float)) else None,
                    "target_domain": target,
                    "text_domain": text,
                    "text_declaration": declaration,
                    "sid_domain": sid[0] if sid else None,
                })
                accepted += 1
        if accepted:
            files.append({
                "path": str(path),
                "run": run_name,
                "source_family": source_family(run_name),
                "candidate_count": accepted,
            })

    by_domain = {domain: summarize_records([r for r in records if r["target_domain"] == domain]) for domain in DOMAINS}
    by_family = {
        family: summarize_records([r for r in records if r["source_family"] == family])
        for family in ("GR_REC", "DSR", "DSR-Simple")
    }
    reward_minus_quarter = [r for r in records if r["reward"] == -0.25]
    reward_nonnegative = [r for r in records if r["reward"] is not None and r["reward"] >= 0]
    declaration_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        if record["text_domain"] and record["text_declaration"]:
            declaration_counts[record["text_domain"]][record["text_declaration"]] += 1
    summary = {
        "runs_root": str(runs_root),
        "eligibility": (
            "route=no_think; 64-hex real group_id; non-empty string gold_sids with one domain; "
            "candidate has recoverable full completion"
        ),
        "source_files": files,
        "overall": summarize_records(records),
        "by_target_domain": by_domain,
        "by_source_family": by_family,
        "natural_language_declarations_observed": {
            domain: [
                {"declaration": declaration, "count": count}
                for declaration, count in declaration_counts[domain].most_common()
            ]
            for domain in DOMAINS
        },
        "reward_minus_0_25": summarize_records(reward_minus_quarter),
        "reward_gte_0": summarize_records(reward_nonnegative),
        "other_reward_candidate_count": len(records) - len(reward_minus_quarter) - len(reward_nonnegative),
    }
    return summary, records


def audit_tokenization(tokenizer_path: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    result = {}
    for domain, definition in DOMAIN_DECLARATIONS.items():
        span = definition["span"]
        token_ids = tokenizer.encode(span, add_special_tokens=False)
        result[domain] = {
            "span": span,
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "decoded_tokens": [tokenizer.decode([token_id]) for token_id in token_ids],
            "full_declaration": definition["declaration"],
            "full_declaration_token_ids": tokenizer.encode(
                definition["declaration"], add_special_tokens=False
            ),
        }
        result[domain]["full_declaration_token_count"] = len(
            result[domain]["full_declaration_token_ids"]
        )
    return {
        "tokenizer_path": str(tokenizer_path),
        "model_weights_loaded": False,
        "spans": result,
    }


def audit_sid_saturation(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "source": str(path),
            "available": False,
            "text_domain_probability": "TEXT-DOMAIN PROBABILITY NOT YET OBSERVED",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    probabilities = []
    for group in payload.get("domain_only_groups", []):
        probabilities.extend(
            candidate["old_probability"]
            for candidate in group.get("domain_candidates", [])
            if isinstance(candidate.get("old_probability"), (int, float))
        )
    return {
        "source": str(path),
        "available": bool(probabilities),
        "evidence_scope": "Read-only reuse of saved sampled final-SID Domain probabilities only; gradient geometry is excluded.",
        "sample_count": len(probabilities),
        "minimum": min(probabilities) if probabilities else None,
        "median": statistics.median(probabilities) if probabilities else None,
        "maximum": max(probabilities) if probabilities else None,
        "count_gte_0_9999": sum(value >= 0.9999 for value in probabilities),
        "rate_gte_0_9999": sum(value >= 0.9999 for value in probabilities) / len(probabilities) if probabilities else None,
        "text_domain_probability": "TEXT-DOMAIN PROBABILITY NOT YET OBSERVED",
    }


def choose_verdict(sft: dict[str, Any], rollouts: dict[str, Any], saturation: dict[str, Any]) -> tuple[str, list[str]]:
    templates_fixed = all(item["fixed_within_domain"] for item in sft["per_domain"].values())
    text_sid_rate = rollouts["overall"]["text_domain_equals_sid_domain"]["rate"]
    text_defined_rate = rollouts["overall"]["text_domain_defined"]["rate"]
    saturated_rate = saturation.get("rate_gte_0_9999")
    evidence = [
        f"SFT known-declaration rate={sft['domain_declaration_rate']}",
        f"all four SFT domain templates fixed={templates_fixed}",
        f"historical text/SID agreement={text_sid_rate}",
        f"historical text-domain defined rate={text_defined_rate}",
        f"saved SID Domain probability >=0.9999 rate={saturated_rate}",
        "the natural-language declaration precedes the final SID Domain token",
        "TEXT-DOMAIN PROBABILITY NOT YET OBSERVED",
    ]
    if (
        sft["domain_declaration_rate"] is not None
        and sft["domain_declaration_rate"] >= 0.99
        and templates_fixed
        and text_sid_rate is not None
        and text_sid_rate >= 0.99
        and text_defined_rate is not None
        and text_defined_rate >= 0.99
        and saturated_rate is not None
        and saturated_rate >= 0.75
    ):
        return "TEXT_DOMAIN_IS_PRIMARY_DECISION", evidence
    if sft["domain_declaration_rate"] is not None and sft["domain_declaration_rate"] < 0.5:
        return "SID_DOMAIN_IS_PRIMARY_DECISION", evidence
    return "MIXED_OR_AMBIGUOUS", evidence


def validate(result: dict[str, Any]) -> None:
    sft = result["sft_template_statistics"]
    if sft["total_samples"] <= 0:
        raise RuntimeError("No recommendation_nocot SFT samples found")
    if set(sft["per_domain"]) != set(DOMAINS):
        raise RuntimeError("SFT domain coverage is incomplete")
    rollouts = result["historical_rollout_consistency"]
    if rollouts["overall"]["candidate_count"] <= 0:
        raise RuntimeError("No eligible historical NoThink candidates found")
    for subset in (rollouts["overall"], rollouts["reward_minus_0_25"], rollouts["reward_gte_0"]):
        if sum(subset["confusion"].values()) + subset["undefined_for_confusion"] != subset["candidate_count"]:
            raise AssertionError("Confusion accounting does not reconcile")


def main() -> None:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sft-data",
        type=Path,
        default=Path("/data/lf_data_versions/alltrain/bata_baseline_v1/onereason_bata_baseline.jsonl"),
    )
    parser.add_argument("--runs-root", type=Path, default=Path("/data/GRPO/runs"))
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("/data/models/onereason-8b-pretrain-competition"),
    )
    parser.add_argument(
        "--sid-saturation-evidence",
        type=Path,
        default=Path(
            "/data/GRPO/audits/GR_REC_ThinkExactClamp_Ablation_v1/"
            "gpu_domain_geometry_diagnostic_g8_seed20260816_20260821.attempt2-float32-cosine.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_root / "results" / "nothink_domain_decision_span_audit.json",
    )
    args = parser.parse_args()

    sft = audit_sft(args.sft_data)
    rollouts, _records = audit_rollouts(args.runs_root)
    tokenization = audit_tokenization(args.tokenizer)
    saturation = audit_sid_saturation(args.sid_saturation_evidence)
    verdict, evidence = choose_verdict(sft, rollouts, saturation)
    result = {
        "audit": "NoThink Domain Decision-Span Audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "gpu_used": False,
            "model_weights_loaded": False,
            "training_started": False,
            "training_algorithm_modified": False,
        },
        "sft_template_statistics": sft,
        "text_domain_vocabulary_and_tokenization": tokenization,
        "historical_rollout_consistency": rollouts,
        "known_sid_domain_saturation_evidence": saturation,
        "final_verdict": verdict,
        "verdict_evidence": evidence,
        "algorithm_change": "NONE; audit-only evidence",
    }
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
