"""Strict Action Set-F1 reward and local constraint attribution."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from user_common import SID_FULL_RE, SID_RE, f1_from_totals, node_text_span, parse_final_json, prefix_invalid_component
from user_constraints import ConstraintViolation, violation
from user_span_attribution import TokenSpanMapper


@dataclass
class ActionRewardResult:
    reward: float
    f1: float
    precision: float
    recall: float
    exact_set_match: bool
    format_valid: bool
    predicted_sids: list[str]
    pred_sids_raw: list[str]
    pred_sids_unique: list[str]
    gold_sids: list[str]
    true_positive_sids: list[str]
    false_positive_sids: list[str]
    missed_gold_sids: list[str]
    recoverable_sids: list[str]
    hallucinated_sids: list[str]
    hallucination_count: int
    hallucination_rate: float
    wrong_selection_sids: list[str]
    duplicate_sids: list[str]
    duplicate_value_count: int
    duplicate_occurrence_count: int
    sid_occurrences: list[dict]
    parser_errors: list[str]
    diagnostics: dict
    violations: list[ConstraintViolation]

    def to_dict(self):
        result = asdict(self)
        return result


def _format_violation(kind: str, span: tuple[int, int], message: str) -> ConstraintViolation:
    return violation(kind, message, span, "action")


def score_action(completion: str, sample: dict, tokenizer=None) -> ActionRewardResult:
    parsed = parse_final_json(completion)
    parser_errors = list(parsed.errors)
    violations: list[ConstraintViolation] = []
    occurrences: list[tuple[str, tuple[int, int]]] = []

    for error in parsed.errors:
        span = (parsed.candidate_start, max(parsed.candidate_start + 1, parsed.candidate_end))
        violations.append(_format_violation(error, span, error.replace("_", " ")))

    strict_array = parsed.strict and isinstance(parsed.value, list)
    if parsed.node is not None and not isinstance(parsed.value, list):
        parser_errors.append("top_level_not_array")
        violations.append(_format_violation("top_level_not_array", (parsed.node.start, parsed.node.end), "Action output must be a JSON array"))
    elif parsed.node is not None:
        if not parsed.node.children:
            parser_errors.append("empty_array")
        for node in parsed.node.children:
            span = node_text_span(node)
            if not isinstance(node.value, str):
                parser_errors.append("non_string_array_element")
                violations.append(_format_violation("non_string_array_element", span, "Array element is not a string"))
            elif SID_FULL_RE.fullmatch(node.value):
                occurrences.append((node.value, span))
            else:
                kind = "invalid_sid_string" if "<s_" in node.value or "_begin|>" in node.value else "invalid_output_element"
                parser_errors.append(kind)
                violations.append(_format_violation(kind, span, "Array string is not one complete SID"))

    # A structurally reliable array can still contain invalid elements. They are
    # excluded from Set-F1 and diagnosed, without discarding its valid SIDs.
    fatal_errors = {"missing_json", "malformed_json", "extra_text_before_json", "extra_text_after_json", "top_level_not_array"}
    format_valid = strict_array and not any(error in fatal_errors for error in parser_errors)
    predicted_order = [sid for sid, _ in occurrences]
    predicted_unique = list(dict.fromkeys(predicted_order)) if format_valid else []
    predicted_set = set(predicted_unique)
    gold_set = set(sample.get("gold_sids", []))
    history_set = set(sample.get("history_sids", []))
    matched = len(predicted_set & gold_set)
    precision, recall, reward = f1_from_totals(matched, len(predicted_set), len(gold_set)) if format_valid else (0.0, 0.0, 0.0)

    hallucinated = sorted(predicted_set - history_set)
    wrong = sorted((predicted_set & history_set) - gold_set)
    counts = Counter(predicted_order)
    duplicates = sorted(sid for sid, count in counts.items() if count > 1)
    occurrence_records: list[dict] = []

    seen: Counter[str] = Counter()
    for sid, span in occurrences:
        seen[sid] += 1
        occurrence_records.append(
            {
                "sid": sid,
                "occurrence": seen[sid],
                "char_start": span[0],
                "char_end": span[1],
                "token_start": None,
                "token_end": None,
                "token_ids": [],
            }
        )
        if seen[sid] > 1:
            violations.append(violation("duplicate_sid", "Duplicate SID occurrence", span, "action", sid=sid, occurrence=seen[sid]))
        if sid in hallucinated:
            component, penalized = prefix_invalid_component(sid, history_set)
            violations.append(
                violation(
                    "hallucinated_sid",
                    "SID does not occur in user history",
                    span,
                    "action",
                    sid=sid,
                    first_invalid_component=component,
                    penalized_components=penalized,
                )
            )
        elif sid in wrong:
            violations.append(violation("wrong_selection_sid", "History SID is not in gold set", span, "action", sid=sid))

    if tokenizer is not None:
        mapper = TokenSpanMapper(tokenizer, completion)
        for item in violations:
            item.attribute(mapper)
        for item in occurrence_records:
            span = mapper.map(item["char_start"], item["char_end"])
            item.update(token_start=span.start, token_end=span.end, token_ids=list(span.token_ids))

    true_positive = sorted(predicted_set & gold_set)
    false_positive = sorted(predicted_set - gold_set)
    missed = sorted(gold_set - predicted_set)
    duplicate_occurrences = sum(max(0, count - 1) for count in counts.values())

    return ActionRewardResult(
        reward=reward,
        f1=reward,
        precision=precision,
        recall=recall,
        exact_set_match=format_valid and predicted_set == gold_set,
        format_valid=format_valid,
        predicted_sids=predicted_unique,
        pred_sids_raw=predicted_order,
        pred_sids_unique=predicted_unique,
        gold_sids=sorted(gold_set),
        true_positive_sids=true_positive,
        false_positive_sids=false_positive,
        missed_gold_sids=missed,
        recoverable_sids=SID_RE.findall(completion),
        hallucinated_sids=hallucinated,
        hallucination_count=len(hallucinated),
        hallucination_rate=len(hallucinated) / len(predicted_set) if predicted_set else 0.0,
        wrong_selection_sids=wrong,
        duplicate_sids=duplicates,
        duplicate_value_count=len(duplicates),
        duplicate_occurrence_count=duplicate_occurrences,
        sid_occurrences=occurrence_records,
        parser_errors=parser_errors,
        diagnostics={
            "valid_json_array": strict_array,
            "invalid_json": any(error in {"missing_json", "malformed_json"} for error in parser_errors),
            "not_array": "top_level_not_array" in parser_errors,
            "empty_array": "empty_array" in parser_errors,
            "extra_text_before_json": "extra_text_before_json" in parser_errors,
            "extra_text_after_json": "extra_text_after_json" in parser_errors,
            "non_string_element_count": parser_errors.count("non_string_array_element"),
            "valid_sid_element_count": len(occurrences),
            "invalid_sid_string_count": parser_errors.count("invalid_sid_string"),
            "invalid_output_element_count": parser_errors.count("invalid_output_element"),
        },
        violations=violations,
    )
