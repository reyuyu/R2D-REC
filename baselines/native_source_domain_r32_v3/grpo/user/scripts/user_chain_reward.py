"""Approximate benchmark-aligned Chain reward with ordered matching and grounding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from user_common import (
    DATE_RE,
    SID_RE,
    JsonNode,
    decoded_string_span,
    f1_from_totals,
    node_text_span,
    normalize_text,
    parse_final_json,
    rouge_l_f1,
    set_f1,
)
from user_constraints import ConstraintViolation, violation
from user_span_attribution import TokenSpanMapper


@dataclass
class ChainMatch:
    predicted_index: int
    gold_index: int
    action_similarity: float
    logic_token_f1: float
    logic_rouge_l_f1: float
    logic_similarity: float


@dataclass
class ChainRewardResult:
    reward: float
    total_reward: float
    action_precision: float
    action_recall: float
    action_f1: float
    logic_precision: float
    logic_recall: float
    logic_f1: float
    format_valid: bool
    parser_errors: list[str]
    diagnostics: dict[str, Any]
    action_similarity_matrix: list[list[float]]
    logic_similarity_matrix: list[list[dict[str, float]]]
    action_alignment: dict[str, float]
    logic_alignment: dict[str, float]
    matches: list[ChainMatch]
    unmatched_gold: list[int]
    unmatched_pred: list[int]
    predicted_events: list[dict[str, str]]
    event_spans: list[dict[str, Any]]
    grounding_status: list[dict[str, Any]]
    violations: list[ConstraintViolation]

    def to_dict(self):
        return asdict(self)


def ordered_action_matching(predicted: list[dict], gold: list[dict]) -> list[tuple[int, int, float]]:
    """Maximum-sum monotonic one-to-one matching; zero-similarity pairs stay unmatched."""
    m, n = len(predicted), len(gold)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    choice = [["" for _ in range(n + 1)] for _ in range(m + 1)]
    sims = [[set_f1(p.get("action", ""), g.get("action", "")) for g in gold] for p in predicted]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            options = [(dp[i - 1][j], "p"), (dp[i][j - 1], "g")]
            if sims[i - 1][j - 1] > 0:
                options.append((dp[i - 1][j - 1] + sims[i - 1][j - 1], "m"))
            value, selected = max(options, key=lambda item: (item[0], item[1] == "m", item[1] == "p"))
            dp[i][j], choice[i][j] = value, selected
    matches: list[tuple[int, int, float]] = []
    i, j = m, n
    while i and j:
        selected = choice[i][j]
        if selected == "m":
            matches.append((i - 1, j - 1, sims[i - 1][j - 1]))
            i -= 1
            j -= 1
        elif selected == "p":
            i -= 1
        else:
            j -= 1
    return list(reversed(matches))


def _chain_nodes(root: JsonNode) -> tuple[JsonNode | None, list[JsonNode]]:
    chain = root.member("logic_chain") if isinstance(root.value, dict) else None
    events = chain.member("events") if chain and isinstance(chain.value, dict) else None
    return chain, events.children if events and isinstance(events.value, list) else []


def _field_span(node: JsonNode, field: str) -> tuple[int, int]:
    child = node.member(field)
    return node_text_span(child) if child else (node.start, node.end)


def _part_spans(text: str, node: JsonNode, action: str) -> list[tuple[str, tuple[int, int], int]]:
    cursor = 0
    output = []
    for original in action.split("；"):
        part = original.strip()
        position = action.find(part, cursor)
        if position < 0:
            position = cursor
        output.append((part, decoded_string_span(text, node, position, position + len(part)), position))
        cursor = position + len(part)
    return output


def _validate_schema(parsed, violations: list[ConstraintViolation]) -> tuple[list[dict[str, str]], list[JsonNode], list[str]]:
    errors = list(parsed.errors)
    if parsed.node is None:
        return [], [], errors
    if not isinstance(parsed.value, dict):
        errors.append("top_level_not_object")
        violations.append(violation("top_level_not_object", "Chain output must be an object", (parsed.node.start, parsed.node.end), "chain"))
        return [], [], errors
    chain, nodes = _chain_nodes(parsed.node)
    if chain is None or not isinstance(chain.value, dict):
        errors.append("missing_logic_chain")
        violations.append(violation("missing_logic_chain", "Missing logic_chain object", (parsed.node.start, parsed.node.end), "chain"))
    events_node = chain.member("events") if chain else None
    if events_node is None:
        errors.append("missing_events")
        violations.append(violation("missing_events", "Missing events field", (chain.start, chain.end) if chain else (parsed.node.start, parsed.node.end), "chain"))
        return [], [], errors
    if not isinstance(events_node.value, list):
        errors.append("events_not_list")
        violations.append(violation("events_not_list", "Events must be a list", (events_node.start, events_node.end), "chain"))
        return [], [], errors
    if not nodes:
        errors.append("empty_events")
        violations.append(violation("empty_events", "Events list is empty", (events_node.start, events_node.end), "chain"))
    if len(nodes) > 5:
        errors.append("too_many_events")
    events: list[dict[str, str]] = []
    valid_nodes: list[JsonNode] = []
    for index, node in enumerate(nodes):
        if not isinstance(node.value, dict):
            errors.append("invalid_event")
            violations.append(violation("invalid_event", "Event must be an object", (node.start, node.end), "chain", event_index=index))
            continue
        event: dict[str, str] = {}
        for field in ("date", "action", "logic"):
            child = node.member(field)
            if child is None or not isinstance(child.value, str):
                kind = f"missing_{field}" if child is None else f"invalid_{field}"
                errors.append(kind)
                violations.append(violation(kind, f"Event {field} must be a string", _field_span(node, field), "chain", event_index=index))
            else:
                event[field] = child.value
        if "date" in event and not DATE_RE.fullmatch(event["date"]):
            errors.append("invalid_date_format")
            violations.append(violation("invalid_date_format", "Date must be YYYY-MM-DD", _field_span(node, "date"), "chain", event_index=index))
        events.append(event)
        valid_nodes.append(node)
    return events, valid_nodes, errors


def _event_spans(completion: str, events: list[dict[str, str]], nodes: list[JsonNode], tokenizer=None):
    mapper = TokenSpanMapper(tokenizer, completion) if tokenizer is not None else None
    output = []
    for index, (event, node) in enumerate(zip(events, nodes)):
        fields = {}
        for name in ("date", "action", "logic"):
            child = node.member(name)
            if child is None:
                continue
            start, end = node_text_span(child)
            record: dict[str, Any] = {"char_start": start, "char_end": end}
            if mapper:
                span = mapper.map(start, end)
                record.update(token_start=span.start, token_end=span.end, token_ids=list(span.token_ids))
            fields[name] = record
        sid_spans = []
        action_node = node.member("action")
        if action_node and "action" in event:
            for match in SID_RE.finditer(event["action"]):
                start, end = decoded_string_span(completion, action_node, match.start(), match.end())
                record = {"sid": match.group(), "char_start": start, "char_end": end}
                if mapper:
                    span = mapper.map(start, end)
                    record.update(token_start=span.start, token_end=span.end, token_ids=list(span.token_ids))
                sid_spans.append(record)
        whole: dict[str, Any] = {"char_start": node.start, "char_end": node.end}
        if mapper:
            span = mapper.map(node.start, node.end)
            whole.update(token_start=span.start, token_end=span.end, token_ids=list(span.token_ids))
        output.append({"event_index": index, "event": whole, "fields": fields, "sids": sid_spans})
    return output


def _ground_events(completion: str, events: list[dict[str, str]], nodes: list[JsonNode], sample: dict, violations: list[ConstraintViolation]):
    history = sample.get("history_events", [])
    history_sids = set(sample.get("history_sids", []))
    status = []
    for event_index, (event, node) in enumerate(zip(events, nodes)):
        action_node = node.member("action")
        if action_node is None or "action" not in event:
            status.append({"event_index": event_index, "status": "ungrounded", "parts": []})
            continue
        part_status = []
        for part, span, decoded_position in _part_spans(completion, action_node, event["action"]):
            date = event.get("date", "")
            sids = SID_RE.findall(part)
            state, reason = "grounded", "exact"
            if sids:
                unseen = [sid for sid in sids if sid not in history_sids]
                if unseen:
                    state, reason = "ungrounded", "hallucinated_sid"
                    for sid in unseen:
                        offset = part.find(sid)
                        sid_span = decoded_string_span(completion, action_node, decoded_position + offset, decoded_position + offset + len(sid))
                        violations.append(violation("hallucinated_sid", "Chain SID does not occur in history", sid_span, "chain", event_index=event_index, sid=sid))
                else:
                    candidates = [row for row in history if any(sid in row.get("raw", "") for sid in sids)]
                    dated = [row for row in candidates if row.get("date") == date]
                    if not dated:
                        state, reason = "ungrounded", "date_mismatch"
                        violations.append(violation("date_mismatch", "SID exists but not on predicted date", _field_span(node, "date"), "chain", event_index=event_index))
                    elif not any(row.get("raw", "").strip() == part for row in dated):
                        state, reason = "partially_grounded", "action_mismatch"
                        violations.append(violation("action_mismatch", "Date and SID match but action text does not", span, "chain", event_index=event_index))
            else:
                exact = [row for row in history if row.get("raw", "").strip() == part]
                if not exact:
                    state, reason = "ungrounded", "action_mismatch"
                    violations.append(violation("action_mismatch", "Action does not occur in history", span, "chain", event_index=event_index))
                elif not any(row.get("date") == date for row in exact):
                    state, reason = "ungrounded", "date_mismatch"
                    violations.append(violation("date_mismatch", "Action exists but not on predicted date", _field_span(node, "date"), "chain", event_index=event_index))
            part_status.append({"action": part, "status": state, "reason": reason})
        event_state = "grounded" if all(x["status"] == "grounded" for x in part_status) else ("partially_grounded" if any(x["status"] != "ungrounded" for x in part_status) else "ungrounded")
        status.append({"event_index": event_index, "status": event_state, "parts": part_status})
    return status


def score_chain(completion: str, sample: dict, tokenizer=None) -> ChainRewardResult:
    parsed = parse_final_json(completion)
    violations: list[ConstraintViolation] = []
    for error in parsed.errors:
        violations.append(violation(error, error.replace("_", " "), (parsed.candidate_start, max(parsed.candidate_start + 1, parsed.candidate_end)), "chain"))
    events, nodes, errors = _validate_schema(parsed, violations)
    if any(error in {"missing_json", "malformed_json"} for error in errors):
        errors.append("invalid_json")
    nonfatal = {"too_many_events"}
    format_valid = parsed.strict and not any(error not in nonfatal for error in errors)
    gold = sample.get("gold_events", [])
    action_matrix = [[set_f1(p.get("action", ""), g.get("action", "")) for g in gold] for p in events]
    logic_matrix = [
        [
            {
                "token_f1": set_f1(p.get("logic", ""), g.get("logic", "")),
                "rouge_l_f1": rouge_l_f1(p.get("logic", ""), g.get("logic", "")),
            }
            for g in gold
        ]
        for p in events
    ]
    for row in logic_matrix:
        for cell in row:
            cell["logic_similarity"] = 0.5 * cell["token_f1"] + 0.5 * cell["rouge_l_f1"]
    matches_raw = ordered_action_matching(events, gold) if format_valid else []
    matches: list[ChainMatch] = []
    action_mass = logic_mass = 0.0
    for pi, gi, action_sim in matches_raw:
        token_score = set_f1(events[pi].get("logic", ""), gold[gi].get("logic", ""))
        rouge_score = rouge_l_f1(events[pi].get("logic", ""), gold[gi].get("logic", ""))
        logic_sim = 0.5 * token_score + 0.5 * rouge_score
        action_mass += action_sim
        logic_mass += logic_sim
        matches.append(ChainMatch(pi, gi, action_sim, token_score, rouge_score, logic_sim))
    ap, ar, af = f1_from_totals(action_mass, len(events), len(gold)) if format_valid else (0.0, 0.0, 0.0)
    lp, lr, lf = f1_from_totals(logic_mass, len(events), len(gold)) if format_valid else (0.0, 0.0, 0.0)

    grounding = _ground_events(completion, events, nodes, sample, violations) if parsed.node is not None else []
    previous_date = ""
    seen: set[tuple[str, str]] = set()
    for index, (event, node) in enumerate(zip(events, nodes)):
        date = event.get("date", "")
        if previous_date and DATE_RE.fullmatch(date) and date < previous_date:
            violations.append(violation("chronology_violation", "Events are not chronological", _field_span(node, "date"), "chain", event_index=index))
        if DATE_RE.fullmatch(date):
            previous_date = date
        key = (date, normalize_text(event.get("action", "")))
        if key in seen:
            violations.append(violation("duplicate_event", "Duplicate normalized event", (node.start, node.end), "chain", event_index=index))
        seen.add(key)
        if index >= 5:
            violations.append(violation("excess_event", "Only the first five events are allowed", (node.start, node.end), "chain", event_index=index))

    if tokenizer is not None:
        mapper = TokenSpanMapper(tokenizer, completion)
        for item in violations:
            item.attribute(mapper)
    matched_pred = {item.predicted_index for item in matches}
    matched_gold = {item.gold_index for item in matches}
    spans = _event_spans(completion, events, nodes, tokenizer)
    total = 0.5 * af + 0.5 * lf
    return ChainRewardResult(
        reward=total,
        total_reward=total,
        action_precision=ap,
        action_recall=ar,
        action_f1=af,
        logic_precision=lp,
        logic_recall=lr,
        logic_f1=lf,
        format_valid=format_valid,
        parser_errors=errors,
        diagnostics={
            "invalid_json": "invalid_json" in errors,
            "missing_logic_chain": "missing_logic_chain" in errors,
            "missing_events": "missing_events" in errors,
            "events_not_list": "events_not_list" in errors,
            "invalid_event": "invalid_event" in errors,
            "missing_date": "missing_date" in errors,
            "missing_action": "missing_action" in errors,
            "missing_logic": "missing_logic" in errors,
            "too_many_events": "too_many_events" in errors,
            "empty_events": "empty_events" in errors,
        },
        action_similarity_matrix=action_matrix,
        logic_similarity_matrix=logic_matrix,
        action_alignment={"precision": ap, "recall": ar, "f1": af, "matched_sum": action_mass},
        logic_alignment={"precision": lp, "recall": lr, "f1": lf, "matched_sum": logic_mass},
        matches=matches,
        unmatched_gold=[index for index in range(len(gold)) if index not in matched_gold],
        unmatched_pred=[index for index in range(len(events)) if index not in matched_pred],
        predicted_events=events,
        event_spans=spans,
        grounding_status=grounding,
        violations=violations,
    )
