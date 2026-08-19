"""Shared deterministic parsing and similarity helpers for GR_USER_v1 rewards."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


SID_PATTERN = r"<\|(?:video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>"
SID_RE = re.compile(SID_PATTERN)
SID_FULL_RE = re.compile(rf"^{SID_PATTERN}$")
SID_PART_RE = re.compile(
    r"^<\|(?P<domain>video|prod|ad|living)_begin\|>"
    r"<s_a_(?P<a>\d+)><s_b_(?P<b>\d+)><s_c_(?P<c>\d+)>$"
)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class JsonNode:
    value: Any
    start: int
    end: int
    children: list["JsonNode"] = field(default_factory=list)
    members: dict[str, list["JsonNode"]] = field(default_factory=dict)

    def member(self, key: str) -> "JsonNode | None":
        values = self.members.get(key, [])
        return values[-1] if values else None


@dataclass
class ParsedJson:
    value: Any | None
    node: JsonNode | None
    candidate_start: int
    candidate_end: int
    strict: bool
    errors: list[str]


class _NodeParser:
    def __init__(self, text: str, base: int = 0):
        self.text = text
        self.base = base
        self.decoder = json.JSONDecoder()

    def ws(self, i: int) -> int:
        while i < len(self.text) and self.text[i].isspace():
            i += 1
        return i

    def parse(self, i: int = 0) -> tuple[JsonNode, int]:
        i = self.ws(i)
        if i >= len(self.text):
            raise ValueError("empty JSON")
        if self.text[i] == "[":
            return self.array(i)
        if self.text[i] == "{":
            return self.object(i)
        value, end = self.decoder.raw_decode(self.text, i)
        return JsonNode(value, self.base + i, self.base + end), end

    def array(self, i: int) -> tuple[JsonNode, int]:
        start = i
        i += 1
        children: list[JsonNode] = []
        i = self.ws(i)
        if i < len(self.text) and self.text[i] == "]":
            return JsonNode([], self.base + start, self.base + i + 1), i + 1
        while True:
            child, i = self.parse(i)
            children.append(child)
            i = self.ws(i)
            if i >= len(self.text):
                raise ValueError("unterminated array")
            if self.text[i] == "]":
                i += 1
                return JsonNode([c.value for c in children], self.base + start, self.base + i, children), i
            if self.text[i] != ",":
                raise ValueError("array expects comma")
            i += 1

    def object(self, i: int) -> tuple[JsonNode, int]:
        start = i
        i += 1
        members: dict[str, list[JsonNode]] = {}
        values: dict[str, Any] = {}
        i = self.ws(i)
        if i < len(self.text) and self.text[i] == "}":
            return JsonNode({}, self.base + start, self.base + i + 1), i + 1
        while True:
            i = self.ws(i)
            key, key_end = self.decoder.raw_decode(self.text, i)
            if not isinstance(key, str):
                raise ValueError("object key is not a string")
            i = self.ws(key_end)
            if i >= len(self.text) or self.text[i] != ":":
                raise ValueError("object expects colon")
            child, i = self.parse(i + 1)
            members.setdefault(key, []).append(child)
            values[key] = child.value
            i = self.ws(i)
            if i >= len(self.text):
                raise ValueError("unterminated object")
            if self.text[i] == "}":
                i += 1
                return JsonNode(values, self.base + start, self.base + i, members=members), i
            if self.text[i] != ",":
                raise ValueError("object expects comma")
            i += 1


def parse_final_json(text: str) -> ParsedJson:
    """Parse one final JSON value while retaining exact source spans."""
    errors: list[str] = []
    marker = text.rfind("</think>")
    start = marker + len("</think>") if marker >= 0 else 0
    leading = text[start:]
    first = next((i for i, ch in enumerate(leading) if not ch.isspace()), len(leading))
    start += first
    if start >= len(text):
        return ParsedJson(None, None, start, start, False, ["missing_json"])
    if text[start] not in "[{":
        candidates = [p for p in (text.find("[", start), text.find("{", start)) if p >= 0]
        if not candidates:
            return ParsedJson(None, None, start, len(text), False, ["missing_json"])
        errors.append("extra_text_before_json")
        start = min(candidates)
    try:
        parser = _NodeParser(text[start:], start)
        node, relative_end = parser.parse(0)
        end = start + relative_end
    except (ValueError, json.JSONDecodeError, TypeError):
        return ParsedJson(None, None, start, len(text), False, errors + ["malformed_json"])
    if text[end:].strip():
        errors.append("extra_text_after_json")
    return ParsedJson(node.value, node, start, end, not errors, errors)


def node_text_span(node: JsonNode) -> tuple[int, int]:
    """Return the content span for ordinary JSON strings, otherwise the node span."""
    if isinstance(node.value, str) and node.end - node.start >= 2:
        return node.start + 1, node.end - 1
    return node.start, node.end


def decoded_string_span(source: str, node: JsonNode, decoded_start: int = 0, decoded_end: int | None = None) -> tuple[int, int]:
    """Map a decoded JSON-string slice back to its exact raw character span."""
    if not isinstance(node.value, str):
        return node.start, node.end
    decoded_end = len(node.value) if decoded_end is None else decoded_end
    raw = source[node.start + 1 : node.end - 1]
    mapping: list[tuple[int, int]] = []
    i = 0
    while i < len(raw):
        absolute = node.start + 1 + i
        if raw[i] != "\\":
            mapping.append((absolute, absolute + 1))
            i += 1
            continue
        if i + 1 >= len(raw):
            break
        if raw[i + 1] != "u":
            mapping.append((absolute, absolute + 2))
            i += 2
            continue
        end = i + 6
        codepoint = int(raw[i + 2 : end], 16)
        if 0xD800 <= codepoint <= 0xDBFF and raw[end : end + 2] == "\\u" and end + 6 <= len(raw):
            mapping.append((absolute, node.start + 1 + end + 6))
            i = end + 6
        else:
            mapping.append((absolute, node.start + 1 + end))
            i = end
    if decoded_start == decoded_end:
        boundary = mapping[decoded_start][0] if decoded_start < len(mapping) else node.end - 1
        return boundary, boundary
    if decoded_start < 0 or decoded_end > len(mapping):
        return node_text_span(node)
    return mapping[decoded_start][0], mapping[decoded_end - 1][1]


def sid_components(sid: str) -> tuple[str, str, str, str] | None:
    match = SID_PART_RE.fullmatch(sid)
    return match.groups() if match else None


def prefix_invalid_component(sid: str, history_sids: set[str]) -> tuple[str, list[str]]:
    components = sid_components(sid)
    if components is None:
        return "format", ["format"]
    history_components = [sid_components(item) for item in history_sids]
    labels = ("domain", "a", "b", "c")
    for index in range(4):
        prefix = components[: index + 1]
        if not any(item is not None and item[: index + 1] == prefix for item in history_components):
            return labels[index], list(labels[index:])
    return "none", []


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower().strip()


def approximate_tokens(text: str) -> list[str]:
    """Deterministic approximation: SID atomic, ASCII words, and individual Han chars."""
    normalized = normalize_text(text)
    pattern = re.compile(rf"{SID_PATTERN}|[a-z0-9_]+|[\u3400-\u9fff]")
    return pattern.findall(normalized)


def set_f1(left: str, right: str) -> float:
    a, b = set(approximate_tokens(left)), set(approximate_tokens(right))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    common = len(a & b)
    precision = common / len(a)
    recall = common / len(b)
    return 2 * precision * recall / (precision + recall) if common else 0.0


def rouge_l_f1(left: str, right: str) -> float:
    a, b = approximate_tokens(left), approximate_tokens(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for j, other in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if token == other else max(previous[j], current[-1]))
        previous = current
    lcs = previous[-1]
    return 2 * lcs / (len(a) + len(b))


def f1_from_totals(matched: float, predicted: int, gold: int) -> tuple[float, float, float]:
    precision = matched / predicted if predicted else (1.0 if gold == 0 else 0.0)
    recall = matched / gold if gold else (1.0 if predicted == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1
