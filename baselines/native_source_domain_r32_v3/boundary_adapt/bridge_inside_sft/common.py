"""Shared contracts for Bridge-Inside Transition SFT V1."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any

import torch
import torch.nn.functional as F

RUNTIME = Path("/data/GRPO")
sys.path[:0] = [str(RUNTIME), str(RUNTIME / "scripts")]

BASE = Path("/data/models/onereason-8b-pretrain-competition")
FRESH_BATA = Path(
    "/data/outputs/baselines/native_source_domain_r32_v3/"
    "BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106"
)
SOURCE = Path("/data/lf_data_versions/alltrain/bata_baseline_v1/onereason_bata_baseline.jsonl")
SOURCE_SHA256 = "f84f288b8a9685b4c9cb769c937a5250407723dfab11bdc548486ac7751ec8ca"
ADAPTER_SHA256 = "4c077d9b0865b883bf42c86a0ffd872785b15558dc46d54482e99612e1be53c3"
SPLIT_SALT = "bridge_inside_transition_v1|"
SEED = 20260824
IGNORE_INDEX = -100
DOMAIN = {
    "video": "<|video_begin|>",
    "prod": "<|prod_begin|>",
    "ad": "<|ad_begin|>",
    "living": "<|living_begin|>",
}
DOMAIN_ORDER = tuple(DOMAIN)
ABC_RE = re.compile(r"^(<s_a_\d+>)(<s_b_\d+>)(<s_c_\d+>)$")
STANDARD_USER_ENDINGS = (
    "请根据以上信息，给出该用户在直播、电商、视频、广告场景中的目标内容。/think",
    "请输出该用户在不同场景下对应的目标内容。/think",
    "请基于这些线索总结该用户在各场景中的目标内容。/think",
)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def token_ids_sha(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("ascii")).hexdigest()


def source_group_id(row: dict[str, Any]) -> str:
    return str(json.loads(row.get("aux_metadata_json") or "{}")["recommendation_group_id"])


def source_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row.get("aux_metadata_json") or "{}")


def infer_domain(metadata: dict[str, Any]) -> str:
    current = str(metadata["recommendation_current_gold_sid"])
    matches = [domain for domain, token in DOMAIN.items() if current.startswith(token)]
    if len(matches) != 1:
        raise ValueError(f"cannot infer unique domain from {current!r}")
    return matches[0]


def abc_golds(metadata: dict[str, Any], domain: str) -> list[str]:
    prefix = DOMAIN[domain]
    values = []
    for raw in metadata["recommendation_all_gold_sids"]:
        raw = str(raw)
        if not raw.startswith(prefix):
            raise ValueError("gold/domain mismatch")
        abc = raw[len(prefix):]
        if not ABC_RE.fullmatch(abc):
            raise ValueError(f"malformed ABC gold {abc!r}")
        values.append(abc)
    if len(set(values)) != len(values) or not values:
        raise ValueError("empty or duplicate gold paths")
    return values


def extract_response(response: str, domain: str) -> tuple[str, str, str]:
    """Return pre-close CoT (including <think>), exact bridge, and one ABC path."""
    if response.count("</think>") != 1:
        raise ValueError("response must contain exactly one </think>")
    left = response.index("</think>")
    domain_pos = response.index(DOMAIN[domain], left + len("</think>"))
    cot_before_close = response[:left]
    bridge = response[left + len("</think>"):domain_pos]
    abc = response[domain_pos + len(DOMAIN[domain]):]
    if not cot_before_close.startswith("<think>") or not bridge or not ABC_RE.fullmatch(abc):
        raise ValueError("response transition contract failed")
    return cot_before_close, bridge, abc


def prompt_family(user: str) -> str:
    last = user.splitlines()[-1] if user.splitlines() else ""
    for index, ending in enumerate(STANDARD_USER_ENDINGS):
        if last == ending:
            return f"standard_{index}"
    return "other"


def render_system_prompt_ids(tokenizer, user: str, system: str) -> list[int]:
    from llamafactory.data.template import TEMPLATES

    ids, _ = TEMPLATES["qwen3_nothink"].encode_oneturn(
        tokenizer,
        [{"role": "user", "content": user}, {"role": "assistant", "content": ""}],
        system=system,
    )
    return list(map(int, ids))


def construct_token_row(tokenizer, row: dict[str, Any]) -> dict[str, Any]:
    metadata = source_metadata(row)
    group = source_group_id(row)
    domain = infer_domain(metadata)
    user = str(row.get("instruction", "")) + str(row.get("input", ""))
    system = str(row.get("system", ""))
    if not system:
        raise ValueError("system is empty")
    cot_text, bridge, output_abc = extract_response(str(row["output"]), domain)
    golds_abc = abc_golds(metadata, domain)
    if output_abc not in golds_abc:
        raise ValueError("output ABC is absent from all-gold paths")
    prompt_ids = render_system_prompt_ids(tokenizer, user, system)
    original_cot_ids = tokenizer.encode(cot_text + "</think>", add_special_tokens=False)
    close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    bridge_ids = tokenizer.encode(bridge, add_special_tokens=False)
    domain_ids = tokenizer.encode(DOMAIN[domain], add_special_tokens=False)
    abc_ids = tokenizer.encode(output_abc, add_special_tokens=False)
    if len(close_ids) != 1 or original_cot_ids[-1:] != close_ids:
        raise ValueError("atomic close boundary failed")
    if len(domain_ids) != 1 or len(abc_ids) != 3:
        raise ValueError("atomic domain/ABC boundary failed")
    input_ids = prompt_ids + original_cot_ids[:-1] + bridge_ids + close_ids + domain_ids + abc_ids
    label_start = len(prompt_ids) + len(original_cot_ids) - 1
    labels = [IGNORE_INDEX] * len(input_ids)
    labels[label_start:label_start + len(bridge_ids) + 1] = bridge_ids + close_ids
    labeled = [value for value in labels if value != IGNORE_INDEX]
    if labeled != bridge_ids + close_ids:
        raise AssertionError("labels are not exact bridge plus close")
    return {
        "group_id": group,
        "target_domain": domain,
        "prompt_family": prompt_family(user),
        "gold_count": len(golds_abc),
        "gold_sids": [DOMAIN[domain] + value for value in golds_abc],
        "output_abc": output_abc,
        "system": system,
        "user": user,
        "prompt_token_ids": prompt_ids,
        "teacher_cot_token_ids": list(map(int, original_cot_ids)),
        "bridge": bridge,
        "bridge_token_ids": list(map(int, bridge_ids)),
        "domain_token_ids": list(map(int, domain_ids)),
        "input_ids": list(map(int, input_ids)),
        "labels": list(map(int, labels)),
        "labeled_token_count": len(labeled),
        "input_ids_sha256": token_ids_sha(input_ids),
        "teacher_cot_ids_sha256": token_ids_sha(original_cot_ids),
        "source_row_sha256": stable_hash(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
    }


def transition_group_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Equal group weight: mean labeled-token CE inside each row, then row mean."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logit/label shape mismatch")
    shifted_logits = logits[:, :-1].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    token_loss = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view_as(shifted_labels)
    mask = shifted_labels.ne(IGNORE_INDEX)
    counts = mask.sum(dim=1)
    if torch.any(counts <= 0):
        raise ValueError("row has no transition labels")
    return ((token_loss * mask).sum(dim=1) / counts).mean()


def close_position(ids: list[int], close_id: int) -> int | None:
    try:
        return ids.index(close_id)
    except ValueError:
        return None


def sequence_occurrences(sequence: list[int], needle: list[int]) -> list[int]:
    if not needle:
        return []
    return [index for index in range(len(sequence) - len(needle) + 1) if sequence[index:index + len(needle)] == needle]


def emission_metrics(ids: list[int], close_id: int, exact_bridge: list[int], known_bridges: dict[str, list[int]], domain: str) -> dict[str, Any]:
    close = close_position(ids, close_id)
    exact_positions = sequence_occurrences(ids, exact_bridge)
    all_hits = [(name, position) for name, bridge in known_bridges.items() for position in sequence_occurrences(ids, bridge)]
    adjacent_exact = close is not None and any(position + len(exact_bridge) == close for position in exact_positions)
    adjacent_known = close is not None and any(position + len(known_bridges[name]) == close for name, position in all_hits)
    wrong = close is not None and any(name != domain and position + len(known_bridges[name]) == close for name, position in all_hits)
    return {
        "closed": close is not None,
        "close_position": close,
        "exact_bridge_positions": exact_positions,
        "known_bridge_hits": [{"domain": name, "position": position} for name, position in all_hits],
        "exact_bridge_before_close": adjacent_exact,
        "any_known_bridge_before_close": adjacent_known,
        "wrong_domain_bridge": wrong,
        "bridge_not_adjacent_to_close": bool(all_hits) and not adjacent_known,
        "multiple_bridge": len(all_hits) > 1,
    }


def percentile(values: list[int], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    fraction = position - low
    return values[low] * (1 - fraction) + values[high] * fraction


def summarize_lengths(values: list[int]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, .50),
        "p90": percentile(values, .90),
        "p99": percentile(values, .99),
    }


def count_distribution(values) -> dict[str, int]:
    return {str(key): count for key, count in sorted(Counter(values).items())}
