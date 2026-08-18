# -*- coding: utf-8 -*-
"""Runtime capture adapters that leave GR_REC_v1 rewards unchanged."""
from __future__ import annotations

import re
from types import SimpleNamespace

from grpo_sid import final_sid, parse_sid, q_reward

from .dsr_objectives import cot_score, exploration_score, prefix_support
from .dsr_parser import parse_interest_section


SA_TOKEN_RE = re.compile(r"<s_a_(\d+)>")
_CAPTURE = None


def _gold_set(values):
    return {parsed for value in values if (parsed := parse_sid(value)) is not None}


def _model_beam_call(model):
    for candidate in (model, getattr(model, "module", None)):
        stats = getattr(candidate, "_beam_stats", None)
        if stats and stats.get("last_call"):
            return stats["last_call"]
    return None


def audit_sa_tokenization(tokenizer) -> dict:
    vocab = tokenizer.get_vocab()
    entries = sorted(
        ((int(match.group(1)), token, token_id)
         for token, token_id in vocab.items()
         if (match := SA_TOKEN_RE.fullmatch(token))),
        key=lambda item: item[0],
    )
    if not entries:
        raise RuntimeError("tokenizer has no <s_a_x> semantic tokens")
    failures = []
    for value, token, token_id in entries:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [token_id]:
            failures.append({"value": value, "token": token, "vocab_id": token_id, "encoded": encoded})
    if failures:
        raise RuntimeError(f"s_A tokens are not all single-token: {failures[:8]}")
    return {
        "count": len(entries),
        "min_a": entries[0][0],
        "max_a": entries[-1][0],
        "all_single_token": True,
    }


def locate_final_sid_a_position(completion_ids, tokenizer):
    """Return (a_value, completion_position), or (None, -1) fail-closed."""
    text = tokenizer.decode(completion_ids, skip_special_tokens=False)
    sid = final_sid(text)
    if sid is None:
        return None, -1
    domain, a, b, c = sid
    pieces = [f"<|{domain}_begin|>", f"<s_a_{a}>", f"<s_b_{b}>", f"<s_c_{c}>"]
    encoded = [tokenizer.encode(piece, add_special_tokens=False) for piece in pieces]
    if any(len(value) != 1 for value in encoded):
        return None, -1
    pattern = [value[0] for value in encoded]
    for start in range(len(completion_ids) - 4, -1, -1):
        if list(completion_ids[start:start + 4]) == pattern:
            return a, start + 1
    return None, -1


class DsrCapture:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.records = []
        self.pending_beam_results = None
        self.tokenizer_audit = audit_sa_tokenization(tokenizer)

    def reset(self):
        self.records = []
        self.pending_beam_results = None


def get_capture(tokenizer=None):
    global _CAPTURE
    if _CAPTURE is None:
        if tokenizer is None:
            raise RuntimeError("DSR capture has not been initialized")
        _CAPTURE = DsrCapture(tokenizer)
    elif tokenizer is not None and _CAPTURE.tokenizer is not tokenizer:
        _CAPTURE = DsrCapture(tokenizer)
    return _CAPTURE


def reset_global_capture():
    global _CAPTURE
    _CAPTURE = None


class DsrBeam32Adapter:
    """Capture already-computed Beam SIDs while returning rewards unchanged."""

    def __init__(self, baseline_fn, model, capture):
        self.baseline_fn = baseline_fn
        self.model = model
        self.capture = capture

    def __call__(self, prompts, completions, completion_ids, gold_sets):
        rewards = self.baseline_fn(prompts, completions, completion_ids, gold_sets)
        call = _model_beam_call(self.model)
        if call is None:
            raise RuntimeError("DSR requires the baseline Beam last_call side channel")
        results = call.get("local_results") or []
        if len(results) != len(rewards):
            raise RuntimeError(f"Beam result count {len(results)} != reward count {len(rewards)}")
        if any(result.get("beam_sids") is None for result in results):
            raise RuntimeError("DSR Beam capture is missing parsed beam_sids")
        self.capture.pending_beam_results = results
        return rewards


def make_dsr_beam32_fn(baseline_factory, model, tokenizer, monitor_writer=None):
    capture = get_capture(tokenizer)
    # The baseline Beam implementation only retains already-parsed SIDs when
    # monitoring is enabled. This proxy enables that compact side channel even
    # if file monitoring is disabled; it does not perform I/O or generation.
    capture_proxy = SimpleNamespace(enabled=True)
    baseline_fn = baseline_factory(model, tokenizer, monitor_writer=capture_proxy)
    return DsrBeam32Adapter(baseline_fn, model, capture)


def make_dsr_think_reward_func(beam32_fn=None):
    def reward_func(prompts, completions, completion_ids, **kwargs):
        routes = kwargs.get("route")
        think_idx = [index for index, route in enumerate(routes or []) if route == "think"]
        if not think_idx:
            return [None] * len(prompts)
        if beam32_fn is None:
            raise RuntimeError("DSR Think reward requires Beam32")
        gold_sets = [_gold_set(values) for values in kwargs["all_gold_sids"]]
        sub_prompts = [prompts[index] for index in think_idx]
        sub_ids = [completion_ids[index] for index in think_idx]
        sub_golds = [gold_sets[index] for index in think_idx]
        primary = beam32_fn(sub_prompts, [completions[index] for index in think_idx], sub_ids, sub_golds)
        capture = get_capture()
        results = capture.pending_beam_results
        if results is None or len(results) != len(primary):
            raise RuntimeError("DSR Think reward did not receive aligned Beam results")
        for local_index, source_index in enumerate(think_idx):
            cot = capture.tokenizer.decode(sub_ids[local_index], skip_special_tokens=False)
            parsed = parse_interest_section(cot, sub_prompts[local_index])
            cot_stats = cot_score(parsed)
            beam_sids = results[local_index]["beam_sids"]
            prefix = prefix_support(beam_sids, sub_golds[local_index])
            target_domain = kwargs.get("target_domain", [None] * len(prompts))[source_index]
            explore = exploration_score(beam_sids, target_domain)
            capture.records.append({
                "route": "think",
                "group_id": kwargs["recommendation_group_id"][source_index],
                "primary_reward": float(primary[local_index]),
                "cot": cot,
                "parsed": parsed.to_dict(),
                **cot_stats,
                **prefix,
                **explore,
                "s_dead": float(cot_stats["s_cot"]) * float(explore["s_explore"]),
                "has_exact": bool(results[local_index].get("exact", 0)),
                "beam_invalid": int(results[local_index].get("invalid", 0)),
            })
        capture.pending_beam_results = None
        output = [None] * len(prompts)
        for index, reward in zip(think_idx, primary):
            output[index] = reward
        return output

    reward_func.__name__ = "think_reward"
    return reward_func


def make_dsr_nothink_reward_func(tokenizer=None):
    if tokenizer is None:
        raise RuntimeError("DSR NoThink reward requires tokenizer")
    capture = get_capture(tokenizer)

    def reward_func(prompts, completions, completion_ids, **kwargs):
        routes = kwargs.get("route")
        output = []
        for index, (ids, values, route) in enumerate(zip(completion_ids, kwargs["all_gold_sids"], routes)):
            if route != "no_think":
                output.append(None)
                continue
            golds = _gold_set(values)
            text = tokenizer.decode(ids, skip_special_tokens=False)
            sid = final_sid(text)
            reward = q_reward(sid, golds)
            predicted_a, position = locate_final_sid_a_position(ids, tokenizer)
            capture.records.append({
                "route": "no_think",
                "group_id": kwargs["recommendation_group_id"][index],
                "primary_reward": float(reward),
                "predicted_a": predicted_a,
                "sa_position": position,
                "parsed_sid": sid,
                "gold_as": sorted({gold[1] for gold in golds}),
            })
            output.append(reward)
        return output

    reward_func.__name__ = "nothink_reward"
    return reward_func
