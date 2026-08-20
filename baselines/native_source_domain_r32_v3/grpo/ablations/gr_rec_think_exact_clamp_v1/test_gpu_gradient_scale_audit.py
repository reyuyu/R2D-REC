"""CPU structure tests for the zero-update GPU gradient audit harness."""

from __future__ import annotations

import ast
import inspect
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

GRPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GRPO_ROOT / "scripts"))

import audit_gpu_gradient_scale as audit


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    mapping = {
        "</think>": 2,
        "商品": 3,
        "视频": 4,
        "广告": 5,
        "主播": 6,
        "该用户最近喜欢的视频有: ": [50, 51, 52, 53, 54, 4, 55],
        "该用户最近点击了商品: ": [50, 51, 52, 56, 57, 3, 55],
        "该用户最近感兴趣的广告有: ": [50, 51, 52, 58, 5, 55],
        "该用户最近首次打赏了主播: ": [50, 51, 52, 59, 60, 61, 57, 6, 55],
        "<|prod_begin|>": 10,
        "<|video_begin|>": 20,
        "<s_a_1>": 11,
        "<s_a_9>": 19,
        "<s_b_11>": 12,
        "<s_b_9>": 29,
        "<s_c_111>": 13,
        "<s_c_9>": 39,
    }

    def encode(self, text, add_special_tokens=False):
        value = self.mapping[text]
        return value if isinstance(value, list) else [value]


class TinyAdapterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_logits = torch.nn.Parameter(torch.zeros((8, 7, 64)))
        self.frozen_base = torch.nn.Parameter(torch.ones(3), requires_grad=False)
        self.input_pointers = []

    def forward(self, input_ids, attention_mask, use_cache=False, logits_to_keep=0):
        self.input_pointers.append(input_ids.data_ptr())
        batch = input_ids.size(0)
        logits = self.lora_logits[:batch]
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return SimpleNamespace(logits=logits)


tokenizer = TinyTokenizer()
completion_ids_list = (
    (2, 3, 10, 19, 29, 39),
    (2, 3, 10, 11, 29, 39),
    (2, 3, 10, 11, 12, 39),
    (2, 3, 10, 11, 12, 13),
) * 2
completion_ids = torch.tensor(completion_ids_list, dtype=torch.long)
prompt_ids = (1,)
input_ids = torch.cat(
    [torch.tensor([prompt_ids] * 8, dtype=torch.long), completion_ids], dim=1
)
batch = audit.RolloutBatch(
    group_id="same-rollout",
    prompt_ids=prompt_ids,
    completion_ids_list=completion_ids_list,
    input_ids=input_ids,
    attention_mask=torch.ones_like(input_ids),
    completion_ids=completion_ids,
    completion_mask=torch.ones_like(completion_ids),
    rewards=(0.0, 0.5, 2.0, 8.0) * 2,
    predicted_sids=(
        ("prod", 9, 9, 9),
        ("prod", 1, 9, 9),
        ("prod", 1, 11, 9),
        ("prod", 1, 11, 111),
    ) * 2,
    gold_sids=(("prod", 1, 11, 111),),
    target_domain="prod",
)


# Pure objective construction matches the formal legacy and hierarchy contracts.
legacy = audit.legacy_advantages(batch.rewards, "cpu")
assert legacy.shape == (8,) and abs(float(legacy.mean())) < 1e-6
hierarchical, credits, alignment = audit.hierarchical_token_advantages(batch, tokenizer)
assert hierarchical.shape == completion_ids.shape
assert alignment["domain_text_alignment_valid"] is True
assert alignment["direct_sid_fallback_count"] == 8
assert credits[0] == (0.0, -0.046875, 0.0, 0.0)
assert credits[1] == (0.0, 0.015625, -0.125, 0.0)
assert credits[2] == (0.0, 0.015625, 0.0625, -0.375)
assert credits[3] == (0.0, 0.015625, 0.0625, 0.375)

# The audit consumes Domain/A/B/C credit and reports the Domain stage.
domain_completion_ids_list = tuple(
    (2, 3, 10, 19, 29, 39) if index in {1, 5}
    else (2, 4, 20, 19, 29, 39)
    for index in range(8)
)
domain_completion_ids = torch.tensor(domain_completion_ids_list, dtype=torch.long)
domain_input_ids = torch.cat(
    [torch.tensor([prompt_ids] * 8, dtype=torch.long), domain_completion_ids], dim=1
)
domain_batch = replace(
    batch,
    group_id="domain-signal",
    completion_ids_list=domain_completion_ids_list,
    input_ids=domain_input_ids,
    attention_mask=torch.ones_like(domain_input_ids),
    completion_ids=domain_completion_ids,
    completion_mask=torch.ones_like(domain_completion_ids),
    rewards=(-0.25, 0.0, -0.25, -0.25, -0.25, 0.0, -0.25, -0.25),
    predicted_sids=tuple(
        ("prod", 9, 9, 9) if index in {1, 5} else ("video", 9, 9, 9)
        for index in range(8)
    ),
)
domain_token_advantages, domain_credits, domain_alignment = audit.hierarchical_token_advantages(
    domain_batch, tokenizer
)
assert domain_alignment["domain_text_alignment_valid"] is True
assert [row[0] for row in domain_credits] == [
    -0.0078125, 0.0234375, -0.0078125, -0.0078125,
    -0.0078125, 0.0234375, -0.0078125, -0.0078125,
]
torch.testing.assert_close(
    domain_token_advantages[:, 2],
    torch.tensor([row[0] for row in domain_credits]),
)
assert torch.count_nonzero(domain_token_advantages[:, :2]) == 0
assert torch.count_nonzero(domain_token_advantages[:, 3:]) == 0

# Noun mismatch/missing does not matter: valid final SID receives candidate fallback.
mismatch_ids = list(domain_completion_ids_list)
mismatch_ids[1] = (2, 4, 10, 19, 29, 39)
mismatch_tensor = torch.tensor(mismatch_ids, dtype=torch.long)
mismatch_batch = replace(
    domain_batch,
    completion_ids_list=tuple(mismatch_ids),
    completion_ids=mismatch_tensor,
    input_ids=torch.cat([torch.tensor([prompt_ids] * 8), mismatch_tensor], dim=1),
    attention_mask=torch.ones((8, 7), dtype=torch.long),
)
mismatch_advantages, mismatch_credits, mismatch_alignment = audit.hierarchical_token_advantages(
    mismatch_batch, tokenizer
)
assert mismatch_alignment["domain_text_alignment_valid"] is True
assert mismatch_alignment["direct_sid_fallback_count"] == 8
assert mismatch_credits == domain_credits

missing_ids = list(domain_completion_ids_list)
missing_ids[1] = (2, 0, 10, 19, 29, 39)
missing_tensor = torch.tensor(missing_ids, dtype=torch.long)
missing_batch = replace(
    domain_batch,
    completion_ids_list=tuple(missing_ids),
    completion_ids=missing_tensor,
    input_ids=torch.cat([torch.tensor([prompt_ids] * 8), missing_tensor], dim=1),
    attention_mask=torch.ones((8, 7), dtype=torch.long),
)
missing_advantages, missing_credits, missing_alignment = audit.hierarchical_token_advantages(
    missing_batch, tokenizer
)
assert missing_alignment["domain_text_alignment_valid"] is True
assert missing_alignment["direct_sid_fallback_count"] == 8
assert missing_credits == domain_credits


# One immutable batch and one old-logp tensor feed both isolated backwards.
model = TinyAdapterModel()
parameters = audit.trainable_parameters(model)
assert [name for name, _parameter in parameters] == ["lora_logits"]
checksum_before = audit.trainable_parameter_checksum(parameters)
with torch.no_grad():
    model.frozen_base.add_(1)
assert audit.trainable_parameter_checksum(parameters) == checksum_before
result = audit.audit_rollout(model, tokenizer, parameters, batch)
assert audit.trainable_parameter_checksum(parameters) == checksum_before
assert result["rollout_fingerprint"] == batch.fingerprint
assert result["legacy_grad_norm"] > 0 and result["hier_grad_norm"] > 0
assert result["hier_over_legacy"] > 0
assert result["legacy_hier_cosine"] is not None
assert len(model.input_pointers) == 3  # old logp, legacy, hierarchy
assert len(set(model.input_pointers)) == 1
assert model.lora_logits.grad is None and model.frozen_base.grad is None
domain_result = audit.audit_rollout(model, tokenizer, parameters, domain_batch)
assert domain_result["stage_active"] == {
    "domain": True, "a": False, "b": False, "c": False,
}
assert domain_result["credited_token_count"] == {
    "domain": 8, "a": 0, "b": 0, "c": 0,
}
assert domain_result["legacy_text_domain_grad_norm"] > 0
assert domain_result["legacy_text_domain_hier_cosine"] > 0.999999
assert all(
    abs(item["text_domain_old_probability"] - 1 / 64) < 1e-8
    and abs(item["sid_domain_old_probability"] - 1 / 64) < 1e-8
    for item in domain_result["candidate_domain_diagnostics"]
)
matched = audit.legacy_text_domain_advantages(
    domain_batch, audit.legacy_advantages(domain_batch.rewards, "cpu"), domain_alignment
)
assert torch.count_nonzero(matched[:, 2]) == 8
assert torch.count_nonzero(matched[:, :2]) == 0
assert torch.count_nonzero(matched[:, 3:]) == 0


# Bridge measurement is gated by a real all-zero reward vector and uses no
# zero-gradient hierarchical group as its reference in final summarization.
all_zero_completion_ids_list = (completion_ids_list[0],) * 8
all_zero_completion_ids = torch.tensor(
    all_zero_completion_ids_list, dtype=torch.long
)
all_zero_input_ids = torch.cat(
    [torch.tensor([prompt_ids] * 8, dtype=torch.long), all_zero_completion_ids],
    dim=1,
)
all_zero = replace(
    batch,
    group_id="real-all-zero",
    completion_ids_list=all_zero_completion_ids_list,
    input_ids=all_zero_input_ids,
    attention_mask=torch.ones_like(all_zero_input_ids),
    completion_ids=all_zero_completion_ids,
    completion_mask=torch.ones_like(all_zero_completion_ids),
    rewards=(0.0,) * 8,
    predicted_sids=(("prod", 9, 9, 9),) * 8,
)
bridge_model = TinyAdapterModel()
bridge_parameters = audit.trainable_parameters(bridge_model)
bridge_result = audit.audit_rollout(
    bridge_model, tokenizer, bridge_parameters, all_zero
)
assert bridge_result["dead_zero_bridge_active"] is True
assert bridge_result["hier_grad_norm"] == 0
assert bridge_result["stage_active"] == {
    "domain": False, "a": False, "b": False, "c": False,
}
assert bridge_result["bridge_raw_grad_norm"] > 0
assert bridge_result["bridge_weighted_grad_norm"] == (
    audit.BRIDGE_LAMBDA * bridge_result["bridge_raw_grad_norm"]
)
try:
    audit.dead_zero_bridge_loss(model, batch, tokenizer)
except ValueError:
    pass
else:
    raise AssertionError("non-all-zero rollout must not receive bridge gradient")


summary_records = [
    {**result, "hier_grad_norm": 10.0, "bridge_weighted_grad_norm": None},
    {**result, "hier_grad_norm": 20.0, "bridge_weighted_grad_norm": None},
    {**bridge_result, "hier_grad_norm": 0.0, "bridge_weighted_grad_norm": 1.5},
]
summary = audit.summarize(summary_records)
assert summary["median_active_hier_grad_norm"] == 15.0
assert summary_records[2]["bridge_over_hier"] == 0.1
assert "REVIEW_BRIDGE_LAMBDA" not in summary["flags"]
summary_records[2]["bridge_weighted_grad_norm"] = 4.0
assert "REVIEW_BRIDGE_LAMBDA" in audit.summarize(summary_records)["flags"]


assert audit.reward_pattern([0] * 8) == "ALL_ZERO"
assert audit.reward_pattern([-0.25] * 8) == "ALL_WRONG_DOMAIN_ZERO_SIGNAL"
assert audit.reward_pattern([0, 0.5] * 4) == "A_ONLY"
assert audit.reward_pattern([0.5, 2] * 4) == "AB_SIGNAL"
assert audit.reward_pattern([2, 8] * 4) == "EXACT_SIGNAL"
assert audit.reward_pattern([-1, -0.25] * 4) == "MIXED"


# Paired comparison accepts only exact group/fingerprint/reward matches and
# preserves old SID-Domain values alongside the new Text-Domain diagnostics.
reference_groups = []
current_groups = []
parity = []
for index in range(8):
    old = {
        **domain_result,
        "group_id": f"group-{index}",
        "rollout_fingerprint": f"fingerprint-{index}",
        "hier_grad_norm": 0.5 + index,
    }
    new = {
        **domain_result,
        "group_id": f"group-{index}",
        "rollout_fingerprint": f"fingerprint-{index}",
        "hier_grad_norm": 1.0 + index,
    }
    reference_groups.append(old)
    current_groups.append(new)
    fake_batch = replace(
        domain_batch,
        group_id=old["group_id"],
        rewards=tuple(old["rewards"]),
        completion_ids_list=((index,),) * 8,
    )
    old["rollout_fingerprint"] = fake_batch.fingerprint
    new["rollout_fingerprint"] = fake_batch.fingerprint
    parity.append(audit.pairing_status(index, fake_batch, old))
assert all(item["valid"] for item in parity)
bad_reference = {**reference_groups[0], "rewards": [99.0] * 8}
assert audit.pairing_status(0, fake_batch, bad_reference)["valid"] is False
comparison = audit.build_paired_comparison(
    {"groups": reference_groups},
    {
        "groups": current_groups,
        "trainable_parameter_checksum_before": "same",
        "trainable_parameter_checksum_after": "same",
        "parameter_change": False,
    },
    parity,
)
assert comparison["paired_audit_valid"] is True
assert comparison["summary"]["fingerprint_parity_count"] == 8
assert comparison["summary"]["group_6_7_abc_parity"] is True


# Structural safety: the harness contains no call whose attribute is `step`.
tree = ast.parse(inspect.getsource(audit))
step_calls = [
    node for node in ast.walk(tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and node.func.attr == "step"
]
assert step_calls == []

try:
    audit.parse_args([])
except SystemExit:
    pass
else:
    raise AssertionError("GPU audit must require an explicit execution flag")

print("GPU GRADIENT AUDIT HARNESS CPU TESTS PASSED")
