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
        "<|prod_begin|>": 10,
        "<s_a_1>": 11,
        "<s_a_9>": 19,
        "<s_b_11>": 12,
        "<s_b_9>": 29,
        "<s_c_111>": 13,
        "<s_c_9>": 39,
    }

    def encode(self, text, add_special_tokens=False):
        return [self.mapping[text]]


class TinyAdapterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_logits = torch.nn.Parameter(torch.zeros((8, 5, 64)))
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
    (10, 19, 29, 39),
    (10, 11, 29, 39),
    (10, 11, 12, 39),
    (10, 11, 12, 13),
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
hierarchical, credits = audit.hierarchical_token_advantages(batch, tokenizer)
assert hierarchical.shape == completion_ids.shape
assert credits[0] == (-0.046875, 0.0, 0.0)
assert credits[1] == (0.015625, -0.125, 0.0)
assert credits[2] == (0.015625, 0.0625, -0.375)
assert credits[3] == (0.015625, 0.0625, 0.375)


# One immutable batch and one old-logp tensor feed both isolated backwards.
model = TinyAdapterModel()
parameters = audit.trainable_parameters(model)
assert [name for name, _parameter in parameters] == ["lora_logits"]
result = audit.audit_rollout(model, tokenizer, parameters, batch)
assert result["rollout_fingerprint"] == batch.fingerprint
assert result["legacy_grad_norm"] > 0 and result["hier_grad_norm"] > 0
assert result["hier_over_legacy"] > 0
assert result["legacy_hier_cosine"] is not None
assert len(model.input_pointers) == 3  # old logp, legacy, hierarchy
assert len(set(model.input_pointers)) == 1
assert model.lora_logits.grad is None and model.frozen_base.grad is None


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
assert audit.reward_pattern([0, 0.5] * 4) == "A_ONLY"
assert audit.reward_pattern([0.5, 2] * 4) == "AB_SIGNAL"
assert audit.reward_pattern([2, 8] * 4) == "EXACT_SIGNAL"
assert audit.reward_pattern([-1, -0.25] * 4) == "MIXED"


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
