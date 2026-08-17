# -*- coding: utf-8 -*-
"""CPU parity check for entropy-free GRPO policy loss."""
from types import MethodType, SimpleNamespace

import torch

from grpo_trl_trainer import RecGRPOTrainer


class TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(17, 8)
        self.head = torch.nn.Linear(8, 17, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


def make_trainer(force_compute_entropy=None):
    trainer = RecGRPOTrainer.__new__(RecGRPOTrainer)
    trainer.temperature = 0.9
    trainer.model_kwarg_keys = set()
    trainer.epsilon_low = 0.2
    trainer.epsilon_high = 0.2
    trainer.args = SimpleNamespace(delta=None, report_to=[])
    trainer.loss_type = "grpo"
    trainer.current_gradient_accumulation_steps = 2
    trainer._smoke_policy_epoch = {}
    trainer._smoke_rollout_id = 1
    trainer._smoke_log = [{"route": "think"}]
    trainer._detailed_monitor = False
    trainer.state = SimpleNamespace(global_step=0)
    trainer.requested_compute_entropy = []

    upstream = RecGRPOTrainer._get_per_token_logps_and_entropies

    def get_logps(self, model, input_ids, attention_mask, logits_to_keep, **kwargs):
        requested = kwargs.get("compute_entropy", False)
        self.requested_compute_entropy.append(requested)
        compute_entropy = requested if force_compute_entropy is None else force_compute_entropy
        return upstream(
            self, model, input_ids, attention_mask, logits_to_keep,
            compute_entropy=compute_entropy,
        )

    trainer._get_per_token_logps_and_entropies = MethodType(get_logps, trainer)
    return trainer


torch.manual_seed(20260817)
model_old = TinyCausalLM()
model_new = TinyCausalLM()
model_new.load_state_dict(model_old.state_dict())

inputs = {
    "prompt_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
    "prompt_mask": torch.ones(2, 3, dtype=torch.long),
    "completion_ids": torch.tensor([[7, 8, 9, 10], [11, 12, 13, 14]]),
    "completion_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
    "advantages": torch.tensor([0.75, -0.5]),
    "old_per_token_logps": torch.tensor([
        [-2.0, -2.1, -2.2, -2.3],
        [-2.4, -2.5, -2.6, -2.7],
    ]),
    "route_id": torch.zeros(2, dtype=torch.long),
}

trainer_old = make_trainer(force_compute_entropy=True)
trainer_new = make_trainer()
joined = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
old_logps, old_entropy = trainer_old._get_per_token_logps_and_entropies(
    model_old, joined, mask, inputs["completion_ids"].size(1),
)
new_logps, new_entropy = trainer_new._get_per_token_logps_and_entropies(
    model_new, joined, mask, inputs["completion_ids"].size(1),
)
assert torch.equal(old_logps, new_logps), "per_token_logps changed"
assert old_entropy is not None and new_entropy is None

old_loss = trainer_old._compute_loss(model_old, inputs)
new_loss = trainer_new._compute_loss(model_new, inputs)
assert trainer_new.requested_compute_entropy[-1] is False, "_compute_loss requested entropy"
old_loss.backward()
new_loss.backward()

assert torch.equal(old_loss, new_loss), f"loss changed: {old_loss} vs {new_loss}"
for (old_name, old_param), (new_name, new_param) in zip(
    model_old.named_parameters(), model_new.named_parameters()
):
    assert old_name == new_name
    assert torch.equal(old_param.grad, new_param.grad), f"gradient changed: {old_name}"

core_keys = trainer_new._smoke_log[-1]
for key in ("ratio_mean_ep1", "clip_fraction_ep1", "approx_kl_ep1", "policy_fwb_sec_ep1"):
    assert key in core_keys, f"missing core metric: {key}"
for key in ("ratio_p95_ep1", "ratio_p99_ep1"):
    assert key not in core_keys, f"detailed metric present while disabled: {key}"

print("[PASS] per_token_logps exact parity")
print("[PASS] loss exact parity")
print("[PASS] gradients exact parity")
print("[PASS] detailed monitoring disabled; core policy metrics retained")
