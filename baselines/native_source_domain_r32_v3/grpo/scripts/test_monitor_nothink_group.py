"""CPU-only parity test for full G=8 NoThink monitor traces."""
from types import SimpleNamespace

from grpo_trl_trainer import RecGRPOTrainer


class CaptureMonitor:
    enabled = True

    def __init__(self):
        self.trace = None

    def write_rollout(self, event):
        return True

    def write_rank(self, event):
        return True

    def trace_due(self, rollout_id):
        return True

    def write_trace(self, event):
        self.trace = event
        return True


class FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        candidate = token_ids[0]
        return f"<s_a_{1000 + candidate}><s_b_42><s_c_7>"


trainer = object.__new__(RecGRPOTrainer)
trainer._monitor = CaptureMonitor()
trainer._generation_profile_log = []
trainer.state = SimpleNamespace(global_step=19)
trainer.num_generations = 8
trainer.processing_class = FakeTokenizer()
trainer.accelerator = SimpleNamespace(process_index=0)

entry = {
    "rollout_id": 20,
    "route": "no_think",
    "num_generations": 8,
    "recommendation_group_ids": ["group-0"],
    "reward_mean": 1.0,
    "reward_std": 2.0,
    "zero_std_ratio": 0.0,
    "completion_length_mean": 3.0,
    "completion_length_min": 3,
    "completion_length_max": 3,
    "gen_wall_sec": 1.0,
    "rollout_sec": 2.0,
}
inputs = [{"recommendation_group_id": "group-0", "all_gold_sids": []} for _ in range(4)]
global_rewards = [-1.0, -0.25, 0.0, 0.5, 2.0, 8.0, 0.0, 0.5]

trainer._write_rollout_monitor(
    entry,
    inputs,
    [[index] for index in range(4)],
    ["local"] * 4,
    global_rewards[:4],
    None,
    ["group-0"] * 8,
    [[index] for index in range(8)],
    global_rewards,
)

trace = trainer._monitor.trace
assert trace is not None
assert trace["scope"] == "global_group"
assert len(trace["candidates"]) == 8
assert [candidate["reward"] for candidate in trace["candidates"]] == global_rewards
assert [candidate["candidate_id"] for candidate in trace["candidates"]] == list(range(8))
print("[PASS] NoThink trace contains one complete G=8 group with aligned rewards")
