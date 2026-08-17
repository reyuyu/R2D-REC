"""CPU tests for formal-run planning, sampler audit, and checkpoint boundaries."""

from grpo_run_support import (
    audit_sampler,
    checkpoint_step,
    resolve_n_groups,
    validate_resume_checkpoint,
    validate_save_steps,
)
from grpo_trl_trainer import RouteAwareRepeatSampler
from run_grpo_trl_smoke import make_grpo_config
from run_grpo_trl_train import build_arg_parser


def expect_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError from {fn.__name__}{args!r}")


class Rows:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def __iter__(self):
        return iter(self.values)


assert resolve_n_groups("all", 1549) == 1549
assert resolve_n_groups("8", 1549) == 8
expect_error(resolve_n_groups, "0", 1549)
expect_error(resolve_n_groups, "1550", 1549)
expect_error(resolve_n_groups, "bad", 1549)

# Mirrors build_route_dataset(chunk=8): the final one-item route segments cannot
# fill either a Think 4-group chunk or a NoThink 2-group chunk.
think = []
nothink = []
for start in range(0, 1549, 8):
    group_ids = list(range(start, min(start + 8, 1549)))
    think.extend({"route": "think", "recommendation_group_id": gid} for gid in group_ids)
    nothink.extend({"route": "no_think", "recommendation_group_id": gid} for gid in group_ids)
rows = []
for start in range(0, 1549, 8):
    rows.extend(think[start:start + 8])
    rows.extend(nothink[start:start + 8])
dataset = Rows(rows)
sampler = RouteAwareRepeatSampler(dataset, generation_batch_size=16, repeat_count=2)
audit = audit_sampler(dataset, sampler)
assert audit["selected_groups"] == 1549
assert audit["trained_groups"] == 1548
assert audit["dropped_groups"] == 1
assert audit["dropped_group_ids"] == [1548]
assert audit["think_unique_groups"] == 1548
assert audit["nothink_unique_groups"] == 1548
assert audit["think_rollouts"] == 387
assert audit["nothink_rollouts"] == 774
assert audit["optimizer_steps"] == 2322

assert validate_save_steps(2) == 2
assert validate_save_steps(100) == 100
expect_error(validate_save_steps, 0)
expect_error(validate_save_steps, 3)
assert checkpoint_step("/tmp/run/checkpoint-120") == 120
assert validate_resume_checkpoint("/tmp/run/checkpoint-120") == 120
assert validate_resume_checkpoint(None) is None
expect_error(validate_resume_checkpoint, "/tmp/run/checkpoint-121")
expect_error(validate_resume_checkpoint, "/tmp/run/latest")

args = build_arg_parser().parse_args([
    "--run-id", "runner-test", "--n-groups", "all", "--max-steps", "120",
    "--lr", "1e-6", "--seed", "20260816", "--output-dir", "/tmp/grpo",
    "--save-steps", "20", "--save-total-limit", "3",
])
assert args.run_id == "runner-test"
assert args.n_groups == "all"
assert args.max_steps == 120
assert args.save_steps == 20
assert args.save_total_limit == 3

config = make_grpo_config(
    "/tmp/grpo/config-test", 120, 1e-6, 20260816,
    save_strategy="steps", save_steps=20, save_total_limit=3, use_cpu=True,
)
assert config.num_iterations == 2
assert config.num_generations == 4
assert config.per_device_train_batch_size == 4
assert config.max_prompt_length == 8192
assert config.max_completion_length == 2048
assert config.importance_sampling_level == "token"
assert config.top_entropy_quantile == 1.0
assert config.mask_truncated_completions is False
assert config.beta == 0.0 and config.epsilon == 0.2
assert config.loss_type == "grpo" and config.use_vllm is False
assert getattr(config.save_strategy, "value", config.save_strategy) == "steps"
assert config.save_steps == 20 and config.save_total_limit == 3
assert config.shuffle_dataset is False

print("FORMAL RUNNER CPU TESTS PASSED")
