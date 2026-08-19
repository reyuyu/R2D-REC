import unittest

from benchmark_user_grpo_runtime import (
    CACHED_TIMED_STEPS,
    CACHED_WARMUP_STEPS,
    CONFIGS,
    GENERATION_TIMED_REPEATS,
    comparison_summary,
    estimate_wall_times,
    padding_counts,
    parity_accepted,
    schedule_rows,
    select_benchmark_rows,
    validate_rollout_cache,
)


def row(sample_id, route, length, *, gold_sid_count=0, gold_event_count=0):
    return {
        "sample_id": sample_id,
        "route": route,
        "prompt_token_count": length,
        "gold_sid_count": gold_sid_count,
        "gold_event_count": gold_event_count,
    }


class RuntimeBenchmarkContractTests(unittest.TestCase):
    def test_only_p0_p1_and_short_contract(self):
        self.assertEqual(["P0", "P1"], [item["name"] for item in CONFIGS])
        self.assertEqual(1, GENERATION_TIMED_REPEATS)
        self.assertEqual(2, CACHED_WARMUP_STEPS)
        self.assertEqual(5, CACHED_TIMED_STEPS)

    def test_selection_is_deterministic_and_covers_buckets(self):
        rows = []
        for low, high in ((2, 5), (6, 10), (11, 20), (21, 24)):
            for index in range(8):
                rows.append(row(f"a-{low}-{index}", "action", 100 + index * 50, gold_sid_count=min(low + index, high)))
        for events in (2, 3, 4, 5):
            for index in range(8):
                rows.append(row(f"c-{events}-{index}", "chain", 200 + index * 70, gold_event_count=events))
        first = select_benchmark_rows(rows)
        second = select_benchmark_rows(list(reversed(rows)))
        self.assertEqual(
            {key: [item["sample_id"] for item in value] for key, value in first.items()},
            {key: [item["sample_id"] for item in value] for key, value in second.items()},
        )
        self.assertEqual(20, len(first["action"]))
        self.assertEqual({2, 3, 4, 5}, {item["gold_event_count"] for item in first["chain"]})

    def test_schedule_preserves_groups_and_balances_final_round(self):
        rows = [row(f"x-{index}", "action", 100 + index * 101) for index in range(20)]
        rounds = schedule_rows(rows, length_bucketing=True)
        self.assertEqual([2, 2, 2, 2], [len(items) for items in rounds[0]])
        self.assertEqual([1, 1, 1, 1], [len(items) for items in rounds[-1]])
        flattened = [item["sample_id"] for batch in rounds for rank_rows in batch for item in rank_rows]
        self.assertEqual(20, len(flattened))
        self.assertEqual({item["sample_id"] for item in rows}, set(flattened))

    def test_length_bucketing_reduces_padding(self):
        rows = [row(f"x-{index}", "action", length) for index, length in enumerate([100, 1000] * 10)]
        unsorted = schedule_rows(rows, length_bucketing=False)
        bucketed = schedule_rows(rows, length_bucketing=True)
        unsorted_padding = sum(padding_counts(rank_rows)[0] for batch in unsorted for rank_rows in batch)
        bucketed_padding = sum(padding_counts(rank_rows)[0] for batch in bucketed for rank_rows in batch)
        self.assertLess(bucketed_padding, unsorted_padding)

    def test_cache_requires_all_g4_fields(self):
        selected = {"action": [row("a", "action", 10)], "chain": [row("c", "chain", 10)]}
        group = {
            "completion_ids": [[1], [2], [3], [4]],
            "rewards": [0.0] * 4,
            "sequence_advantages": [0.0] * 4,
            "token_advantages": [[0.0]] * 4,
            "penalty_masks": [[False]] * 4,
        }
        cache = {"G": 4, "groups": {"a": dict(group), "c": dict(group)}}
        validate_rollout_cache(cache, selected)
        cache["groups"]["a"]["token_advantages"][0] = []
        with self.assertRaises(RuntimeError):
            validate_rollout_cache(cache, selected)

    def test_p1_acceptance_uses_two_percent_and_high_cosine(self):
        base = {
            "reward_exact": True,
            "sequence_advantage_exact": True,
            "token_advantage_exact": True,
            "penalty_mask_exact": True,
            "loss_abs_error": 0.0,
            "gradient": {"relative_l2_error": 0.019, "cosine_similarity": 0.99991},
            "one_step_update": {"relative_l2_error": 0.20, "cosine_similarity": 0.9951},
        }
        self.assertTrue(parity_accepted(base))
        rejected = {**base, "gradient": {"relative_l2_error": 0.021, "cosine_similarity": 0.99991}}
        self.assertFalse(parity_accepted(rejected))
        rejected_update = {**base, "one_step_update": {"relative_l2_error": 0.01, "cosine_similarity": 0.9949}}
        self.assertFalse(parity_accepted(rejected_update))

    def test_estimates_add_generation_and_cached_policy(self):
        generation = [
            {"name": "P0", "seconds_per_prompt": 2.0},
            {"name": "P1", "seconds_per_prompt": 1.0},
        ]
        cached = [
            {"name": "P0", "seconds_per_prompt": 0.5},
            {"name": "P1", "seconds_per_prompt": 0.25},
        ]
        estimates = estimate_wall_times(generation, cached)
        self.assertEqual(375.0, estimates["P0"]["wall_seconds"]["150"])
        self.assertEqual(3750.0, estimates["P1"]["wall_seconds"]["3000"])
        comparison = comparison_summary(
            [
                {"generation_wall_seconds": 20.0, "prompt_padding_ratio": 0.2},
                {"generation_wall_seconds": 10.0, "prompt_padding_ratio": 0.1},
            ],
            [
                {"seconds_per_step": 5.0, "policy_padding_ratio": 0.4},
                {"seconds_per_step": 4.0, "policy_padding_ratio": 0.2},
            ],
            estimates,
        )
        self.assertEqual(100.0, comparison["generation_speedup_percent"])
        self.assertEqual(50.0, comparison["generation_padding_relative_reduction_percent"])


if __name__ == "__main__":
    unittest.main()
