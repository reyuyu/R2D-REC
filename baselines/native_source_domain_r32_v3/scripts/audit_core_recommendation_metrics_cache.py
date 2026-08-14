"""Read-only sanity check for pack-mean task exposure on verified BETA cache."""

from __future__ import annotations

from datasets import load_from_disk


CACHE = "/data/tokenized/onereason_beta_material_aligned_packratio_verified_v1"
TASKS = ("material", "recommendation", "user_action", "user_chain")


def pack_share(feature):
    labels = feature["labels"][1:]
    sample_ids = feature["sample_ids"][1:]
    task_ids = feature["sample_task_ids"][1:]
    segments = {}
    for label, sample_id, task_id in zip(labels, sample_ids, task_ids):
        if label != -100 and sample_id >= 0:
            segments.setdefault(sample_id, set()).add(task_id)
    assert segments and all(len(values) == 1 for values in segments.values())
    counts = [0] * len(TASKS)
    for values in segments.values():
        task_id = next(iter(values))
        assert 0 <= task_id < len(TASKS)
        counts[task_id] += 1
    total = sum(counts)
    return counts, [count / total for count in counts]


def main():
    dataset = load_from_disk(CACHE)["train"]
    assert len(dataset) == 33616
    indices = (0, len(dataset) // 3, 2 * len(dataset) // 3, len(dataset) - 1)
    for index in indices:
        counts, share = pack_share(dataset[index])
        assert abs(sum(share) - 1.0) < 1e-12
        print(f"PACK_SHARE index={index} segments={sum(counts)} " + ",".join(
            f"{task}={value:.6f}" for task, value in zip(TASKS, share)
        ))
    print("CORE_METRICS_REAL_CACHE_PASS")


if __name__ == "__main__":
    main()
