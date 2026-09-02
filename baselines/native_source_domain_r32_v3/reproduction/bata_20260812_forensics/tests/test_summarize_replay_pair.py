import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "summarize_replay_pair.py"
SPEC = importlib.util.spec_from_file_location("bata_summarize_replay_pair_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_run(root: Path, label: str, *, grad: float, final_suffix: str) -> Path:
    run = root / label
    evidence = run / "evidence"
    evidence.mkdir(parents=True)
    for rank in range(4):
        rows = [
            {
                "event": "optimizer_step",
                "rank": rank,
                "microbatch_count": 2,
                "ordered_batch_fingerprint": f"batch-{rank}",
                "ordered_microbatch_sha256": [f"micro-{rank}-0", f"micro-{rank}-1"],
                "rng": {"torch_cuda": f"rng-{rank}"},
                "rank_local_micro_losses": [1.0 + rank, 2.0 + rank],
                "rank_local_loss_mean": 1.5 + rank,
                "grad_norm": grad,
            }
        ]
        if rank == 0:
            rows.extend(
                [
                    {
                        "event": "heavy_fingerprint",
                        "label": "initial553",
                        "canonical_lora_sha256": "initial-lora",
                        "optimizer": {"sha256": "initial-optimizer"},
                    },
                    {
                        "event": "heavy_fingerprint",
                        "label": "step554",
                        "canonical_lora_sha256": f"lora-{final_suffix}",
                        "effective_ba": {"sha256": f"ba-{final_suffix}"},
                        "optimizer": {"sha256": f"optimizer-{final_suffix}"},
                    },
                ]
            )
        (evidence / f"rank{rank}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    return run


def test_d2_when_every_compared_value_matches(tmp_path: Path) -> None:
    a = write_run(tmp_path, "A", grad=0.5, final_suffix="same")
    b = write_run(tmp_path, "B", grad=0.5, final_suffix="same")
    result = MODULE.summarize_pair(a, b)
    assert result["classification"] == "D2"
    assert result["pre_backward_local_loss_repeatable"] is True
    assert result["post_backward_state_repeatable"] is True


def test_d3_when_backward_state_differs_after_equal_losses(tmp_path: Path) -> None:
    a = write_run(tmp_path, "A", grad=0.5, final_suffix="a")
    b = write_run(tmp_path, "B", grad=0.6, final_suffix="b")
    result = MODULE.summarize_pair(a, b)
    assert result["classification"] == "D3"
    assert result["pre_backward_local_loss_repeatable"] is True
    assert result["post_backward_state_repeatable"] is False
