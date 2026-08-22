import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel


USER_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = USER_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from run_mc_user_formal_k4_ddp_v1 import (  # noqa: E402
    FROZEN_CONFIG,
    candidate_seed,
    distributed_mc_optimizer_step,
    load_k4_config,
    probe_queue_value,
    update_probe_queue,
)
from run_mc_user_formal_v1 import select_formal_rows  # noqa: E402
from user_mc_policy import compute_mc_model_loss  # noqa: E402


CONFIG_PATH = USER_DIR / "configs" / "mc_user_formal_stage1_512_k4_ddp.json"


class TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(17, 8)
        self.position_embedding = torch.nn.Embedding(16, 8)
        self.output = torch.nn.Linear(8, 17, bias=False)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
        logits_to_keep=None,
    ):
        hidden = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        logits = self.output(hidden)
        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


def candidate_batch(rank):
    return {
        "prompt_ids": torch.tensor([[1, 2]]),
        "prompt_mask": torch.ones((1, 2), dtype=torch.long),
        "completion_ids": torch.tensor([[3 + rank, 7]]),
        "completion_mask": torch.ones((1, 2), dtype=torch.float32),
    }


def candidate_units(delta):
    return [[{"delta": delta, "generated_token_indices": [0]}]]


def merged_reference_batch():
    batches = [candidate_batch(rank) for rank in range(4)]
    return {
        key: torch.cat([batch[key] for batch in batches], dim=0)
        for key in batches[0]
    }


def _ddp_worker(rank, rendezvous_path, output_path, deltas):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=4,
    )
    try:
        torch.manual_seed(41)
        model = TinyCausalLM()
        initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
        ddp = DistributedDataParallel(model)
        optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-3, weight_decay=0.0)
        result = distributed_mc_optimizer_step(
            ddp,
            optimizer,
            candidate_batch(rank),
            candidate_units(deltas[rank]),
        )
        rank_results = [None] * 4
        dist.all_gather_object(
            rank_results,
            {
                "loss": result["loss"],
                "skipped": result["skipped_update"],
                "step": result["optimizer_step_performed"],
                "state_size": len(optimizer.state),
            },
        )
        if rank == 0:
            torch.save(
                {
                    "state": {name: value.detach().cpu() for name, value in ddp.module.state_dict().items()},
                    "initial": initial,
                    "grads": {
                        name: parameter.grad.detach().cpu()
                        for name, parameter in ddp.module.named_parameters()
                        if parameter.grad is not None
                    },
                    "rank_results": rank_results,
                },
                output_path,
            )
    finally:
        dist.destroy_process_group()


class K4ContractTests(unittest.TestCase):
    def test_frozen_config_and_selection_contract(self):
        self.assertEqual(load_k4_config(CONFIG_PATH), FROZEN_CONFIG)
        rows = [
            {"sample_id": f"{route}-{index:04d}", "route": route}
            for route in ("action", "chain")
            for index in range(300)
        ]
        selected = select_formal_rows(rows, FROZEN_CONFIG["selection_seed"])
        self.assertEqual(len(selected), 512)
        self.assertEqual(len({row["sample_id"] for row in selected}), 512)
        self.assertEqual(
            [row["route"] for row in selected],
            [route for _ in range(256) for route in ("action", "chain")],
        )

    def test_candidate_seeds_are_stable_and_distinct(self):
        seeds = [candidate_seed(20260823, 1, "sample", rank) for rank in range(4)]
        self.assertEqual(len(set(seeds)), 4)
        self.assertEqual(
            seeds,
            [candidate_seed(20260823, 1, "sample", rank) for rank in range(4)],
        )

    def test_probe_queue_contract(self):
        queue = probe_queue_value()
        self.assertEqual([item["step"] for item in queue["items"]], [0, 128, 256, 384, 512])
        updated = update_probe_queue(queue, 128)
        self.assertEqual(updated["items"][1]["status"], "ready")
        self.assertTrue(updated["items"][1]["available_for_probe"])
        self.assertEqual(queue["items"][1]["status"], "waiting")


class K4DistributedParityTests(unittest.TestCase):
    def run_distributed(self, deltas):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.pt"
            rendezvous = Path(temporary) / "gloo-rendezvous"
            mp.spawn(
                _ddp_worker,
                args=(str(rendezvous), str(output), tuple(deltas)),
                nprocs=4,
                join=True,
            )
            return torch.load(output, map_location="cpu", weights_only=False)

    def test_loss_gradient_and_adamw_update_match_k4_reference(self):
        deltas = (0.5, -0.25, 0.0, 0.75)
        distributed = self.run_distributed(deltas)

        torch.manual_seed(41)
        reference = TinyCausalLM()
        optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3, weight_decay=0.0)
        units = [candidate_units(delta)[0] for delta in deltas]
        loss, metadata, _ = compute_mc_model_loss(reference, merged_reference_batch(), units)
        self.assertEqual(metadata["active_unit_count"], 3)
        loss.backward()
        reference_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in reference.named_parameters()
        }
        optimizer.step()

        self.assertLessEqual(abs(distributed["rank_results"][0]["loss"] - float(loss)), 1e-6)
        grad_diff = max(
            float((distributed["grads"][name] - value).abs().max())
            for name, value in reference_grads.items()
        )
        update_diff = max(
            float((distributed["state"][name] - value).abs().max())
            for name, value in reference.state_dict().items()
        )
        self.assertLessEqual(grad_diff, 1e-6)
        self.assertLessEqual(update_diff, 1e-6)
        self.assertTrue(all(item["step"] and not item["skipped"] for item in distributed["rank_results"]))

    def test_all_zero_skips_backward_and_optimizer_state(self):
        distributed = self.run_distributed((0.0, 0.0, 0.0, 0.0))
        self.assertTrue(all(item["skipped"] and not item["step"] for item in distributed["rank_results"]))
        self.assertTrue(all(item["state_size"] == 0 for item in distributed["rank_results"]))
        for name, initial in distributed["initial"].items():
            self.assertTrue(torch.equal(distributed["state"][name], initial))
        self.assertEqual(distributed["grads"], {})


if __name__ == "__main__":
    unittest.main()
