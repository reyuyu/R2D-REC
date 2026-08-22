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


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from user_mc_hybrid_distributed import distributed_hybrid_optimizer_step  # noqa: E402
from user_mc_hybrid_objective import mc_hybrid_objective  # noqa: E402
from user_mc_policy import get_mc_completion_logps  # noqa: E402


REWARDS = (0.4, 0.5, 0.9, 0.2)
DELTAS = (0.4, -0.2, None, 0.15)


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
        "completion_ids": torch.tensor([[3 + rank, 7, 8]]),
        "completion_mask": torch.tensor([[1.0, 1.0, 1.0]]),
    }


def candidate_units(delta):
    if delta is None:
        return [[]]
    return [[{"delta": delta, "generated_token_indices": [0, 1]}]]


def merged_reference_batch():
    batches = [candidate_batch(rank) for rank in range(4)]
    return {
        key: torch.cat([batch[key] for batch in batches], dim=0)
        for key in batches[0]
    }


def _worker(rank, rendezvous_path, output_path, rewards, deltas):
    if os.name != "nt":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=4,
    )
    try:
        torch.manual_seed(73)
        model = TinyCausalLM().double()
        initial = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        ddp = DistributedDataParallel(model)
        optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-3, weight_decay=0.0)
        result = distributed_hybrid_optimizer_step(
            ddp,
            optimizer,
            candidate_batch(rank),
            rewards[rank],
            candidate_units(deltas[rank]),
        )
        rank_results = [None] * 4
        dist.all_gather_object(
            rank_results,
            {
                "group_rewards": result["group_rewards"],
                "advantage": result["sequence_advantage"],
                "skipped": result["skipped_update"],
                "step": result["optimizer_step_performed"],
                "state_size": len(optimizer.state),
            },
        )
        if rank == 0:
            torch.save(
                {
                    "state": {
                        name: value.detach().cpu()
                        for name, value in ddp.module.state_dict().items()
                    },
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


class HybridDistributedParityTests(unittest.TestCase):
    def run_distributed(self, rewards, deltas):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.pt"
            rendezvous = Path(temporary) / "gloo-rendezvous"
            mp.spawn(
                _worker,
                args=(str(rendezvous), str(output), tuple(rewards), tuple(deltas)),
                nprocs=4,
                join=True,
            )
            return torch.load(output, map_location="cpu", weights_only=False)

    def test_four_rank_gradient_and_adamw_update_match_k4_reference(self):
        distributed = self.run_distributed(REWARDS, DELTAS)

        torch.manual_seed(73)
        reference = TinyCausalLM().double()
        optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3, weight_decay=0.0)
        batch = merged_reference_batch()
        logps = get_mc_completion_logps(reference, **batch)
        units = [candidate_units(delta)[0] for delta in DELTAS]
        loss, metadata = mc_hybrid_objective(
            logps, REWARDS, units, batch["completion_mask"]
        )
        loss.backward()
        reference_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in reference.named_parameters()
        }
        optimizer.step()

        expected_advantages = [
            float(value) for value in metadata["sequence_advantages"]
        ]
        expected_rewards = torch.tensor(REWARDS)
        self.assertTrue(
            all(
                torch.allclose(torch.tensor(item["group_rewards"]), expected_rewards)
                for item in distributed["rank_results"]
            )
        )
        for rank, item in enumerate(distributed["rank_results"]):
            self.assertAlmostEqual(item["advantage"], expected_advantages[rank], places=6)
            self.assertTrue(item["step"])
            self.assertFalse(item["skipped"])

        gradient_max_diff = max(
            float((distributed["grads"][name] - value).abs().max())
            for name, value in reference_grads.items()
        )
        update_max_diff = max(
            float((distributed["state"][name] - value).abs().max())
            for name, value in reference.state_dict().items()
        )
        print(
            "HYBRID_PARITY "
            f"gradient_max_diff={gradient_max_diff:.12g} "
            f"update_max_diff={update_max_diff:.12g}"
        )
        self.assertLessEqual(gradient_max_diff, 1e-6)
        self.assertLessEqual(update_max_diff, 1e-6)

    def test_equal_rewards_and_empty_local_units_skip_update(self):
        distributed = self.run_distributed((0.5,) * 4, (None,) * 4)
        self.assertTrue(
            all(item["skipped"] and not item["step"] for item in distributed["rank_results"])
        )
        self.assertTrue(
            all(item["state_size"] == 0 for item in distributed["rank_results"])
        )
        self.assertEqual(distributed["grads"], {})
        for name, initial in distributed["initial"].items():
            self.assertTrue(torch.equal(distributed["state"][name], initial))


if __name__ == "__main__":
    unittest.main()
