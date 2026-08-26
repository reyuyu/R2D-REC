"""Phase 1.2C1c CPU audit for the formal rollout EOS/PAD contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch


TRUE_REC_ROOT = Path(__file__).resolve().parents[1]
for dependency in (TRUE_REC_ROOT / "analysis", TRUE_REC_ROOT / "diagnostics", TRUE_REC_ROOT / "trainer"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from rollout_runtime_v1 import FORMAL_EOS_TOKEN_IDS, FORMAL_PAD_TOKEN_ID, GENERATION_SCORE_LOGPS_ROLE, GENERATION_SCORES_USED_FOR_PPO, PPO_OLD_LOGP_SOURCE, generation_contract, rollout_business_group, trim_generated_completion  # noqa: E402


VOCAB = 32
ROLLOUT_RUNTIME_SHA256 = "4643a0389d0bdcc7206b74cb36f18dbbd45183bdb7aa796cff2b86d045d1505b"


class MockOutput:
    def __init__(self, logits):
        self.logits = logits


class AuditModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.generate_kwargs: dict[str, Any] | None = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        completions = torch.tensor([[1, 2, 3]] * 8)
        output = type("GenerateOutput", (), {})()
        output.sequences = torch.cat([torch.tensor([[10, 11, 12, 13]] * 8), completions], dim=1)
        torch.manual_seed(120103)
        output.scores = tuple(torch.randn(8, VOCAB) for _ in range(3))
        return output

    def forward(self, input_ids, attention_mask):
        values = torch.arange(input_ids.shape[0] * input_ids.shape[1] * VOCAB, dtype=torch.float32)
        return MockOutput(values.reshape(input_ids.shape[0], input_ids.shape[1], VOCAB) / 1000.0 + self.anchor * 0.0)


class AuditRenderer:
    def rl_context_ids(self, system, user, domain):
        return [10, 11, 12, 13]


def record() -> dict[str, Any]:
    return {
        "recommendation_group_id": "phase1.2c1c-cpu",
        "system": "system",
        "user_content_nothink": "prompt",
        "fixed_domain_token": "<video>",
        "all_gold_abc": ("<s_a_1><s_b_2><s_c_3>",),
        "history_sids": (),
    }


def id_to_token(token_id: int) -> str:
    return {1: "<s_a_1>", 2: "<s_b_2>", 3: "<s_c_3>"}.get(int(token_id), f"token_{token_id}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(output_dir: Path) -> dict[str, Any]:
    model = AuditModel()
    group = rollout_business_group(
        model, record(), AuditRenderer(), id_to_token,
        FORMAL_PAD_TOKEN_ID, FORMAL_EOS_TOKEN_IDS, "cpu",
    )
    kwargs = model.generate_kwargs or {}
    sampling = {key: kwargs[key] for key in generation_contract()}
    rollout_path = TRUE_REC_ROOT / "trainer" / "rollout_runtime_v1.py"
    audit = {
        "status": "PASS",
        "formal_generate_eos_ids": kwargs.get("eos_token_id"),
        "formal_generate_pad_id": kwargs.get("pad_token_id"),
        "eos_passed_to_model_generate": kwargs.get("eos_token_id") == [151645, 151643],
        "pad_passed_to_model_generate": kwargs.get("pad_token_id") == 151643,
        "generation_contract": sampling,
        "generation_contract_match_phase07": sampling == generation_contract(),
        "trim": {
            "eos_151645": trim_generated_completion([1, 2, 151645, 9], FORMAL_EOS_TOKEN_IDS, FORMAL_PAD_TOKEN_ID),
            "eos_pad_151643": trim_generated_completion([1, 2, 151643, 151643], FORMAL_EOS_TOKEN_IDS, FORMAL_PAD_TOKEN_ID),
            "no_eos_trailing_pad": trim_generated_completion([1, 2, 151643, 151643], [151645], FORMAL_PAD_TOKEN_ID),
        },
        "G": len(group.candidates),
        "ppo_old_logp_source": PPO_OLD_LOGP_SOURCE,
        "generation_scores_used_for_ppo": GENERATION_SCORES_USED_FOR_PPO,
        "generation_score_logps_role": GENERATION_SCORE_LOGPS_ROLE,
        "rollout_runtime_sha256": hashlib.sha256(rollout_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "execution": {"gpu_inference_started": False, "real_generate_started": False, "frontier_started": False, "hpr_started": False, "backward_started": False, "optimizer_steps": 0},
    }
    required = (
        audit["eos_passed_to_model_generate"], audit["pad_passed_to_model_generate"],
        audit["generation_contract_match_phase07"], audit["trim"]["eos_151645"] == [1, 2, 151645],
        audit["trim"]["eos_pad_151643"] == [1, 2, 151643], audit["trim"]["no_eos_trailing_pad"] == [1, 2],
        audit["G"] == 8, audit["ppo_old_logp_source"] == "FULL_FORWARD_RESCORE",
        not audit["generation_scores_used_for_ppo"], audit["rollout_runtime_sha256"] == ROLLOUT_RUNTIME_SHA256,
    )
    if not all(required):
        audit["status"] = "FAIL"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "formal_rollout_eos_pad_contract.json", audit)
    (output_dir / "CHATGPT_PHASE1_2C1C_REVIEW.txt").write_text(
        "TrueRec-GRPO Phase 1.2C1c formal rollout EOS/PAD CPU contract audit: " + audit["status"] + "\n"
        "The mock generate call received frozen EOS/PAD ids; post-generation trimming and the full-forward PPO old-logp contract remained intact. No GPU, real generation, Frontier, HPR, backward, optimizer, or training ran.\n",
        encoding="utf-8",
    )
    if audit["status"] != "PASS":
        raise RuntimeError(f"C1c CPU contract failed: {audit}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = run(args.output_dir)
    print(json.dumps({"PHASE1_2C1C": audit["status"], "eos": audit["formal_generate_eos_ids"], "pad": audit["formal_generate_pad_id"]}))


if __name__ == "__main__":
    main()
