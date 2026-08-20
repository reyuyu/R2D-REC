import json
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from run_mc_user_generation_smoke import (  # noqa: E402
    K,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    extract_generated_ids,
    generate_k2_route,
    gpu_preflight,
    run_cli,
    score_generated_route,
)


SID_RE = re.compile(
    r"<\|(?:video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>"
)
A = "<|video_begin|><s_a_101><s_b_201><s_c_301>"
B = "<|prod_begin|><s_a_102><s_b_202><s_c_302>"


class GenerationTokenizer:
    pad_token_id = 0
    eos_token_id = 9001
    im_end_token_id = 9002

    def __init__(self):
        self.piece_to_id = {}
        self.id_to_piece = {
            self.eos_token_id: "<eos>",
            self.im_end_token_id: "<|im_end|>",
            9003: "ignored-after-stop",
        }
        self.batch_calls = []

    def _id(self, piece):
        if piece not in self.piece_to_id:
            token_id = len(self.piece_to_id) + 1
            self.piece_to_id[piece] = token_id
            self.id_to_piece[token_id] = piece
        return self.piece_to_id[piece]

    @staticmethod
    def _sid_pieces(sid, base):
        boundaries = [
            0,
            sid.index("><s_a_") + 1,
            sid.index("><s_b_") + 1,
            sid.index("><s_c_") + 1,
            len(sid),
        ]
        return [
            (
                sid[boundaries[index] : boundaries[index + 1]],
                base + boundaries[index],
                base + boundaries[index + 1],
            )
            for index in range(4)
        ]

    def _pieces(self, text):
        pieces = []
        cursor = 0
        for match in SID_RE.finditer(text):
            pieces.extend(
                (text[index], index, index + 1)
                for index in range(cursor, match.start())
            )
            pieces.extend(self._sid_pieces(match.group(), match.start()))
            cursor = match.end()
        pieces.extend(
            (text[index], index, index + 1) for index in range(cursor, len(text))
        )
        return pieces

    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_offsets_mapping=False,
        padding=False,
        padding_side=None,
        return_tensors=None,
    ):
        if isinstance(text, list):
            self.batch_calls.append(
                {"padding": padding, "padding_side": padding_side, "count": len(text)}
            )
            ids = [self.encode(value, add_special_tokens=False) for value in text]
            width = max(map(len, ids))
            padded = [[self.pad_token_id] * (width - len(value)) + value for value in ids]
            masks = [[0] * (width - len(value)) + [1] * len(value) for value in ids]
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        pieces = self._pieces(text)
        result = {"input_ids": [self._id(piece) for piece, _, _ in pieces]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(start, end) for _, start, end in pieces]
        return result

    def encode(self, text, add_special_tokens=False):
        return self(text, add_special_tokens=add_special_tokens)["input_ids"]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.id_to_piece[int(token_id)] for token_id in ids)

    def convert_tokens_to_ids(self, token):
        return self.im_end_token_id if token == "<|im_end|>" else -1

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ):
        return f"<chat>{messages[0]['content']}</chat>"

    def noncanonical_split(self, completion, canonical_index):
        canonical_ids = self.encode(completion, add_special_tokens=False)
        old_id = canonical_ids[canonical_index]
        piece = self.id_to_piece[old_id]
        split_at = max(1, len(piece) // 2)
        first_id = max(self.id_to_piece) + 1
        second_id = first_id + 1
        self.id_to_piece[first_id] = piece[:split_at]
        self.id_to_piece[second_id] = piece[split_at:]
        return canonical_ids[:canonical_index] + [first_id, second_id] + canonical_ids[canonical_index + 1 :]


class FakeGenerateModel:
    def __init__(self, candidate_suffixes):
        self.candidate_suffixes = candidate_suffixes
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        prompt = kwargs["input_ids"][0].tolist()
        return torch.tensor(
            [prompt + list(suffix) for suffix in self.candidate_suffixes],
            dtype=torch.long,
        )


def action_row(tokenizer):
    row = {
        "sample_id": "action-generation",
        "route": "action",
        "prompt": "predict action",
        "gold_sids": [A, B],
        "history_sids": [A, B],
    }
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    row["prompt_token_count"] = len(tokenizer.encode(rendered, add_special_tokens=False))
    return row


class GenerationContractTests(unittest.TestCase):
    def test_k2_prompt_strip_and_stop_trimming_preserve_exact_ids(self):
        output = torch.tensor(
            [[7, 8, 101, 102, 9001, 999], [7, 8, 201, 9002, 999, 999]],
            dtype=torch.long,
        )
        actual = extract_generated_ids(
            output, prompt_width=2, stop_ids={9001, 9002}
        )
        self.assertEqual(actual, [[101, 102], [201]])

    def test_non_k2_generation_fails(self):
        with self.assertRaisesRegex(RuntimeError, "K=2"):
            extract_generated_ids([[1, 2, 3]], prompt_width=1, stop_ids={9})

    def test_generate_uses_frozen_config_left_padding_and_order(self):
        tokenizer = GenerationTokenizer()
        row = action_row(tokenizer)
        completions = [json.dumps([A]), json.dumps([B])]
        ids = [tokenizer.encode(text) for text in completions]
        model = FakeGenerateModel(
            [ids[0] + [tokenizer.eos_token_id, 9003], ids[1] + [tokenizer.im_end_token_id, 9003]]
        )
        generated, contract = generate_k2_route(
            model, tokenizer, row, torch.device("cpu")
        )
        self.assertEqual(generated, ids)
        self.assertEqual(contract["prompt_token_count"], row["prompt_token_count"])
        self.assertEqual(tokenizer.batch_calls[-1]["padding_side"], "left")
        self.assertEqual(model.kwargs["num_return_sequences"], K)
        self.assertEqual(model.kwargs["temperature"], TEMPERATURE)
        self.assertEqual(model.kwargs["top_p"], TOP_P)
        self.assertEqual(model.kwargs["max_new_tokens"], MAX_NEW_TOKENS)
        self.assertTrue(model.kwargs["do_sample"])
        self.assertTrue(model.kwargs["use_cache"])

    def test_malformed_candidate_is_retained_in_policy_batch(self):
        tokenizer = GenerationTokenizer()
        row = action_row(tokenizer)
        valid = tokenizer.encode(json.dumps([A]))
        malformed = tokenizer.encode("not-json")
        scored = score_generated_route(
            "action", row, [valid, malformed], tokenizer
        )
        self.assertEqual(scored["candidate_count"], 2)
        self.assertTrue(scored["candidates"][0]["format_valid"])
        self.assertFalse(scored["candidates"][1]["format_valid"])
        self.assertEqual(scored["rollout"]["credit_units_per_candidate"][1], [])
        self.assertEqual(
            scored["policy_batch"]["completion_ids"][1, : len(malformed)].tolist(),
            malformed,
        )

    def test_noncanonical_generated_ids_reach_projection_unchanged(self):
        tokenizer = GenerationTokenizer()
        row = action_row(tokenizer)
        completion = json.dumps([A])
        canonical = tokenizer.encode(completion)
        sid_start_token = next(
            index
            for index, token_id in enumerate(canonical)
            if tokenizer.id_to_piece[token_id].startswith("<|video_begin|>")
        )
        generated = tokenizer.noncanonical_split(completion, sid_start_token + 1)
        scored = score_generated_route(
            "action", row, [generated, list(generated)], tokenizer
        )
        self.assertTrue(scored["candidates"][0]["projection_required"])
        self.assertEqual(scored["rollout"]["completion_ids_list"][0], generated)
        self.assertEqual(
            scored["policy_batch"]["completion_ids"][0, : len(generated)].tolist(),
            generated,
        )

    def test_route_must_be_homogeneous(self):
        tokenizer = GenerationTokenizer()
        row = action_row(tokenizer)
        ids = tokenizer.encode(json.dumps([A]))
        with self.assertRaisesRegex(RuntimeError, "homogeneous"):
            score_generated_route("chain", row, [ids, ids], tokenizer)


class SafetyGateTests(unittest.TestCase):
    def test_runner_has_no_training_or_checkpoint_path(self):
        source = (SCRIPTS_DIR / "run_mc_user_generation_smoke.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn(".backward(", source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn("save_pretrained", source)

    def test_without_execute_does_not_call_model_execution(self):
        preflight = Mock(
            return_value={
                "status": "READY_TO_EXECUTE",
                "gpu": {"index": 0},
                "selected_sample_ids": {"action": "a", "chain": "c"},
                "prompt_token_counts": {"action": 1, "chain": 1},
                "paths": {"train_sha256": "sha"},
            }
        )
        execute = Mock()
        output = run_cli(
            ["--gpu-id", "0"], preflight_fn=preflight, execute_fn=execute
        )
        self.assertEqual(output["status"], "READY_TO_EXECUTE")
        execute.assert_not_called()

    def test_execute_requires_explicit_gate(self):
        preflight_result = {"status": "READY_TO_EXECUTE"}
        preflight = Mock(return_value=preflight_result)
        execute = Mock(return_value={"status": "PASS"})
        output = run_cli(
            ["--gpu-id", "0", "--execute"],
            preflight_fn=preflight,
            execute_fn=execute,
        )
        self.assertEqual(output["status"], "PASS")
        execute.assert_called_once()

    def test_busy_gpu_is_refused(self):
        responses = [
            SimpleNamespace(stdout="0, GPU-0, 100, 80000, 0\n"),
            SimpleNamespace(stdout="GPU-0, 123, python, 100\n"),
        ]
        with self.assertRaisesRegex(RuntimeError, "busy"):
            gpu_preflight(0, run_command=Mock(side_effect=responses))


if __name__ == "__main__":
    unittest.main()
