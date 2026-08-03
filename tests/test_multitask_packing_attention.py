import importlib.util
import unittest

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask
from llamafactory.data.multitask import TaskPackCollator
from llamafactory.data.template import TEMPLATES
from llamafactory.train.sft.trainer import MultiTaskMacroSeq2SeqTrainer


VOCAB_SIZE = 64


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    vocab.update({f"token_{token_id}": token_id for token_id in range(4, VOCAB_SIZE)})
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        padding_side="right",
    )


def _tiny_qwen3(attn_implementation: str) -> Qwen3ForCausalLM:
    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        use_cache=False,
    )
    config._attn_implementation = attn_implementation
    model = Qwen3ForCausalLM(config)
    model.eval()
    assert model.config._attn_implementation == attn_implementation
    assert model.config.attention_dropout == 0.0
    assert all(module.p == 0.0 for module in model.modules() if isinstance(module, torch.nn.Dropout))
    return model


def _sample(input_ids: list[int], sample_id: int) -> dict:
    return {
        "input_ids": input_ids,
        "labels": list(input_ids),
        "task_name": "material",
        "task_id": 0,
        "subtask_name": "cot",
        "subtask_id": 0,
        "sample_id": sample_id,
        "supervised_token_count": len(input_ids),
        "sample_metadata": {"sample_id": sample_id},
    }


def _real_pack_collator(model: Qwen3ForCausalLM, attn_implementation: str, dtype: torch.dtype):
    base_collator = SFTDataCollatorWith4DAttentionMask(
        tokenizer=_tiny_tokenizer(),
        model=model,
        template=TEMPLATES["qwen3_nothink"],
        pad_to_multiple_of=None,
        label_pad_token_id=-100,
        block_diag_attn=True,
        neat_packing=True,
        attn_implementation=attn_implementation,
        compute_dtype=dtype,
    )
    return TaskPackCollator(base_collator)


def _model_inputs(batch: dict, device: torch.device) -> dict:
    inputs = {
        key: value
        for key, value in batch.items()
        if key in MultiTaskMacroSeq2SeqTrainer._MODEL_INPUT_KEYS
    }
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def _assert_reset_positions(batch: dict) -> None:
    for start, end in batch["segment_offsets"].tolist():
        assert batch["position_ids"][0, start:end].tolist() == list(range(end - start))


def _run_isolation_case(attn_implementation: str) -> tuple[float, float, set[str], object]:
    if attn_implementation == "flash_attention_2":
        device = torch.device("cuda")
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    model = _tiny_qwen3(attn_implementation).to(device=device, dtype=dtype)
    collator = _real_pack_collator(model, attn_implementation, dtype)

    a1 = _sample([5, 6, 7, 8, 9], sample_id=1)
    a2 = _sample([20, 21, 22, 23, 24], sample_id=2)
    b = _sample([40, 41, 42, 43], sample_id=3)
    pack_a1_b = collator([a1, b])
    pack_a2_b = collator([a2, b])
    pack_b = collator([b])

    for batch in (pack_a1_b, pack_a2_b, pack_b):
        _assert_reset_positions(batch)
        assert batch["cu_seqlens"].tolist() == [offset[0] for offset in batch["segment_offsets"].tolist()] + [
            batch["input_ids"].shape[-1]
        ]

    model_inputs_a1_b = _model_inputs(pack_a1_b, device)
    model_inputs_a2_b = _model_inputs(pack_a2_b, device)
    model_inputs_b = _model_inputs(pack_b, device)
    expected_fields = {"input_ids", "labels", "attention_mask", "position_ids"}
    assert set(model_inputs_a1_b) == expected_fields
    assert "cu_seqlens" in pack_a1_b
    assert "cu_seqlens" not in model_inputs_a1_b

    if attn_implementation == "flash_attention_2":
        assert model_inputs_a1_b["attention_mask"] is None
    else:
        assert model_inputs_a1_b["attention_mask"].shape == (1, 1, 9, 9)

    with torch.no_grad():
        logits_a1_b = model(**model_inputs_a1_b).logits.float().cpu()
        logits_a2_b = model(**model_inputs_a2_b).logits.float().cpu()
        logits_b = model(**model_inputs_b).logits.float().cpu()

    b_start, b_end = pack_a1_b["segment_offsets"][1].tolist()
    packed_a1_b_logits = logits_a1_b[:, b_start:b_end]
    packed_a2_b_logits = logits_a2_b[:, b_start:b_end]
    standalone_b_logits = logits_b[:, : b_end - b_start]
    cross_prefix_max_diff = (packed_a1_b_logits - packed_a2_b_logits).abs().max().item()
    standalone_max_diff = (packed_a1_b_logits - standalone_b_logits).abs().max().item()

    tolerance = 2e-2 if attn_implementation == "flash_attention_2" else 2e-5
    torch.testing.assert_close(packed_a1_b_logits, packed_a2_b_logits, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(packed_a1_b_logits, standalone_b_logits, rtol=tolerance, atol=tolerance)
    return cross_prefix_max_diff, standalone_max_diff, set(model_inputs_a1_b), model_inputs_a1_b["attention_mask"]


class MultiTaskPackingAttentionTest(unittest.TestCase):
    def test_eager_block_diagonal_attention_isolates_packed_segments(self):
        cross_prefix_diff, standalone_diff, fields, attention_mask = _run_isolation_case("eager")
        print(
            "eager",
            f"cross_prefix_max_diff={cross_prefix_diff:.8g}",
            f"standalone_max_diff={standalone_diff:.8g}",
            f"model_fields={sorted(fields)}",
            f"attention_mask_shape={tuple(attention_mask.shape)}",
        )

    @unittest.skipUnless(
        torch.cuda.is_available() and importlib.util.find_spec("flash_attn") is not None,
        "FlashAttention-2 requires CUDA and the flash_attn package.",
    )
    def test_flash_attention_2_isolates_packed_segments_from_reset_position_ids(self):
        cross_prefix_diff, standalone_diff, fields, attention_mask = _run_isolation_case("flash_attention_2")
        print(
            "flash_attention_2",
            f"cross_prefix_max_diff={cross_prefix_diff:.8g}",
            f"standalone_max_diff={standalone_diff:.8g}",
            f"model_fields={sorted(fields)}",
            f"attention_mask={attention_mask}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
