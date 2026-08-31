# -*- coding: utf-8 -*-
"""REC-MP-GRPO-v1 smoke: model loading + batched generation (inference-only)."""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from llamafactory.data.template import TEMPLATES

BASE = os.environ.get("GRPO_BASE_MODEL", "/data/models/onereason-8b-pretrain-competition")
DEFAULT_ADAPTER = os.environ.get(
    "GRPO_DEFAULT_PARENT_ADAPTER",
    "/data/outputs/baselines/native_source_domain_r32_v3/BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333",
)
ADAPTER = os.environ.get("GRPO_PARENT_ADAPTER", DEFAULT_ADAPTER)
TEMPLATE_NAME = "qwen3_nothink"


def load_model(device="cuda:0", dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    template = TEMPLATES[TEMPLATE_NAME]
    return model, tokenizer, template


def render_prompt(tokenizer, prompt: str) -> str:
    """SHARED prompt renderer: SFT/LLaMA-Factory qwen3_nothink template text.

    Renders the user message exactly as LLaMA-Factory's qwen3_nothink template
    does during SFT (format_user; no system), including the trailing
    '<|im_start|>assistant\n' generation head. TRL rollout and Beam32 must use
    this renderer so all three paths (SFT / TRL / Beam) see identical ids."""
    tpl = TEMPLATES[TEMPLATE_NAME]
    slots = tpl.format_user.apply(content=prompt)
    return "".join(str(s) for s in slots)


def encode_prompt(tokenizer, prompt: str):
    """qwen3_nothink single-user format (SFT-identical via render_prompt)."""
    return tokenizer.encode(render_prompt(tokenizer, prompt), add_special_tokens=False)


@torch.inference_mode()
def generate_batch(
    model, tokenizer, input_ids_list,
    max_new_tokens=128,
    min_new_tokens=None,
    do_sample=False,
    temperature=1.0,
    top_p=1.0,
    num_beams=1,
    num_return_sequences=1,
    return_ids=False,
):
    """Generate from a list of input prompts (each may produce R sequences).
    Returns flat list of texts ordered by (input_idx, sample_idx); with
    return_ids=True returns (texts, ids_list) where ids are the raw generated
    token ids (post right-pad strip)."""
    device = model.device
    B = len(input_ids_list)
    max_len = max(len(x) for x in input_ids_list)
    # TRUE decoder-only LEFT padding: each row's ids occupy the TRAILING part
    # [max_len-len(ids):max_len], pad (real pad token 151643) on the left.
    # Right-padding (ids at [0:len]) puts the pad token at the LAST position of
    # short rows, so generation starts from a pad token -> garbage output.
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_t = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, ids in enumerate(input_ids_list):
        start = max_len - len(ids)
        ids_t = torch.tensor(ids, dtype=torch.long, device=device)
        input_t[i, start:] = ids_t
        attn[i, start:] = 1
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        pad_token_id=pad_id,
    )
    if min_new_tokens is not None:
        gen_kwargs["min_new_tokens"] = min_new_tokens
    out = model.generate(inputs=input_t, attention_mask=attn, **gen_kwargs)
    texts = []
    ids_out = []
    for i in range(out.shape[0]):
        # generated tokens of EVERY row start at the unified padded width
        # max_len (generate appends new tokens after the input width), so we
        # must slice out[..., max_len:], NOT per-row plen.
        seq = out[i, max_len:].tolist()
        while seq and seq[-1] == pad_id:
            seq.pop()
        ids_out.append(seq)
        texts.append(tokenizer.decode(seq, skip_special_tokens=False))
    return (texts, ids_out) if return_ids else texts
