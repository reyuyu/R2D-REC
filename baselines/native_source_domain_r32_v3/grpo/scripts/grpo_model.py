# -*- coding: utf-8 -*-
"""REC-MP-GRPO-v1 smoke: model loading + batched generation (inference-only)."""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from llamafactory.data.template import TEMPLATES

BASE = "/data/models/onereason-8b-pretrain-competition"
ADAPTER = "/data/outputs/baselines/native_source_domain_r32_v3/BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333"
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
    do_sample=False,
    temperature=1.0,
    top_p=1.0,
    num_beams=1,
    num_return_sequences=1,
):
    """Generate from a list of input prompts (each may produce R sequences).
    Returns flat list of texts ordered by (input_idx, sample_idx)."""
    device = model.device
    B = len(input_ids_list)
    max_len = max(len(x) for x in input_ids_list)
    # decoder-only correct LEFT padding: pad with the real pad token (151643),
    # NOT 0 (which is the plain char '!') and NOT eos (151645); attention mask
    # keeps padded positions out of the computation.
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_t = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, ids in enumerate(input_ids_list):
        ids_t = torch.tensor(ids, dtype=torch.long, device=device)
        input_t[i, :len(ids)] = ids_t
        attn[i, :len(ids)] = 1
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        pad_token_id=pad_id,
    )
    out = model.generate(inputs=input_t, attention_mask=attn, **gen_kwargs)
    texts = []
    for i in range(out.shape[0]):
        input_idx = i // num_return_sequences
        plen = len(input_ids_list[input_idx])
        seq = out[i, plen:].tolist()
        # strip trailing right-pad tokens so decode never sees pad garbage;
        # each output maps back to its OWN input context via per-row plen.
        while seq and seq[-1] == pad_id:
            seq.pop()
        texts.append(tokenizer.decode(seq, skip_special_tokens=False))
    return texts
