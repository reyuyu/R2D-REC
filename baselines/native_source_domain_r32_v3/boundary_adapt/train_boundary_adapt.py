"""Boundary Adapt loader and zero-update distributed preflight. Formal training is deliberately opt-in."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
sys.path.insert(0,"/data/GRPO/scripts")
from grpo_model import render_prompt
from boundary_adapt_loss import IGNORE_INDEX, assert_three_labels, weighted_boundary_loss

BASE="/data/models/onereason-8b-pretrain-competition"
ADAPTER="/data/outputs/baselines/native_source_domain_r32_v3/BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106"

def load(adapter_trainable=False, device="cuda:0"):
    tok=AutoTokenizer.from_pretrained(BASE,trust_remote_code=True)
    base=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.bfloat16,device_map=device,trust_remote_code=True,attn_implementation="flash_attention_2")
    model=PeftModel.from_pretrained(base,ADAPTER,is_trainable=adapter_trainable)
    for n,p in model.named_parameters(): p.requires_grad=("lora" in n.lower()) if adapter_trainable else False
    model.eval()  # parity preflight must not activate LoRA dropout.
    return model,tok

def encode_row(tok,row,device):
    text=render_prompt(tok,row["prompt"])+row["adapted_response"]
    ids=tok.encode(text,add_special_tokens=False)
    abc=tok.encode(row["boundary_gold_sid"],add_special_tokens=False)
    domain=tok.encode({"video":"<|video_begin|>","prod":"<|prod_begin|>","ad":"<|ad_begin|>","living":"<|living_begin|>"}[row["target_domain"]],add_special_tokens=False)
    if len(abc)!=3 or len(domain)!=1 or ids[-3:]!=abc: raise ValueError("token boundary contract failed")
    labels=[IGNORE_INDEX]*len(ids); labels[-3:]=abc; assert_three_labels(torch.tensor([labels]))
    return torch.tensor([ids],device=device),torch.tensor([labels],device=device),torch.tensor([row["boundary_sample_weight"]],device=device)

def checksum(model, *, lora: bool):
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if ("lora" in n.lower()) == lora: h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()

def preflight(rows_path,result_path):
    rank=int(os.environ.get("LOCAL_RANK","0")); world=int(os.environ.get("WORLD_SIZE","1")); torch.cuda.set_device(rank); dist.init_process_group("nccl")
    rows=[json.loads(x) for x in Path(rows_path).read_text(encoding="utf-8").splitlines()]
    targets=(("video",1),("living",2),("prod",1),("ad",2))
    domain, minimum_k=targets[rank]
    row=next((x for x in rows if x["target_domain"]==domain and x["boundary_group_size"]>=minimum_k),None)
    if row is None: raise ValueError(f"no preflight row for {domain}, K>={minimum_k}")
    device=f"cuda:{rank}"
    ref,tok=load(False,device); ids,labels,weights=encode_row(tok,row,device)
    with torch.inference_mode(): ref_logits=ref(input_ids=ids,attention_mask=torch.ones_like(ids)).logits.float()
    del ref; torch.cuda.empty_cache()
    model,_=load(True,device); before_lora=checksum(model,lora=True); before_base=checksum(model,lora=False)
    out=model(input_ids=ids,attention_mask=torch.ones_like(ids)); diff=(out.logits.float()-ref_logits).abs(); loss=weighted_boundary_loss(out.logits,labels,weights); loss.backward()
    lora_nonzero=any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for n,p in model.named_parameters() if "lora" in n.lower())
    base_nonzero=any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if "lora" not in n.lower())
    after_lora=checksum(model,lora=True); after_base=checksum(model,lora=False)
    top_ref=set(ref_logits[0,-4].topk(32).indices.tolist()); top_new=set(out.logits.float()[0,-4].topk(32).indices.tolist())
    item={"rank":rank,"loss":float(loss),"lora_grad_nonzero":bool(lora_nonzero),"base_grad_nonzero":bool(base_nonzero),"lora_checksum_unchanged":before_lora==after_lora,"base_checksum_unchanged":before_base==after_base,"max_abs_diff":float(diff.max()),"mean_abs_diff":float(diff.mean()),"top32_overlap":len(top_ref&top_new)/32,"optimizer_steps":0,"labels":int(labels.ne(IGNORE_INDEX).sum())}
    rank_path=Path(f"{result_path}.rank{rank}.json"); rank_path.write_text(json.dumps(item),encoding="utf-8")
    if rank==0:
        rank_paths=[Path(f"{result_path}.rank{i}.json") for i in range(world)]
        deadline=time.time()+120
        while not all(p.exists() for p in rank_paths):
            if time.time()>deadline: raise TimeoutError("rank result files did not appear")
            time.sleep(0.2)
        gathered=[json.loads(p.read_text()) for p in rank_paths]
        Path(result_path).write_text(json.dumps({"four_gpu_preflight":{"ranks":gathered,"optimizer_steps":0},"init_logit_parity":{"max_abs_diff":max(x["max_abs_diff"] for x in gathered),"mean_abs_diff":max(x["mean_abs_diff"] for x in gathered),"top32_overlap":min(x["top32_overlap"] for x in gathered)}},indent=2),encoding="utf-8")
    dist.destroy_process_group()

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--preflight-rows"); p.add_argument("--result"); a=p.parse_args();
    if not a.preflight_rows: raise SystemExit("Formal training intentionally not implemented; use --preflight-rows only.")
    preflight(a.preflight_rows,a.result)
