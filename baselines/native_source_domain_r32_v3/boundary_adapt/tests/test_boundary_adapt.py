import importlib.util
from pathlib import Path
import torch

ROOT=Path(__file__).parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/f"{name}.py"); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
b=load("build_boundary_adapt_dataset"); loss=load("boundary_adapt_loss")

def sample():
    return {"route":"think","recommendation_group_id":"g","prompt":"P","target_domain":"video","response":"<think>原始 CoT，逐字保留。</think>这是必须删除的 bridge。<|video_begin|><s_a_1><s_b_2><s_c_3>","recommendation_all_gold_sids":["<s_a_1><s_b_2><s_c_3>","<s_a_4><s_b_5><s_c_6>","<s_a_7><s_b_8><s_c_9>"]}

def test_expand_bridge_domain_cot_and_weight():
    rows=b.expand_row(sample()); assert len(rows)==3
    assert sum(x["boundary_sample_weight"] for x in rows)==1.0
    assert all(x["adapted_response"].startswith("<think>原始 CoT，逐字保留。</think><|video_begin|>") for x in rows)
    assert all("这是必须删除的 bridge" not in x["adapted_response"] for x in rows)
    assert len({x["cot_sha256"] for x in rows})==1

def test_domain_mapping_and_fail_closed():
    sid="<s_a_1><s_b_2><s_c_3>"
    for domain, token in b.DOMAIN.items(): assert b.adapt_response("<think>x</think>bridge",domain,sid)[0]==f"<think>x</think>{token}{sid}"
    for bad in ("<s_a_1><s_b_2>", "<s_a_1><s_b_2><s_c_3>x"):
        try: b.adapt_response("<think>x</think>b","video",bad); assert False
        except ValueError: pass
    try: b.adapt_response("no think","video",sid); assert False
    except ValueError: pass

def test_three_labels_and_full_path_weighted_parity():
    torch.manual_seed(0); logits=torch.randn(3,6,11,dtype=torch.float64); labels=torch.full((3,6),-100,dtype=torch.long); labels[:,-3:]=torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
    weights=torch.tensor([1/3]*3,dtype=torch.float64); actual=loss.weighted_boundary_loss(logits,labels,weights)
    manual=sum(torch.nn.functional.cross_entropy(logits[i,-4:-1],labels[i,-3:],reduction="mean") for i in range(3))/3
    assert torch.allclose(actual,manual)
    singleton=loss.weighted_boundary_loss(logits[:1],labels[:1],torch.ones(1,dtype=torch.float64)); expected=torch.nn.functional.cross_entropy(logits[0,-4:-1],labels[0,-3:],reduction="mean")
    assert torch.allclose(singleton,expected)

def test_wrong_label_count_fails_closed():
    try: loss.assert_three_labels(torch.tensor([[-100,1]])); assert False
    except ValueError: pass
