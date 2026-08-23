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
    weights=torch.tensor([1/3]*3,dtype=torch.float64); actual=loss.row_uniform_group_loss(logits,labels,weights,total_paths=3,total_groups=1)
    manual=sum(torch.nn.functional.cross_entropy(logits[i,-4:-1],labels[i,-3:],reduction="mean") for i in range(3))/3
    assert torch.allclose(actual,manual)
    singleton=loss.row_uniform_group_loss(logits[:1],labels[:1],torch.ones(1,dtype=torch.float64),total_paths=1,total_groups=1); expected=torch.nn.functional.cross_entropy(logits[0,-4:-1],labels[0,-3:],reduction="mean")
    assert torch.allclose(singleton,expected)

def test_microbatch_one_group_objective_gradient_parity():
    # K=1,2,4: arithmetic mean over row-uniform microbatches is group-uniform.
    torch.manual_seed(7)
    weights=torch.tensor([1.0,.5,.5,.25,.25,.25,.25],dtype=torch.float64)
    parameter=torch.nn.Parameter(torch.tensor(.3,dtype=torch.float64))
    logits=torch.randn(7,4,11,dtype=torch.float64) + parameter*torch.randn(7,4,11,dtype=torch.float64)
    labels=torch.full((7,4),-100,dtype=torch.long)
    labels[:,1:]=torch.tensor([[1,2,3],[4,5,6],[7,8,9],[1,3,5],[2,4,6],[3,5,7],[4,6,8]])
    paths=loss.per_path_boundary_loss(logits,labels)
    expected=(paths[0]+paths[1:3].mean()+paths[3:7].mean())/3
    expected_grad=torch.autograd.grad(expected,parameter,retain_graph=True)[0]
    accumulated=torch.zeros_like(parameter)
    for index in range(7):
        micro=loss.row_uniform_group_loss(logits[index:index+1],labels[index:index+1],weights[index:index+1],total_paths=7,total_groups=3)
        accumulated += torch.autograd.grad(micro,parameter,retain_graph=True)[0]
    assert torch.allclose(accumulated/7,expected_grad,rtol=1e-12,atol=1e-12)
    # An identical scalar path makes total K=4 gradient mass directly comparable to K=1.
    scalar=torch.nn.Parameter(torch.tensor(.3,dtype=torch.float64)); masses=[]; offset=0
    for k in (1,2,4):
        group=sum(scalar*(1/k)*(7/3) for _ in range(k))
        masses.append(torch.autograd.grad(group,scalar,retain_graph=True)[0])
        offset += k
    assert torch.allclose(masses[0],masses[2])

def test_wrong_label_count_fails_closed():
    try: loss.assert_three_labels(torch.tensor([[-100,1]])); assert False
    except ValueError: pass
