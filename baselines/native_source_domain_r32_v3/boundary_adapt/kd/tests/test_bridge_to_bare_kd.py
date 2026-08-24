import importlib.util
import sys
from pathlib import Path

import torch

KD = Path(__file__).parents[1]
BOUNDARY = KD.parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module


kd = load(KD / "bridge_to_bare_kd_loss.py", "bridge_to_bare_kd_loss")
boundary = load(BOUNDARY / "boundary_adapt_loss.py", "boundary_adapt_loss")
trainer = None


def fixture():
    torch.manual_seed(3)
    student = torch.randn(7, 3, 17, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn(7, 3, 17, dtype=torch.float64, requires_grad=True)
    gold = torch.tensor([[1, 5, 9], [2, 6, 10], [3, 7, 11], [1, 6, 11], [2, 7, 9], [3, 5, 10], [1, 7, 11]])
    families = tuple(torch.tensor(value) for value in ([1, 2, 3], [5, 6, 7], [9, 10, 11]))
    weights = torch.tensor([1, .5, .5, .25, .25, .25, .25], dtype=torch.float64)
    return student, teacher, gold, families, weights


def test_lambda0_forward_and_gradient_parity():
    student, teacher, gold, families, weights = fixture()
    losses = kd.per_path_kd_losses(student, teacher, gold, families)
    actual = kd.row_uniform_kd_objective(losses, weights, lambda_kd=0, total_paths=7, total_groups=3)
    labels = torch.full((7, 4), -100, dtype=torch.long); labels[:, 1:] = gold
    logits = torch.zeros(7, 4, 17, dtype=torch.float64); logits[:, :3] = student
    expected = boundary.row_uniform_group_loss(logits, labels, weights, total_paths=7, total_groups=3)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.autograd.grad(actual, student, retain_graph=True)[0], torch.autograd.grad(expected, student)[0])


def test_family_isolation_detach_zero_and_ranking_swap():
    student, teacher, gold, families, _ = fixture()
    base = kd.per_path_kd_losses(student, teacher, gold, families)
    altered = teacher.clone(); altered[..., 0] += 1000; altered[..., 4] -= 1000; altered[..., 15] += 1000
    isolated = kd.per_path_kd_losses(student, altered, gold, families)
    for key in ("kl_A", "kl_B", "kl_C"):
        assert torch.equal(base[key], isolated[key])
    identical = kd.per_path_kd_losses(student, student, gold, families)["kd"]
    assert identical.abs().max() < 1e-12
    swapped = student.detach().clone(); swapped[:, 0, [1, 2]] = swapped[:, 0, [2, 1]]
    assert kd.per_path_kd_losses(student, swapped, gold, families)["kl_A"].max() > 0
    base["kd"].sum().backward()
    assert teacher.grad is None
    assert student.grad is not None and torch.isfinite(student.grad).all() and student.grad.abs().sum() > 0


def test_k1_k2_k4_group_objective_and_gradient_parity():
    student, teacher, gold, families, weights = fixture()
    losses = kd.per_path_kd_losses(student, teacher, gold, families)
    per_path = losses["gold"] + .3 * losses["kd"]
    expected = (per_path[0] + per_path[1:3].mean() + per_path[3:7].mean()) / 3
    actual = kd.row_uniform_kd_objective(losses, weights, lambda_kd=.3, total_paths=7, total_groups=3)
    assert torch.allclose(actual, expected, atol=1e-14, rtol=1e-14)
    expected_grad = torch.autograd.grad(expected, student, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(actual, student)[0]
    assert torch.allclose(actual_grad, expected_grad, atol=1e-14, rtol=1e-14)
    masses = [actual_grad[0].abs().sum(), actual_grad[1:3].abs().sum(), actual_grad[3:7].abs().sum()]
    assert all(torch.isfinite(value) for value in masses)


def test_bridge_insertion_anchors_final_boundary_suffix():
    global trainer
    if trainer is None:
        # Avoid importing GPU dependencies for the mathematical tests unless needed.
        trainer = load(KD / "train_bridge_to_bare_kd.py", "train_bridge_to_bare_kd")
    bridge = "\n原始桥接: "
    row = {"target_domain": "video", "boundary_gold_sid": "<s_a_1><s_b_2><s_c_3>",
           "adapted_response": "<think>正文提及<|video_begin|></think><|video_begin|><s_a_1><s_b_2><s_c_3>",
           "exact_bridge": bridge, "bridge_sha256": __import__("hashlib").sha256(bridge.encode()).hexdigest()}
    bare, teacher = trainer.response_pair(row)
    assert bare == row["adapted_response"]
    assert teacher == "<think>正文提及<|video_begin|></think>" + bridge + "<|video_begin|><s_a_1><s_b_2><s_c_3>"


def test_large_parameter_fingerprint_indices_stay_in_bounds():
    numel = 152064 * 4096
    indices = torch.arange(96, dtype=torch.int64) * (numel - 1) // 95
    assert int(indices[0]) == 0 and int(indices[-1]) == numel - 1
    assert bool(torch.all(indices[1:] > indices[:-1]))
