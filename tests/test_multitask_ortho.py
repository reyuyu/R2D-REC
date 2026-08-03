import copy
import math
import tempfile
from types import SimpleNamespace

import torch
from torch import nn

from llamafactory.hparams.data_args import DataArguments
from llamafactory.train.sft.multitask_gradient_controller import MultiTaskGradientController


TASKS = ("material", "user", "recommendation")


class WeightHolder(nn.Module):
    def __init__(self, width=3):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))


class ToyOrthoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        block = self.model.layers[0]
        block.self_attn = nn.Module()
        block.self_attn.q_proj = nn.Module()
        block.self_attn.q_proj.lora_A = nn.ModuleDict({"default": WeightHolder()})
        block.self_attn.q_proj.lora_B = nn.ModuleDict({"default": WeightHolder()})
        self.unselected = nn.Parameter(torch.zeros(3))

    def forward(self, feature):
        selected = self.model.layers[0].self_attn.q_proj.lora_B["default"].weight
        return torch.dot(selected, feature) + torch.dot(self.unselected, feature)


def make_args(**overrides):
    values = {
        "multitask_gradnorm_tasks": list(TASKS),
        "multitask_gradient_monitor_enabled": False,
        "multitask_gradnorm_enabled": False,
        "multitask_world_loss_weight": 1.0,
        "multitask_gradnorm_warmup_steps": 200,
        "multitask_gradnorm_update_interval": 10,
        "multitask_gradnorm_alpha": 0.5,
        "multitask_gradnorm_update_rate": 0.1,
        "multitask_gradnorm_loss_ema_beta": 0.9,
        "multitask_gradnorm_grad_ema_beta": 0.9,
        "multitask_gradnorm_weight_min": 0.5,
        "multitask_gradnorm_weight_max": 2.0,
        "multitask_gradnorm_step_ratio_min": 0.9,
        "multitask_gradnorm_step_ratio_max": 1.1,
        "multitask_grad_reference_last_n_layers": 1,
        "multitask_grad_reference_modules": ["q_proj"],
        "multitask_grad_reference_lora_matrix": "B",
        "multitask_ortho_enabled": True,
        "multitask_ortho_monitor_only": False,
        "multitask_ortho_start_step": 0,
        "multitask_ortho_interval": 1,
        "multitask_ortho_current_cosine_threshold": -0.05,
        "multitask_ortho_ema_cosine_threshold": 0.0,
        "multitask_ortho_cosine_ema_beta": 0.9,
        "multitask_ortho_use_ema_gate": True,
        "multitask_ortho_norm_ratio_min": 0.7,
        "multitask_ortho_norm_ratio_max": 1.3,
        "multitask_ortho_rotate_order": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def selected_grad(controller):
    return torch.cat([item.parameter.grad.reshape(-1) for item in controller.reference_parameters])


def run_step(model, controller, vectors, step=0, world=None):
    model.zero_grad(set_to_none=True)
    controller.begin_macro_step(step)
    for task in TASKS:
        controller.set_current_task(task)
        try:
            model(vectors[task]).mul(controller.task_weight(task)).backward()
        finally:
            controller.clear_current_task()
    if world is not None:
        model(world).mul(controller.task_weight("world")).backward()
    metrics = controller.finish_macro_step(dict.fromkeys(TASKS, 1.0), dict.fromkeys(TASKS, 1))
    return metrics


def conflict_vectors():
    return {
        "material": torch.tensor([1.0, 0.0, 0.0]),
        "user": torch.tensor([-1.0, 1.0, 0.0]),
        "recommendation": torch.tensor([0.0, 1.0, 1.0]),
    }


def positive_vectors():
    return {
        "material": torch.tensor([1.0, 0.0, 0.0]),
        "user": torch.tensor([1.0, 1.0, 0.0]),
        "recommendation": torch.tensor([1.0, 0.0, 1.0]),
    }


def test_data_arguments_accept_ortho_only_and_validate_monitor_only():
    args = DataArguments(
        multitask_macro_training=True,
        multitask_gradient_control_enabled=True,
        multitask_gradnorm_enabled=False,
        multitask_gradient_monitor_enabled=False,
        multitask_ortho_enabled=True,
    )
    assert args.multitask_ortho_enabled and not args.multitask_gradnorm_enabled
    try:
        DataArguments(multitask_ortho_monitor_only=True)
    except ValueError as error:
        assert "requires multitask_ortho_enabled" in str(error)
    else:
        raise AssertionError("Expected monitor-only validation error.")


def test_ortho_disabled_is_experiment_a_path():
    model = ToyOrthoModel()
    controller = MultiTaskGradientController(
        model,
        make_args(
            multitask_gradient_monitor_enabled=True,
            multitask_gradnorm_update_interval=1,
            multitask_ortho_enabled=False,
        ),
        loss_divisor=1,
    )
    expected = sum(conflict_vectors().values())
    metrics = run_step(model, controller, conflict_vectors())
    assert torch.equal(selected_grad(controller), expected)
    assert not any(key.startswith("mtg/ortho/") for key in metrics)
    assert not controller._ortho_buffers
    assert "cosine_ema" not in controller.state_dict()


def test_start_and_interval_gate_capture():
    controller = MultiTaskGradientController(
        ToyOrthoModel(), make_args(multitask_ortho_start_step=4, multitask_ortho_interval=2), loss_divisor=1
    )
    controller.begin_macro_step(2)
    assert not controller.ortho_step_due and not controller.capture_enabled
    controller.begin_macro_step(5)
    assert not controller.ortho_step_due and not controller.capture_enabled
    controller.begin_macro_step(6)
    assert controller.ortho_step_due and controller.capture_enabled


def test_monitor_only_simulates_but_never_changes_grad():
    model = ToyOrthoModel()
    controller = MultiTaskGradientController(
        model, make_args(multitask_ortho_monitor_only=True, multitask_ortho_use_ema_gate=False), loss_divisor=1
    )
    expected = sum(conflict_vectors().values())
    metrics = run_step(model, controller, conflict_vectors())
    assert torch.equal(selected_grad(controller), expected)
    assert metrics["mtg/ortho/pair_count"] >= 1
    assert metrics["mtg/ortho/removed_ratio"] > 0
    assert metrics["mtg/ortho/time/writeback_ms"] == 0


def test_positive_cosines_do_not_project_and_reconstruction_is_exact():
    model = ToyOrthoModel()
    controller = MultiTaskGradientController(model, make_args(multitask_ortho_use_ema_gate=False), loss_divisor=1)
    expected = sum(positive_vectors().values())
    metrics = run_step(model, controller, positive_vectors())
    assert torch.equal(selected_grad(controller), expected)
    assert metrics["mtg/ortho/pair_count"] == 0
    assert metrics["mtg/ortho/removed_ratio"] == 0
    assert torch.equal(controller._ortho_buffers["new_total"], expected)


def test_current_conflict_but_ema_above_threshold_does_not_project():
    vectors = conflict_vectors()
    model = ToyOrthoModel()
    controller = MultiTaskGradientController(model, make_args(), loss_divisor=1)
    controller.cosine_ema["material_user"] = 0.2
    expected = sum(vectors.values())
    metrics = run_step(model, controller, vectors)
    assert controller.last_cosines["material_user"] < controller.ortho_current_cosine_threshold
    assert controller.cosine_ema["material_user"] >= controller.ortho_ema_cosine_threshold
    assert metrics["mtg/ortho/pair/material_user"] == 0
    assert torch.equal(selected_grad(controller), expected)


def test_current_nonnegative_blocks_historical_negative_ema():
    model = ToyOrthoModel()
    controller = MultiTaskGradientController(model, make_args(), loss_divisor=1)
    controller.cosine_ema["material_user"] = -1.0
    metrics = run_step(model, controller, positive_vectors())
    assert controller.cosine_ema["material_user"] < controller.ortho_ema_cosine_threshold
    assert controller.last_cosines["material_user"] >= 0
    assert metrics["mtg/ortho/pair/material_user"] == 0


def test_split_current_and_ema_thresholds_must_both_pass():
    model = ToyOrthoModel()
    controller = MultiTaskGradientController(model, make_args(), loss_divisor=1)
    controller.last_raw_norms.update(material=1.0, user=1.0)

    controller.last_cosines["material_user"] = -0.04
    controller.cosine_ema["material_user"] = -0.1
    assert not controller._pair_passes_gate("material_user")

    controller.last_cosines["material_user"] = -0.06
    controller.cosine_ema["material_user"] = 0.01
    assert not controller._pair_passes_gate("material_user")

    controller.cosine_ema["material_user"] = -0.01
    assert controller._pair_passes_gate("material_user")


def test_clear_negative_conflict_projection_weakens_negative_dots():
    model = ToyOrthoModel()
    controller = MultiTaskGradientController(
        model, make_args(multitask_ortho_use_ema_gate=False, multitask_ortho_norm_ratio_min=0.01), loss_divisor=1
    )
    run_step(model, controller, conflict_vectors())
    projected_material = controller._ortho_buffers["projected_material"]
    user_reference = controller.actual_weighted_contributions["user"]
    assert torch.dot(projected_material, user_reference) >= -1.0e-6
    assert not torch.equal(selected_grad(controller), sum(conflict_vectors().values()))


def test_rotation_order_is_fixed_and_reproducible():
    controller = MultiTaskGradientController(ToyOrthoModel(), make_args(), loss_divisor=1)
    expected = [
        ("material", "user", "recommendation"),
        ("user", "recommendation", "material"),
        ("recommendation", "material", "user"),
    ]
    for step in range(6):
        controller.current_macro_step = step
        assert controller._ortho_order() == expected[step % 3]


def test_residual_world_and_unselected_grad_are_preserved():
    model = ToyOrthoModel()
    controller = MultiTaskGradientController(
        model, make_args(multitask_ortho_use_ema_gate=False, multitask_ortho_norm_ratio_min=0.01), loss_divisor=1
    )
    vectors = conflict_vectors()
    world = torch.tensor([0.25, -0.5, 0.75])
    run_step(model, controller, vectors, world=world)
    projected_sum = sum(controller._ortho_buffers[f"projected_{task}"] for task in TASKS)
    assert torch.allclose(selected_grad(controller) - projected_sum, world, atol=1.0e-6, rtol=0)
    assert torch.equal(model.unselected.grad, sum(vectors.values()) + world)


def test_norm_ratio_protection_and_finite_parameter_update():
    model = ToyOrthoModel()
    controller = MultiTaskGradientController(
        model,
        make_args(
            multitask_ortho_use_ema_gate=False,
            multitask_ortho_norm_ratio_min=0.95,
            multitask_ortho_norm_ratio_max=1.05,
        ),
        loss_divisor=1,
    )
    run_step(model, controller, conflict_vectors())
    assert 0.95 - 1.0e-6 <= controller.last_tracked_norm_ratio <= 1.05 + 1.0e-6
    assert math.isclose(controller.last_tracked_norm_ratio, 1.05, abs_tol=1.0e-6)
    before = controller.reference_parameters[0].parameter.detach().clone()
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    after = controller.reference_parameters[0].parameter.detach()
    assert torch.isfinite(after).all() and not torch.equal(before, after)


def test_norm_ratio_lower_bound_is_applied():
    model = ToyOrthoModel()
    controller = MultiTaskGradientController(
        model,
        make_args(
            multitask_ortho_use_ema_gate=False,
            multitask_ortho_norm_ratio_min=0.95,
            multitask_ortho_norm_ratio_max=1.05,
        ),
        loss_divisor=1,
    )
    vectors = {
        "material": torch.tensor([10.0, 0.0, 0.0]),
        "user": torch.tensor([-1.0, 1.0, 0.0]),
        "recommendation": torch.tensor([0.0, 0.0, 1.0]),
    }
    run_step(model, controller, vectors)
    assert math.isclose(controller.last_tracked_norm_ratio, 0.95, abs_tol=1.0e-6)


def test_monitor_only_optimizer_step_matches_gradnorm_only():
    baseline = ToyOrthoModel()
    monitored = copy.deepcopy(baseline)
    baseline_controller = MultiTaskGradientController(
        baseline,
        make_args(multitask_gradient_monitor_enabled=True, multitask_gradnorm_update_interval=1, multitask_ortho_enabled=False),
        loss_divisor=1,
    )
    monitored_controller = MultiTaskGradientController(
        monitored,
        make_args(multitask_ortho_monitor_only=True, multitask_ortho_use_ema_gate=False),
        loss_divisor=1,
    )
    run_step(baseline, baseline_controller, conflict_vectors())
    run_step(monitored, monitored_controller, conflict_vectors())
    torch.optim.SGD(baseline.parameters(), lr=0.1).step()
    torch.optim.SGD(monitored.parameters(), lr=0.1).step()
    for left, right in zip(baseline.parameters(), monitored.parameters()):
        assert torch.equal(left, right)


def test_projection_optimizer_step_differs_but_is_bounded():
    baseline = ToyOrthoModel()
    projected = copy.deepcopy(baseline)
    baseline_controller = MultiTaskGradientController(
        baseline,
        make_args(multitask_gradient_monitor_enabled=True, multitask_gradnorm_update_interval=1, multitask_ortho_enabled=False),
        loss_divisor=1,
    )
    projected_controller = MultiTaskGradientController(
        projected, make_args(multitask_ortho_use_ema_gate=False), loss_divisor=1
    )
    run_step(baseline, baseline_controller, conflict_vectors())
    run_step(projected, projected_controller, conflict_vectors())
    torch.optim.SGD(baseline.parameters(), lr=0.1).step()
    torch.optim.SGD(projected.parameters(), lr=0.1).step()
    baseline_selected = baseline_controller.reference_parameters[0].parameter
    projected_selected = projected_controller.reference_parameters[0].parameter
    assert not torch.equal(baseline_selected, projected_selected)
    assert torch.linalg.vector_norm(projected_selected) <= 1.3 * torch.linalg.vector_norm(baseline_selected) + 0.1


def test_checkpoint_preserves_ortho_schedule_ema_and_counters():
    args = make_args(multitask_ortho_interval=2)
    model = ToyOrthoModel()
    controller = MultiTaskGradientController(model, args, loss_divisor=1)
    run_step(model, controller, conflict_vectors())

    # Use explicit state here because model/controller ownership is intentionally checked on restore.
    controller.cosine_ema = {"material_user": -0.4}
    controller.ortho_execution_count = 7
    controller.ortho_pair_counts["material_user"] = 5
    controller.last_removed_ratio = 0.25
    with tempfile.TemporaryDirectory() as directory:
        controller.save_checkpoint(directory, is_main_process=True)
        restored = MultiTaskGradientController(ToyOrthoModel(), args, loss_divisor=1)
        assert restored.load_checkpoint(directory)
    assert restored.cosine_ema == controller.cosine_ema
    assert restored.ortho_execution_count == 7
    assert restored.ortho_pair_counts["material_user"] == 5
    restored.begin_macro_step(7)
    assert not restored.ortho_step_due
    restored.begin_macro_step(8)
    assert restored.ortho_step_due
    assert restored._ortho_order() == ("recommendation", "material", "user")


def test_hook_registration_and_buffers_do_not_pollute_steps():
    model = ToyOrthoModel()
    controller = MultiTaskGradientController(model, make_args(), loss_divisor=1)
    hook_count = len(controller._hook_handles)
    controller.register_hooks()
    assert len(controller._hook_handles) == hook_count
    run_step(model, controller, conflict_vectors(), step=0)
    assert controller.current_task is None
    controller.begin_macro_step(1)
    assert all(buffer.count_nonzero() == 0 for buffer in controller.flat_buffers.values())


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} multitask Ortho tests passed.")
