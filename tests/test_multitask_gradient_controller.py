import copy
import math
import tempfile
from contextlib import nullcontext
from types import SimpleNamespace

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from llamafactory.train.sft.multitask_gradient_controller import MultiTaskGradientController
from llamafactory.train.sft.trainer import MultiTaskMacroSeq2SeqTrainer


TASKS = ("material", "user", "recommendation")


class WeightHolder(nn.Module):
    def __init__(self, width: int = 2):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))


class LoraProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": WeightHolder()})
        self.lora_B = nn.ModuleDict({"default": WeightHolder()})


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = LoraProjection()
        self.self_attn.v_proj = LoraProjection()
        self.self_attn.o_proj = LoraProjection()
        self.mlp = nn.Module()
        self.mlp.down_proj = LoraProjection()
        self.mlp.up_proj = LoraProjection()


class ToyLoraModel(nn.Module):
    def __init__(self, layers: int = 3):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([ToyBlock() for _ in range(layers)])

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        outputs = []
        for block in self.model.layers:
            for projection in (
                block.self_attn.q_proj,
                block.self_attn.v_proj,
                block.self_attn.o_proj,
                block.mlp.down_proj,
                block.mlp.up_proj,
            ):
                outputs.append(torch.dot(projection.lora_B["default"].weight, feature))
        return torch.stack(outputs).sum()


def make_args(**overrides):
    values = {
        "multitask_gradnorm_tasks": list(TASKS),
        "multitask_gradient_monitor_enabled": True,
        "multitask_gradnorm_enabled": True,
        "multitask_world_loss_weight": 1.0,
        "multitask_gradnorm_warmup_steps": 2,
        "multitask_gradnorm_update_interval": 1,
        "multitask_gradnorm_alpha": 0.5,
        "multitask_gradnorm_update_rate": 1.0,
        "multitask_gradnorm_loss_ema_beta": 0.5,
        "multitask_gradnorm_grad_ema_beta": 0.5,
        "multitask_gradnorm_weight_min": 0.5,
        "multitask_gradnorm_weight_max": 2.0,
        "multitask_gradnorm_step_ratio_min": 0.9,
        "multitask_gradnorm_step_ratio_max": 1.1,
        "multitask_grad_reference_last_n_layers": 2,
        "multitask_grad_reference_modules": ["q_proj", "v_proj", "o_proj", "down_proj"],
        "multitask_grad_reference_lora_matrix": "B",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def flatten_reference_gradients(controller):
    return torch.cat([reference.parameter.grad.detach().reshape(-1) for reference in controller.reference_parameters])


def assert_model_gradients_equal(left, right):
    for left_parameter, right_parameter in zip(left.parameters(), right.parameters()):
        assert (left_parameter.grad is None) == (right_parameter.grad is None)
        if left_parameter.grad is not None:
            assert torch.equal(left_parameter.grad, right_parameter.grad)


def backward_tasks(model, controller, task_features, divisor):
    for task, features in task_features.items():
        for feature in features:
            controller.set_current_task(task)
            try:
                model(feature).mul(controller.task_weight(task) / divisor).backward()
            finally:
                controller.clear_current_task()


def test_total_switch_off_matches_original_loss_and_gradient():
    original = ToyLoraModel()
    switched_off = copy.deepcopy(original)
    features = [torch.tensor([1.0, -2.0]), torch.tensor([0.5, 3.0])]
    original_loss = sum(original(feature) / len(features) for feature in features)
    original_loss.backward()

    fake_trainer = SimpleNamespace(
        args=SimpleNamespace(device=torch.device("cpu"), n_gpu=1),
        accelerator=SimpleNamespace(backward=lambda loss: loss.backward()),
        gradient_controller=None,
        _MODEL_INPUT_KEYS=MultiTaskMacroSeq2SeqTrainer._MODEL_INPUT_KEYS,
        _loss_scale_denominator=len(features),
        _prepare_inputs=lambda inputs: inputs,
        compute_loss_context_manager=lambda: nullcontext(),
        compute_loss=lambda model, inputs: model(inputs["input_ids"]),
    )
    microbatches = [
        {
            "input_ids": feature,
            "supervised_token_count": 1,
            "num_segments": 1,
        }
        for feature in features
    ]
    switched_off_loss, _ = MultiTaskMacroSeq2SeqTrainer.compute_task_microbatches(
        fake_trainer, switched_off, "material", microbatches
    )
    assert torch.equal(original_loss.detach(), switched_off_loss)
    assert_model_gradients_equal(original, switched_off)
    assert fake_trainer.gradient_controller is None


def test_reference_selection_is_last_layers_expected_modules_and_lora_b_only():
    model = ToyLoraModel(layers=4)
    references = MultiTaskGradientController.select_reference_parameters(
        model, last_n_layers=2, module_names=("q_proj", "v_proj", "o_proj", "down_proj"), lora_matrix="B"
    )
    assert {reference.layer for reference in references} == {2, 3}
    assert len(references) == 8
    assert all("lora_B" in reference.name for reference in references)
    assert all("up_proj" not in reference.name for reference in references)
    assert all(reference.parameter.requires_grad for reference in references)


def test_reference_selection_error_lists_lora_candidates():
    model = ToyLoraModel(layers=1)
    try:
        MultiTaskGradientController.select_reference_parameters(
            model, last_n_layers=1, module_names=("gate_proj",), lora_matrix="B"
        )
    except ValueError as error:
        message = str(error)
        assert "Candidate LoRA parameters" in message
        assert "q_proj.lora_B.default.weight" in message
    else:
        raise AssertionError("Expected a clear reference-selection error.")


def test_monitor_only_keeps_weights_and_model_gradient_unchanged():
    baseline = ToyLoraModel()
    monitored = copy.deepcopy(baseline)
    args = make_args(multitask_gradnorm_enabled=False)
    controller = MultiTaskGradientController(monitored, args, loss_divisor=3)
    task_features = {
        "material": [torch.tensor([1.0, 0.0])],
        "user": [torch.tensor([0.0, 1.0])],
        "recommendation": [torch.tensor([1.0, 1.0])],
    }
    for features in task_features.values():
        baseline(features[0]).div(3).backward()
    controller.begin_macro_step(1)
    backward_tasks(monitored, controller, task_features, divisor=3)
    controller.finish_macro_step(dict.fromkeys(TASKS, 1.0), dict.fromkeys(TASKS, 1))
    assert_model_gradients_equal(baseline, monitored)
    assert controller.task_weights == dict.fromkeys(TASKS, 1.0)
    assert controller.pending_weights is None


def test_trainer_applies_task_weight_before_existing_loss_divisor():
    model = ToyLoraModel()
    controller = MultiTaskGradientController(
        model,
        make_args(multitask_gradient_monitor_enabled=False, multitask_gradnorm_enabled=False),
        loss_divisor=4,
    )
    controller.task_weights["material"] = 1.5
    fake_trainer = SimpleNamespace(
        args=SimpleNamespace(device=torch.device("cpu"), n_gpu=1),
        accelerator=SimpleNamespace(backward=lambda loss: loss.backward()),
        gradient_controller=controller,
        _MODEL_INPUT_KEYS=MultiTaskMacroSeq2SeqTrainer._MODEL_INPUT_KEYS,
        _loss_scale_denominator=4,
        _prepare_inputs=lambda inputs: inputs,
        compute_loss_context_manager=lambda: nullcontext(),
        compute_loss=lambda current_model, inputs: current_model(inputs["input_ids"]),
    )
    feature = torch.tensor([2.0, -1.0])
    microbatch = {"input_ids": feature, "supervised_token_count": 1, "num_segments": 1}
    MultiTaskMacroSeq2SeqTrainer.compute_task_microbatches(
        fake_trainer, model, "material", [microbatch], collect_metrics=True
    )
    expected = feature.repeat(len(controller.reference_parameters)) * (1.5 / 4)
    assert torch.equal(flatten_reference_gradients(controller), expected)


def test_hook_contributions_sum_to_final_gradient_and_restore_raw_task_means():
    model = ToyLoraModel()
    controller = MultiTaskGradientController(model, make_args(multitask_gradnorm_enabled=False), loss_divisor=4)
    task_features = {
        "material": [torch.tensor([1.0, 0.0]), torch.tensor([3.0, 0.0])],
        "user": [torch.tensor([0.0, 2.0])],
        "recommendation": [torch.tensor([1.0, 1.0]), torch.tensor([-1.0, 3.0])],
    }
    controller.begin_macro_step(1)
    backward_tasks(model, controller, task_features, divisor=4)
    counts = {task: len(features) for task, features in task_features.items()}
    controller.finish_macro_step({task: float(count) for task, count in counts.items()}, counts)

    captured_sum = sum(controller.actual_weighted_contributions.values())
    assert torch.allclose(captured_sum, flatten_reference_gradients(controller), atol=0, rtol=0)
    for task, features in task_features.items():
        reference_model = ToyLoraModel()
        reference_controller = MultiTaskGradientController(
            reference_model, make_args(multitask_gradnorm_enabled=False), loss_divisor=1
        )
        torch.stack([reference_model(feature) for feature in features]).mean().backward()
        assert torch.allclose(
            controller.raw_average_gradients[task],
            flatten_reference_gradients(reference_controller),
            atol=1.0e-7,
            rtol=1.0e-7,
        )


def test_cosines_use_raw_average_gradients():
    model = ToyLoraModel()
    controller = MultiTaskGradientController(model, make_args(multitask_gradnorm_enabled=False), loss_divisor=3)
    features = {
        "material": [torch.tensor([1.0, 0.0])],
        "user": [torch.tensor([0.0, 1.0])],
        "recommendation": [torch.tensor([1.0, 1.0])],
    }
    controller.begin_macro_step(1)
    backward_tasks(model, controller, features, divisor=3)
    controller.finish_macro_step(dict.fromkeys(TASKS, 1.0), dict.fromkeys(TASKS, 1))
    assert abs(controller.last_cosines["material_user"]) < 1.0e-7
    assert math.isclose(controller.last_cosines["material_recommendation"], 1 / math.sqrt(2), rel_tol=1.0e-6)
    assert math.isclose(controller.last_cosines["user_recommendation"], 1 / math.sqrt(2), rel_tol=1.0e-6)


def seed_raw_norms(controller, norms):
    for task, norm in norms.items():
        vector = torch.zeros_like(controller.flat_buffers[task])
        vector[0] = norm * controller.task_weights[task] / controller.loss_divisor
        controller.flat_buffers[task].copy_(vector)


def test_ema_warmup_baseline_lagged_update_and_bounds():
    controller = MultiTaskGradientController(ToyLoraModel(), make_args(), loss_divisor=1)
    counts = dict.fromkeys(TASKS, 1)

    controller.begin_macro_step(1)
    seed_raw_norms(controller, dict.fromkeys(TASKS, 1.0))
    controller.finish_macro_step(dict.fromkeys(TASKS, 2.0), counts)
    assert controller.loss_ema == dict.fromkeys(TASKS, 2.0)
    assert not controller.baseline_initialized
    assert controller.task_weights == dict.fromkeys(TASKS, 1.0)

    controller.begin_macro_step(2)
    seed_raw_norms(controller, dict.fromkeys(TASKS, 1.0))
    controller.finish_macro_step(dict.fromkeys(TASKS, 4.0), counts)
    assert controller.loss_ema == dict.fromkeys(TASKS, 3.0)
    assert controller.initial_loss_baseline == dict.fromkeys(TASKS, 3.0)
    assert controller.pending_weights is None

    controller.begin_macro_step(3)
    seed_raw_norms(controller, {"material": 10.0, "user": 1.0, "recommendation": 0.1})
    frozen_baseline = dict(controller.initial_loss_baseline)
    controller.finish_macro_step({"material": 12.0, "user": 3.0, "recommendation": 0.75}, counts)
    assert controller.task_weights == dict.fromkeys(TASKS, 1.0)
    assert controller.pending_weights is not None
    assert math.isclose(sum(controller.pending_weights.values()), 3.0, abs_tol=1.0e-10)
    assert all(0.9 <= value <= 1.1 for value in controller.pending_weights.values())
    assert controller.weight_update_active

    pending = dict(controller.pending_weights)
    controller.begin_macro_step(4)
    assert controller.task_weights == pending
    seed_raw_norms(controller, dict.fromkeys(TASKS, 1.0))
    controller.finish_macro_step(dict.fromkeys(TASKS, 100.0), counts)
    assert controller.initial_loss_baseline == frozen_baseline


def test_checkpoint_round_trip_preserves_pending_and_ema_state():
    args = make_args()
    controller = MultiTaskGradientController(ToyLoraModel(), args, loss_divisor=1)
    controller.task_weights = {"material": 0.9, "user": 1.0, "recommendation": 1.1}
    controller.pending_weights = {"material": 0.95, "user": 1.02, "recommendation": 1.03}
    controller.loss_ema = {task: float(index + 1) for index, task in enumerate(TASKS)}
    controller.grad_norm_ema = {task: float(index + 4) for index, task in enumerate(TASKS)}
    controller.initial_loss_baseline = dict.fromkeys(TASKS, 1.0)
    controller.baseline_initialized = True
    controller.measurement_count = 7
    controller.weight_update_count = 3
    controller.last_cosines = {"material_user": -0.25}
    with tempfile.TemporaryDirectory() as directory:
        controller.save_checkpoint(directory, is_main_process=True)
        restored = MultiTaskGradientController(ToyLoraModel(), args, loss_divisor=1)
        assert restored.load_checkpoint(directory)
    assert restored.state_dict() == controller.state_dict()
    restored.begin_macro_step(8)
    assert restored.task_weights == {"material": 0.95, "user": 1.02, "recommendation": 1.03}


def test_hooks_are_idempotent_clear_task_and_do_not_pollute_across_steps():
    model = ToyLoraModel()
    controller = MultiTaskGradientController(model, make_args(multitask_gradnorm_enabled=False), loss_divisor=1)
    handle_count = len(controller._hook_handles)
    controller.register_hooks()
    assert len(controller._hook_handles) == handle_count
    controller.begin_macro_step(1)
    controller.set_current_task("material")
    model(torch.tensor([1.0, 0.0])).backward()
    controller.finish_macro_step({"material": 1.0}, {"material": 1})
    assert controller.current_task is None
    assert controller.flat_buffers["material"].abs().sum() > 0
    controller.begin_macro_step(2)
    assert all(buffer.count_nonzero() == 0 for buffer in controller.flat_buffers.values())
    controller.finish_macro_step({}, {})
    controller.begin_macro_step(3)
    assert all(buffer.count_nonzero() == 0 for buffer in controller.flat_buffers.values())


def test_non_measurement_step_hook_is_immediate_noop():
    model = ToyLoraModel()
    controller = MultiTaskGradientController(
        model,
        make_args(multitask_gradnorm_enabled=False, multitask_gradnorm_update_interval=2),
        loss_divisor=1,
    )
    controller.begin_macro_step(1)
    before = controller.flat_buffers["material"].clone()
    controller.set_current_task("material")
    model(torch.tensor([2.0, 3.0])).backward()
    assert controller.current_task is None
    assert torch.equal(controller.flat_buffers["material"], before)
    controller.finish_macro_step({"material": 1.0}, {"material": 1})


def test_metrics_keep_weights_norms_loss_ratios_and_conflict_cosines_only():
    controller = MultiTaskGradientController(ToyLoraModel(), make_args(), loss_divisor=1)
    controller.baseline_initialized = True
    controller.loss_ema = dict.fromkeys(TASKS, 2.0)
    controller.initial_loss_baseline = dict.fromkeys(TASKS, 1.0)
    controller.last_raw_norms = dict.fromkeys(TASKS, 3.0)
    controller.last_effective_norms = dict.fromkeys(TASKS, 3.0)
    controller.last_cosines = {
        "material_user": -0.2,
        "material_recommendation": 0.1,
        "user_recommendation": -0.3,
    }
    metrics = controller.metrics()
    assert all(f"a_gn_w_{task}" in metrics for task in TASKS)
    assert all(f"b_gn_loss_ratio_{task}" in metrics for task in TASKS)
    assert all(f"c_gn_raw_norm_{task}" in metrics for task in TASKS)
    assert all(f"c_gn_effective_norm_{task}" in metrics for task in TASKS)
    assert all(f"d_gn_cos_{pair}" in metrics for pair in controller.last_cosines)
    assert not any("loss_ema" in name or name.startswith("f_gn_") for name in metrics)


def test_tiny_random_qwen3_uses_real_peft_lora_b_gradient_path():
    torch.manual_seed(7)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        use_cache=False,
    )
    model = get_peft_model(
        Qwen3ForCausalLM(config),
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj", "o_proj", "down_proj"],
        ),
    ).to(device)
    model.eval()
    controller = MultiTaskGradientController(model, make_args(multitask_gradnorm_enabled=False), loss_divisor=3)
    assert len(controller.reference_parameters) == 8
    controller.begin_macro_step(1)
    losses = {}
    for task_index, task in enumerate(TASKS):
        input_ids = torch.tensor([[1, 3 + task_index, 8, 13, 21]], device=device)
        labels = input_ids.clone()
        controller.set_current_task(task)
        try:
            loss = model(input_ids=input_ids, labels=labels).loss
            loss.div(3).backward()
        finally:
            controller.clear_current_task()
        losses[task] = float(loss.detach())
    controller.finish_macro_step(losses, dict.fromkeys(TASKS, 1))
    assert torch.allclose(
        sum(controller.actual_weighted_contributions.values()),
        flatten_reference_gradients(controller),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} multitask gradient controller tests passed.")
