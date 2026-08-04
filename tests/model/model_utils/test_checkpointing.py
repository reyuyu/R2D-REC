# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from llamafactory.extras.misc import get_current_device
from llamafactory.model.model_utils.checkpointing import configure_selective_gradient_checkpointing
from llamafactory.train.test_utils import load_train_model


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_target": "all",
    "dataset": "llamafactory/tiny-supervised-dataset",
    "dataset_dir": "ONLINE",
    "template": "llama3",
    "cutoff_len": 1024,
    "output_dir": "dummy_dir",
    "overwrite_output_dir": True,
    "fp16": True,
}


@pytest.mark.parametrize("disable_gradient_checkpointing", [False, True])
def test_vanilla_checkpointing(disable_gradient_checkpointing: bool):
    model = load_train_model(disable_gradient_checkpointing=disable_gradient_checkpointing, **TRAIN_ARGS)
    for module in filter(lambda m: hasattr(m, "gradient_checkpointing"), model.modules()):
        assert getattr(module, "gradient_checkpointing") != disable_gradient_checkpointing


def test_unsloth_gradient_checkpointing():
    model = load_train_model(use_unsloth_gc=True, **TRAIN_ARGS)
    for module in filter(lambda m: hasattr(m, "gradient_checkpointing"), model.modules()):
        assert module._gradient_checkpointing_func.__self__.__name__ == "UnslothGradientCheckpointing"


def test_selective_gradient_checkpointing_keeps_last_half_of_decoder_layers():
    layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(4)])
    for layer in layers:
        layer.gradient_checkpointing = True

    model = SimpleNamespace(base_model_prefix="model", model=SimpleNamespace(layers=layers))
    checkpointed, total = configure_selective_gradient_checkpointing(model, 0.5)
    assert (checkpointed, total) == (2, 4)
    assert [layer.gradient_checkpointing for layer in layers] == [False, False, True, True]


def test_selective_gradient_checkpointing_matches_full_checkpoint_loss_and_gradients():
    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        attention_dropout=0.0,
        use_cache=False,
    )
    full_model = Qwen3ForCausalLM(config)
    half_model = copy.deepcopy(full_model)
    for model in (full_model, half_model):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        model.enable_input_require_grads()
        model.train()

    configure_selective_gradient_checkpointing(half_model, 0.5)
    input_ids = torch.randint(0, config.vocab_size, (1, 12))
    labels = input_ids.clone()
    full_loss = full_model(input_ids=input_ids, labels=labels).loss
    half_loss = half_model(input_ids=input_ids, labels=labels).loss
    full_loss.backward()
    half_loss.backward()
    torch.testing.assert_close(full_loss, half_loss, atol=1e-7, rtol=1e-7)
    for (full_name, full_param), (half_name, half_param) in zip(
        full_model.named_parameters(), half_model.named_parameters()
    ):
        assert full_name == half_name
        if full_param.grad is not None or half_param.grad is not None:
            torch.testing.assert_close(full_param.grad, half_param.grad, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("layer_ratio", [-0.01, 1.01])
def test_gradient_checkpointing_layer_ratio_validation(layer_ratio: float):
    with pytest.raises(ValueError, match="gradient_checkpointing_layer_ratio"):
        load_train_model(gradient_checkpointing_layer_ratio=layer_ratio, **TRAIN_ARGS)


def test_upcast_layernorm():
    model = load_train_model(upcast_layernorm=True, **TRAIN_ARGS)
    for name, param in model.named_parameters():
        if param.ndim == 1 and "norm" in name:
            assert param.dtype == torch.float32


def test_upcast_lmhead_output():
    model = load_train_model(upcast_lmhead_output=True, **TRAIN_ARGS)
    inputs = torch.randn((1, 16), dtype=torch.float16, device=get_current_device())
    outputs: torch.Tensor = model.get_output_embeddings()(inputs)
    assert outputs.dtype == torch.float32
