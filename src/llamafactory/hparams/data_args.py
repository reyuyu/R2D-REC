# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class DataArguments:
    r"""Arguments pertaining to what data we are going to input our model for training and evaluation."""

    template: str | None = field(
        default=None,
        metadata={"help": "Which template to use for constructing prompts in training and inference."},
    )
    dataset: str | None = field(
        default=None,
        metadata={"help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."},
    )
    eval_dataset: str | None = field(
        default=None,
        metadata={"help": "The name of dataset(s) to use for evaluation. Use commas to separate multiple datasets."},
    )
    dataset_dir: str = field(
        default="data",
        metadata={"help": "Path to the folder containing the datasets."},
    )
    media_dir: str | None = field(
        default=None,
        metadata={"help": "Path to the folder containing the images, videos or audios. Defaults to `dataset_dir`."},
    )
    cutoff_len: int = field(
        default=2048,
        metadata={"help": "The cutoff length of the tokenized inputs in the dataset."},
    )
    train_on_prompt: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )
    mask_history: bool = field(
        default=False,
        metadata={"help": "Whether or not to mask the history and train on the last turn only."},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Enable dataset streaming."},
    )
    buffer_size: int = field(
        default=16384,
        metadata={"help": "Size of the buffer to randomly sample examples from in dataset streaming."},
    )
    mix_strategy: Literal["concat", "interleave_under", "interleave_over", "interleave_once"] = field(
        default="concat",
        metadata={
            "help": "Strategy to use in dataset mixing (concat/interleave) (undersampling/oversampling/sampling w.o. replacement)."
        },
    )
    interleave_probs: str | None = field(
        default=None,
        metadata={"help": "Probabilities to sample data from datasets. Use commas to separate multiple datasets."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    preprocessing_batch_size: int = field(
        default=1000,
        metadata={"help": "The number of examples in one group in pre-processing."},
    )
    preprocessing_num_workers: int | None = field(
        default=None,
        metadata={"help": "The number of processes to use for the pre-processing."},
    )
    max_samples: int | None = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples for each dataset."},
    )
    eval_num_beams: int | None = field(
        default=None,
        metadata={"help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`"},
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "Whether or not to ignore the tokens corresponding to the pad label in loss computation."},
    )
    val_size: float = field(
        default=0.0,
        metadata={"help": "Size of the validation set, should be an integer or a float in range `[0,1)`."},
    )
    eval_on_each_dataset: bool = field(
        default=False,
        metadata={"help": "Whether or not to evaluate on each dataset separately."},
    )
    packing: bool | None = field(
        default=None,
        metadata={"help": "Enable sequences packing in training. Will automatically enable in pre-training."},
    )
    neat_packing: bool = field(
        default=False,
        metadata={"help": "Enable sequence packing without cross-attention."},
    )
    tool_format: str | None = field(
        default=None,
        metadata={"help": "Tool format to use for constructing function calling examples."},
    )
    default_system: str | None = field(
        default=None,
        metadata={"help": "Override the default system message in the template."},
    )
    enable_thinking: bool | None = field(
        default=True,
        metadata={"help": "Whether or not to enable thinking mode for reasoning models."},
    )
    preserve_thinking: bool = field(
        default=False,
        metadata={"help": "Whether or not to preserve thinking content in historical turns for reasoning models."},
    )
    tokenized_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to save or load the tokenized datasets. "
                "If tokenized_path not exists, it will save the tokenized datasets. "
                "If tokenized_path exists, it will load the tokenized datasets."
            )
        },
    )
    data_shared_file_system: bool = field(
        default=False,
        metadata={"help": "Whether or not to use a shared file system for the datasets."},
    )
    multitask_macro_training: bool = field(
        default=False,
        metadata={"help": "Enable the optional four-task macro-step SFT data pipeline."},
    )
    multitask_task_layout: str = field(
        default="legacy",
        metadata={
            "help": (
                "Multitask task layout. legacy keeps material/user/recommendation/world with user "
                "as one task; user_split_no_world uses material/user_action/user_chain/recommendation "
                "and drops world (Experiment E)."
            )
        },
    )
    multitask_microbatch_allocation: dict[str, int] = field(
        default_factory=lambda: {"material": 2, "user": 2, "recommendation": 3, "world": 1},
        metadata={"help": "Task microbatches per macro-step when multitask macro training is enabled."},
    )
    multitask_global_microbatch_ddp: bool = field(
        default=False,
        metadata={
            "help": (
                "For exactly two DDP ranks, split one global eight-microbatch macro-step across the ranks "
                "instead of running all eight microbatches on every rank."
            )
        },
    )
    multitask_supercycle_mode: Literal["fixed", "balanced_40"] = field(
        default="fixed",
        metadata={
            "help": (
                "Macro allocation scheduler for multitask mode. `balanced_40` uses the configured "
                "40-macro-step task mix and requires global two-rank microbatch DDP."
            )
        },
    )
    multitask_train_dataset_suffix: str = field(
        default="",
        metadata={
            "help": "Optional suffix resolved against every registered multitask subdataset, e.g. _train98."
        },
    )
    multitask_dataset_version: str = field(
        default="raw",
        metadata={
            "help": (
                "Version selected for all macro-training datasets. raw keeps the legacy names; other versions "
                "resolve as <logical_dataset>_<version><multitask_train_dataset_suffix>."
            )
        },
    )
    multitask_dataset_version_overrides: dict[str, str] = field(
        default_factory=dict,
        metadata={
            "help": "Optional per-logical-dataset version overrides for macro training, e.g. {onereason_user_action_nocot: v2}."
        },
    )
    multitask_monitoring: bool = field(
        default=False,
        metadata={"help": "Write task losses and packing statistics to output_dir/monitor/metrics.jsonl."},
    )
    multitask_max_pack_length: int | None = field(
        default=None,
        metadata={"help": "Task-wise pack length. Defaults to cutoff_len."},
    )
    multitask_max_segments_per_pack: int | None = field(
        default=None,
        metadata={"help": "Optional upper bound on segments in one multitask pack."},
    )
    multitask_gradient_control_enabled: bool = field(
        default=False,
        metadata={"help": "Enable the optional multitask gradient controller."},
    )
    multitask_gradient_monitor_enabled: bool = field(
        default=False,
        metadata={"help": "Capture reference gradients and report task norms/cosines."},
    )
    multitask_gradnorm_enabled: bool = field(
        default=False,
        metadata={"help": "Apply lagged GradNorm-lite task loss weights."},
    )
    multitask_gradnorm_tasks: list[str] = field(
        default_factory=lambda: ["material", "user", "recommendation"],
        metadata={"help": "Ordered tasks participating in GradNorm-lite."},
    )
    multitask_world_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Fixed world-task loss weight; world never participates in GradNorm-lite."},
    )
    multitask_gradnorm_warmup_steps: int = field(default=200)
    multitask_gradnorm_update_interval: int = field(default=10)
    multitask_gradnorm_alpha: float = field(default=0.5)
    multitask_gradnorm_update_rate: float = field(default=0.10)
    multitask_gradnorm_loss_ema_beta: float = field(default=0.90)
    multitask_gradnorm_grad_ema_beta: float = field(default=0.90)
    multitask_gradnorm_weight_min: float = field(default=0.5)
    multitask_gradnorm_weight_max: float = field(default=2.0)
    multitask_gradnorm_step_ratio_min: float = field(default=0.9)
    multitask_gradnorm_step_ratio_max: float = field(default=1.1)
    multitask_grad_reference_last_n_layers: int = field(default=4)
    multitask_grad_reference_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "o_proj", "down_proj"]
    )
    multitask_grad_reference_lora_matrix: str = field(default="B")
    multitask_ortho_enabled: bool = field(
        default=False,
        metadata={"help": "Enable local PCGrad-style projection on the selected LoRA reference gradients."},
    )
    multitask_ortho_monitor_only: bool = field(
        default=False,
        metadata={"help": "Compute Ortho projection metrics without writing projected gradients back."},
    )
    multitask_ortho_start_step: int = field(default=600)
    multitask_ortho_interval: int = field(default=2)
    multitask_ortho_current_cosine_threshold: float = field(default=-0.05)
    multitask_ortho_ema_cosine_threshold: float = field(default=0.0)
    multitask_ortho_cosine_ema_beta: float = field(default=0.90)
    multitask_ortho_use_ema_gate: bool = field(default=True)
    multitask_ortho_norm_ratio_min: float = field(default=0.7)
    multitask_ortho_norm_ratio_max: float = field(default=1.3)
    multitask_ortho_rotate_order: bool = field(default=True)
    sid_token_weighting_enabled: bool = field(
        default=False,
        metadata={"help": "Use normalized weighted SFT CE for supervised <s_a_*>, <s_b_*>, and <s_c_*> tokens."},
    )
    sid_token_weight: float = field(default=8.0)
    sid_text_weight: float = field(default=1.0)
    user_action_aux_enabled: bool = field(
        default=False,
        metadata={"help": "Enable the Action Select history and length auxiliary objective."},
    )
    user_action_history_trie_enabled: bool = field(default=True)
    user_action_history_trie_weight: float = field(default=0.06)
    user_action_length_guard_enabled: bool = field(default=True)
    user_action_continue_domain_extra: float = field(default=0.75)
    user_action_continue_separator_extra: float = field(default=0.20)
    user_action_no_early_stop_weight: float = field(default=0.02)
    user_action_stop_domain_weight: float = field(default=0.05)
    user_action_stop_tail_extra: float = field(default=1.0)
    user_action_max_stop_tail_positions: int = field(default=4)
    user_action_aux_cap_ratio: float = field(default=0.08)
    user_action_aux_split_cap_enabled: bool = field(
        default=False,
        metadata={"help": "Cap Action Trie and length objectives independently before the total safety cap."},
    )
    user_action_trie_cap_ratio: float = field(default=0.06)
    user_action_length_cap_ratio: float = field(default=0.02)
    user_action_aux_warmup_steps: int = field(default=100)
    user_action_aux_vectorized_enabled: bool = field(
        default=False,
        metadata={"help": "Use the mathematically equivalent batched Action Select loss backend."},
    )
    user_action_aux_full_vocab_chunk_size: int = field(
        default=64,
        metadata={"help": "Maximum Action target rows per full-vocabulary reduction chunk."},
    )
    user_action_topk_illegal_enabled: bool = field(
        default=False,
        metadata={"help": "Penalize the highest-scoring full-vocabulary illegal tokens at Action SID slots."},
    )
    user_action_topk_illegal_k: int = field(default=5)
    user_action_topk_illegal_margin: float = field(default=0.0)
    user_action_topk_illegal_weight: float = field(default=0.02)
    user_action_topk_illegal_cap_ratio: float = field(default=0.02)

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.dataset = split_arg(self.dataset)
        self.eval_dataset = split_arg(self.eval_dataset)
        self.multitask_gradnorm_tasks = split_arg(self.multitask_gradnorm_tasks)
        self.multitask_grad_reference_modules = split_arg(self.multitask_grad_reference_modules)

        if not isinstance(self.multitask_dataset_version, str) or not self.multitask_dataset_version:
            raise ValueError("multitask_dataset_version must be a non-empty string.")
        if not isinstance(self.multitask_dataset_version_overrides, dict):
            raise ValueError("multitask_dataset_version_overrides must be a mapping of dataset names to version names.")
        if any(
            not isinstance(key, str) or not isinstance(value, str) or not value
            for key, value in self.multitask_dataset_version_overrides.items()
        ):
            raise ValueError("multitask_dataset_version_overrides must contain non-empty string dataset names and versions.")

        if self.media_dir is None:
            self.media_dir = self.dataset_dir

        if self.dataset is None and self.val_size > 1e-6:
            raise ValueError("Cannot specify `val_size` if `dataset` is None.")

        if self.eval_dataset is not None and self.val_size > 1e-6:
            raise ValueError("Cannot specify `val_size` if `eval_dataset` is not None.")

        if self.interleave_probs is not None:
            if self.mix_strategy == "concat":
                raise ValueError("`interleave_probs` is only valid for interleaved mixing.")

            self.interleave_probs = list(map(float, split_arg(self.interleave_probs)))
            if self.dataset is not None and len(self.dataset) != len(self.interleave_probs):
                raise ValueError("The length of dataset and interleave probs should be identical.")

            if self.eval_dataset is not None and len(self.eval_dataset) != len(self.interleave_probs):
                raise ValueError("The length of eval dataset and interleave probs should be identical.")

        if self.streaming and self.val_size > 1e-6 and self.val_size < 1:
            raise ValueError("Streaming mode should have an integer val size.")

        if self.streaming and self.max_samples is not None:
            raise ValueError("`max_samples` is incompatible with `streaming`.")

        if self.mask_history and self.train_on_prompt:
            raise ValueError("`mask_history` is incompatible with `train_on_prompt`.")

        if self.neat_packing:
            self.packing = True

        if self.multitask_macro_training:
            if self.multitask_task_layout not in {"legacy", "user_split_no_world"}:
                raise ValueError("multitask_task_layout must be either 'legacy' or 'user_split_no_world'.")
            if self.multitask_task_layout == "user_split_no_world":
                required_tasks = {"material", "user_action", "user_chain", "recommendation"}
            else:
                required_tasks = {"material", "user", "recommendation", "world"}
            if set(self.multitask_microbatch_allocation) != required_tasks:
                raise ValueError(
                    f"multitask_microbatch_allocation must contain {sorted(required_tasks)} for layout {self.multitask_task_layout!r}."
                )
            if any(value <= 0 for value in self.multitask_microbatch_allocation.values()):
                raise ValueError("multitask_microbatch_allocation values must be positive.")
            if sum(self.multitask_microbatch_allocation.values()) != 8:
                raise ValueError("multitask_microbatch_allocation must sum to 8.")
            if self.packing or self.neat_packing:
                raise ValueError("multitask_macro_training owns packing; set packing=false and neat_packing=false.")
            if self.multitask_max_pack_length is None:
                self.multitask_max_pack_length = self.cutoff_len
            if self.multitask_max_pack_length <= 0:
                raise ValueError("multitask_max_pack_length must be positive.")
            if self.multitask_supercycle_mode not in {"fixed", "balanced_40"}:
                raise ValueError("multitask_supercycle_mode must be either `fixed` or `balanced_40`.")
            if self.multitask_supercycle_mode == "balanced_40" and not self.multitask_global_microbatch_ddp:
                raise ValueError("balanced_40 super-cycle requires multitask_global_microbatch_ddp=true.")

        if self.multitask_gradient_control_enabled and not self.multitask_macro_training:
            raise ValueError("multitask_gradient_control_enabled requires multitask_macro_training=true.")
        if self.multitask_gradient_monitor_enabled and not self.multitask_gradient_control_enabled:
            raise ValueError("multitask_gradient_monitor_enabled requires multitask_gradient_control_enabled=true.")
        if self.multitask_gradnorm_enabled and not self.multitask_gradient_monitor_enabled:
            raise ValueError("multitask_gradnorm_enabled requires multitask_gradient_monitor_enabled=true.")
        if self.multitask_task_layout == "user_split_no_world":
            expected_gradnorm_tasks = {"material", "user_action", "user_chain", "recommendation"}
        else:
            expected_gradnorm_tasks = {"material", "user", "recommendation"}
        if set(self.multitask_gradnorm_tasks) != expected_gradnorm_tasks:
            raise ValueError(
                f"multitask_gradnorm_tasks must contain {sorted(expected_gradnorm_tasks)} exactly once for layout {self.multitask_task_layout!r}."
            )
        if len(self.multitask_gradnorm_tasks) != len(expected_gradnorm_tasks):
            raise ValueError("multitask_gradnorm_tasks cannot contain duplicates.")
        if self.multitask_world_loss_weight <= 0:
            raise ValueError("multitask_world_loss_weight must be positive.")
        if self.multitask_gradnorm_warmup_steps < 0:
            raise ValueError("multitask_gradnorm_warmup_steps cannot be negative.")
        if self.multitask_gradnorm_update_interval <= 0:
            raise ValueError("multitask_gradnorm_update_interval must be positive.")
        if self.multitask_gradnorm_alpha < 0 or self.multitask_gradnorm_update_rate < 0:
            raise ValueError("GradNorm alpha and update rate cannot be negative.")
        if not 0 <= self.multitask_gradnorm_loss_ema_beta < 1:
            raise ValueError("multitask_gradnorm_loss_ema_beta must be in [0, 1).")
        if not 0 <= self.multitask_gradnorm_grad_ema_beta < 1:
            raise ValueError("multitask_gradnorm_grad_ema_beta must be in [0, 1).")
        if not 0 < self.multitask_gradnorm_weight_min <= 1 <= self.multitask_gradnorm_weight_max:
            raise ValueError("GradNorm weight bounds must be positive and include 1.0.")
        if not 0 < self.multitask_gradnorm_step_ratio_min <= 1 <= self.multitask_gradnorm_step_ratio_max:
            raise ValueError("GradNorm step-ratio bounds must be positive and include 1.0.")
        if self.multitask_grad_reference_last_n_layers <= 0:
            raise ValueError("multitask_grad_reference_last_n_layers must be positive.")
        if not self.multitask_grad_reference_modules:
            raise ValueError("multitask_grad_reference_modules cannot be empty.")
        if self.multitask_grad_reference_lora_matrix.upper() not in {"A", "B"}:
            raise ValueError("multitask_grad_reference_lora_matrix must be A or B.")
        if self.multitask_ortho_enabled and not self.multitask_gradient_control_enabled:
            raise ValueError("multitask_ortho_enabled requires multitask_gradient_control_enabled=true.")
        if self.multitask_ortho_monitor_only and not self.multitask_ortho_enabled:
            raise ValueError("multitask_ortho_monitor_only requires multitask_ortho_enabled=true.")
        if self.multitask_ortho_start_step < 0:
            raise ValueError("multitask_ortho_start_step cannot be negative.")
        if self.multitask_ortho_interval <= 0:
            raise ValueError("multitask_ortho_interval must be positive.")
        if not -1 <= self.multitask_ortho_current_cosine_threshold <= 0:
            raise ValueError("multitask_ortho_current_cosine_threshold must be in [-1, 0].")
        if not -1 <= self.multitask_ortho_ema_cosine_threshold <= 1:
            raise ValueError("multitask_ortho_ema_cosine_threshold must be in [-1, 1].")
        if not 0 <= self.multitask_ortho_cosine_ema_beta < 1:
            raise ValueError("multitask_ortho_cosine_ema_beta must be in [0, 1).")
        if not 0 < self.multitask_ortho_norm_ratio_min <= 1 <= self.multitask_ortho_norm_ratio_max:
            raise ValueError("Ortho norm-ratio bounds must be positive and include 1.0.")
        if self.sid_token_weighting_enabled and not self.multitask_macro_training:
            raise ValueError("sid_token_weighting_enabled requires multitask_macro_training=true.")
        if self.sid_token_weight <= 0 or self.sid_text_weight <= 0:
            raise ValueError("sid_token_weight and sid_text_weight must be positive.")
        if self.user_action_aux_enabled and not self.multitask_macro_training:
            raise ValueError("user_action_aux_enabled requires multitask_macro_training=true.")
        action_aux_strengths = (
            self.user_action_history_trie_weight,
            self.user_action_continue_domain_extra,
            self.user_action_continue_separator_extra,
            self.user_action_no_early_stop_weight,
            self.user_action_stop_domain_weight,
            self.user_action_stop_tail_extra,
            self.user_action_topk_illegal_weight,
        )
        if any(value < 0 for value in action_aux_strengths):
            raise ValueError("Action Select auxiliary loss weights cannot be negative.")
        if not 0 <= self.user_action_aux_cap_ratio <= 1:
            raise ValueError("user_action_aux_cap_ratio must be in [0, 1].")
        if not 0 <= self.user_action_trie_cap_ratio <= 1:
            raise ValueError("user_action_trie_cap_ratio must be in [0, 1].")
        if not 0 <= self.user_action_length_cap_ratio <= 1:
            raise ValueError("user_action_length_cap_ratio must be in [0, 1].")
        if not 0 <= self.user_action_topk_illegal_cap_ratio <= 1:
            raise ValueError("user_action_topk_illegal_cap_ratio must be in [0, 1].")
        if self.user_action_topk_illegal_k <= 0:
            raise ValueError("user_action_topk_illegal_k must be positive.")
        if self.user_action_topk_illegal_margin < 0:
            raise ValueError("user_action_topk_illegal_margin cannot be negative.")
        if (
            self.user_action_aux_split_cap_enabled
            and self.user_action_trie_cap_ratio + self.user_action_length_cap_ratio
            > self.user_action_aux_cap_ratio + 1e-12
        ):
            raise ValueError("Action Trie and length cap ratios cannot exceed the total auxiliary cap ratio.")
        if self.user_action_aux_warmup_steps < 0:
            raise ValueError("user_action_aux_warmup_steps cannot be negative.")
        if self.user_action_max_stop_tail_positions <= 0:
            raise ValueError("user_action_max_stop_tail_positions must be positive.")
        if self.user_action_aux_full_vocab_chunk_size <= 0:
            raise ValueError("user_action_aux_full_vocab_chunk_size must be positive.")

        if self.packing:
            self.cutoff_len -= 1  # avoid pad_to_multiple_of, needs improve

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
