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

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.dataset = split_arg(self.dataset)
        self.eval_dataset = split_arg(self.eval_dataset)

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
            required_tasks = {"material", "user", "recommendation", "world"}
            if set(self.multitask_microbatch_allocation) != required_tasks:
                raise ValueError("multitask_microbatch_allocation must contain material, user, recommendation and world.")
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

        if self.packing:
            self.cutoff_len -= 1  # avoid pad_to_multiple_of, needs improve

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
