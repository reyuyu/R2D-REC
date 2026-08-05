# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
import time
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...data.multitask import (
    SUBTASK_RATIOS,
    TASK_IDS,
    Balanced40SuperCycle,
    MultiTaskMacroStepLoader,
    TaskDataLoader,
    TaskPackCollator,
    split_global_microbatch_allocation,
)
from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .multitask_gradient_controller import MultiTaskGradientController
from .user_action_auxiliary import (
    ACTION_STAT_SIZE,
    UserActionAuxiliaryController,
    action_statistics_to_metrics,
)


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        ref_model: Optional["torch.nn.Module"] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args: TrainingArguments = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        self.ref_model = ref_model

        if ref_model is not None:
            from trl.models.utils import prepare_deepspeed, prepare_fsdp

            if getattr(self.accelerator.state, "deepspeed_plugin", None) is not None:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif getattr(self.accelerator.state, "fsdp_plugin", None) is not None:
                if self.accelerator.is_fsdp2:
                    from accelerate.utils.fsdp_utils import fsdp2_prepare_model

                    self.ref_model = fsdp2_prepare_model(self.accelerator, self.ref_model)
                else:
                    self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        elif finetuning_args.use_eaft_loss:
            from ..trainer_utils import eaft_loss_func

            self.compute_loss_func = lambda outputs, labels, num_items_in_batch=None: eaft_loss_func(
                outputs, labels, num_items_in_batch, finetuning_args.eaft_alpha
            )
        elif finetuning_args.use_asft_loss:
            from ..trainer_utils import asft_loss_func

            self.compute_loss_func = partial(
                asft_loss_func,
                asft_alpha=finetuning_args.asft_alpha,
            )

        if training_args.fp8 and hasattr(self, "accelerator"):  # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)

    @override
    def create_optimizer(self, *args, **kwargs) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer(*args, **kwargs)

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        if self.finetuning_args.use_asft_loss:
            return_outputs = kwargs.pop("return_outputs", False)
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                )
                ref_logits = ref_outputs.logits
            outputs = model(**inputs)
            loss = self.compute_loss_func(outputs, inputs["labels"], ref_logits)
            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        input_ids_column = dataset["input_ids"]
        try:
            input_ids_list = input_ids_column.to_pylist()
        except AttributeError:
            input_ids_list = list(input_ids_column)

        decoded_inputs = self.processing_class.batch_decode(input_ids_list, skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")


class MultiTaskMacroSeq2SeqTrainer(CustomSeq2SeqTrainer):
    """Optional Trainer path where one outer step is one task-grouped macro-step.

    The normal ``Seq2SeqTrainer`` loop still owns optimizer/scheduler/checkpoint
    handling.  This class only replaces its train dataloader and expands one
    macro batch into its explicitly ordered task microbatches in
    :meth:`training_step`.
    """

    _MODEL_INPUT_KEYS = {"input_ids", "labels", "attention_mask", "position_ids", "token_type_ids"}

    def __init__(self, multitask_datasets, data_args, **kwargs):
        training_args = kwargs["args"]
        if training_args.max_steps <= 0:
            raise ValueError("multitask_macro_training requires max_steps > 0 (macro-step count).")
        if training_args.gradient_accumulation_steps != 1:
            raise ValueError(
                "multitask_macro_training requires gradient_accumulation_steps=1, "
                "because one macro-step already contains multiple task microbatches."
            )
        self._pending_gradient_log_metrics: dict[str, float] | None = None
        super().__init__(**kwargs)
        self.multitask_datasets = multitask_datasets
        self.multitask_data_args = data_args
        self.multitask_allocation = {
            task_name: int(data_args.multitask_microbatch_allocation[task_name]) for task_name in TASK_IDS
        }
        self.multitask_world_size = int(getattr(training_args, "world_size", 1))
        self.multitask_rank = int(getattr(training_args, "process_index", 0))
        self.global_microbatch_ddp = bool(data_args.multitask_global_microbatch_ddp)
        self.multitask_supercycle_mode = data_args.multitask_supercycle_mode
        if self.global_microbatch_ddp and self.multitask_world_size != 2:
            raise ValueError("multitask_global_microbatch_ddp currently requires exactly two DDP ranks.")

        if self.multitask_supercycle_mode == "balanced_40":
            if not self.global_microbatch_ddp:
                raise ValueError("balanced_40 super-cycle requires global two-rank DDP.")
            # All ranks advance an identical global source.  Rank-local slot
            # selection happens in MultiTaskMacroStepLoader, preserving unique
            # global packs and four backward calls per rank.
            task_active_ranks = {task_name: [0] for task_name in TASK_IDS}
            rank_allocations = None
            self.local_multitask_allocation = {}
        elif self.global_microbatch_ddp:
            rank_allocations, task_active_ranks = split_global_microbatch_allocation(
                self.multitask_allocation, self.multitask_world_size
            )
            self.local_multitask_allocation = rank_allocations[self.multitask_rank]
        else:
            task_active_ranks = {task_name: list(range(self.multitask_world_size)) for task_name in TASK_IDS}
            self.local_multitask_allocation = dict(self.multitask_allocation)

        loader_task_names = TASK_IDS if self.multitask_supercycle_mode == "balanced_40" else self.local_multitask_allocation
        task_loaders = {
            task_name: TaskDataLoader(
                task_name,
                multitask_datasets[task_name],
                SUBTASK_RATIOS[task_name],
                collator=TaskPackCollator(kwargs["data_collator"]),
                max_pack_length=int(data_args.multitask_max_pack_length),
                max_segments=data_args.multitask_max_segments_per_pack,
                seed=int(training_args.seed),
                rank=(0 if self.multitask_supercycle_mode == "balanced_40" else task_active_ranks[task_name].index(self.multitask_rank)),
                world_size=(1 if self.multitask_supercycle_mode == "balanced_40" else len(task_active_ranks[task_name])),
            )
            for task_name in loader_task_names
        }
        supercycle = Balanced40SuperCycle(int(training_args.seed)) if self.multitask_supercycle_mode == "balanced_40" else None
        self.macro_loader = MultiTaskMacroStepLoader(
            task_loaders,
            self.local_multitask_allocation,
            training_args.max_steps,
            global_allocation=self.multitask_allocation,
            supercycle=supercycle,
            rank=self.multitask_rank,
            world_size=self.multitask_world_size,
            synchronized_global_consumption=self.multitask_supercycle_mode == "balanced_40",
        )
        # DDP averages rank-local gradients. With a global 8-microbatch macro
        # spread over two ranks, divide each local loss by 4 so that its DDP
        # average is exactly the global mean over all eight microbatches.
        self._loss_scale_denominator = 8 // self.multitask_world_size if self.global_microbatch_ddp else sum(self.local_multitask_allocation.values())
        self.gradient_controller = None
        if data_args.multitask_gradient_control_enabled:
            self.gradient_controller = MultiTaskGradientController(
                self.model,
                data_args,
                loss_divisor=self._loss_scale_denominator,
            )
        self.user_action_auxiliary = None
        if data_args.user_action_aux_enabled:
            self.user_action_auxiliary = UserActionAuxiliaryController(self.processing_class, data_args)
        self._task_stat_size = 5 + (ACTION_STAT_SIZE if self.user_action_auxiliary is not None else 0)
        self.monitoring_enabled = bool(data_args.multitask_monitoring)
        self.monitoring_path = os.path.join(training_args.output_dir, "monitor", "metrics.jsonl")
        if (
            self.monitoring_enabled
            or self.gradient_controller is not None
            or self.user_action_auxiliary is not None
        ) and self.is_world_process_zero():
            os.makedirs(os.path.dirname(self.monitoring_path), exist_ok=True)
        # Loader state, not Trainer's default data-skip, is the source of truth on resume.
        self.args.ignore_data_skip = True
        allocation_text = ", ".join(f"{task}={count}" for task, count in self.multitask_allocation.items())
        logger.warning_rank0(
            "multitask_macro_training is enabled: num_train_epochs is ignored and max_steps is interpreted as macro-step count."
        )
        logger.info_rank0(
            "Multitask macro training: enabled\n"
            f"Macro-step limit: {training_args.max_steps}\n"
            f"Microbatch allocation: {allocation_text}\n"
            "Microbatches per macro-step: 8\n"
            f"Optimizer steps: {training_args.max_steps}\n"
            "Gradient accumulation steps: 1\n"
            "num_train_epochs: ignored\n"
            "Logging/eval/save units: macro-steps"
        )
        if self.global_microbatch_ddp:
            if self.multitask_supercycle_mode == "balanced_40":
                logger.info_rank0(
                    "Global microbatch DDP: enabled\n"
                    "Super-cycle: balanced_40 (35 base, 3 user-boost, 1 recommendation-boost, 1 world macro)\n"
                    "Global microbatches per macro-step: 8\n"
                    "Per-rank microbatches: 4 (interleaved global slots, all task loaders synchronized)"
                )
            else:
                local_text = ", ".join(f"{task}={count}" for task, count in self.local_multitask_allocation.items())
                logger.info_rank0(
                    "Global microbatch DDP: enabled\n"
                    "Global microbatches per macro-step: 8\n"
                    "Per-rank microbatches: 4\n"
                    f"Rank 0 allocation: {', '.join(f'{task}={count}' for task, count in rank_allocations[0].items())}\n"
                    f"Rank 1 allocation: {', '.join(f'{task}={count}' for task, count in rank_allocations[1].items())}\n"
                    f"This rank allocation: {local_text}"
                )

    @override
    def get_train_dataloader(self):
        # Do not call Accelerator.prepare_data_loader here: each subtask sampler
        # has already rank-sharded a common global pack plan.
        return self.macro_loader

    def compute_task_microbatches(
        self, model, task_name: str, microbatches: list[dict[str, Any]], collect_metrics: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Backprop one task group without changing the macro-step optimizer boundary."""
        total_loss = torch.zeros((), device=self.args.device)
        total_microbatches = self._loss_scale_denominator
        statistics = torch.zeros(getattr(self, "_task_stat_size", 5), device=self.args.device) if collect_metrics else None
        task_weight = self.gradient_controller.task_weight(task_name) if self.gradient_controller is not None else 1.0
        for microbatch in microbatches:
            model_inputs = self._prepare_inputs(
                {key: value for key, value in microbatch.items() if key in self._MODEL_INPUT_KEYS}
            )
            action_active = (
                getattr(self, "user_action_auxiliary", None) is not None
                and task_name == "user"
                and microbatch["subtask_name"] == "action_nocot"
            )
            with self.compute_loss_context_manager():
                if action_active:
                    base_loss, outputs = self.compute_loss(model, model_inputs, return_outputs=True)
                    action_result = self.user_action_auxiliary.compute(
                        outputs.logits,
                        model_inputs["labels"],
                        microbatch.get("action_aux_metadata", []),
                        base_loss,
                        self.state.global_step,
                    )
                    loss = base_loss + action_result.loss
                else:
                    loss = self.compute_loss(model, model_inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()
            if statistics is not None:
                statistics[0] += loss.detach()
                statistics[1] += 1
                statistics[2] += model_inputs["input_ids"].numel()
                statistics[3] += int(microbatch["supervised_token_count"])
                statistics[4] += int(microbatch["num_segments"])
                if action_active:
                    statistics[5:] += action_result.statistics
            weighted_loss = loss * task_weight if self.gradient_controller is not None else loss
            scaled_loss = weighted_loss / total_microbatches
            if self.gradient_controller is not None:
                self.gradient_controller.set_current_task(task_name)
            try:
                self.accelerator.backward(scaled_loss)
            finally:
                if self.gradient_controller is not None:
                    self.gradient_controller.clear_current_task()
            total_loss = total_loss + scaled_loss.detach()
        return total_loss, statistics

    def _loss_monitoring_fields(self, task_statistics: torch.Tensor) -> dict[str, float]:
        """Return raw task means and the exact DDP-equivalent weighted macro loss."""
        fields: dict[str, float] = {}
        weighted_sum = 0.0
        global_microbatch_count = 0.0
        for task_name, task_id in TASK_IDS.items():
            loss_sum, count = task_statistics[task_id, :2].tolist()
            if not count:
                continue
            raw_mean = float(loss_sum) / float(count)
            fields[f"loss_raw_{task_name}"] = raw_mean
            task_weight = (
                self.gradient_controller.task_weight(task_name)
                if self.gradient_controller is not None
                else 1.0
            )
            weighted_sum += float(task_weight) * float(loss_sum)
            global_microbatch_count += float(count)
        if global_microbatch_count:
            # With global DDP microbatch scheduling this is exactly
            # sum(weight * raw_microbatch_loss) / global_microbatch_count,
            # the scalar whose gradient reaches the optimizer after DDP averaging.
            fields["loss_gradnorm_total"] = weighted_sum / global_microbatch_count
        return fields

    def _write_monitor_record(
        self,
        task_statistics: torch.Tensor,
        macro_seconds: float,
        gradient_metrics: dict[str, float] | None = None,
    ) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(task_statistics, op=torch.distributed.ReduceOp.SUM)
        if not self.is_world_process_zero():
            return

        macro_step = self.state.global_step + 1
        record: dict[str, Any] = {
            "time": time.time(),
            "macro_step": macro_step,
            "logical_microbatch_step": macro_step * sum(self.multitask_allocation.values()),
            "global_microbatch_step": macro_step
            * sum(self.multitask_allocation.values())
            * (1 if self.global_microbatch_ddp else self.multitask_world_size),
            "macro_seconds": macro_seconds,
            "learning_rate": self.optimizer.param_groups[0]["lr"] if self.optimizer is not None else None,
        }
        record.update(self._loss_monitoring_fields(task_statistics))
        for task_name, task_id in TASK_IDS.items():
            count = int(task_statistics[task_id, 1].item())
            if count:
                record[f"{task_name}_microbatches"] = count
        if self.user_action_auxiliary is not None:
            action_metrics = action_statistics_to_metrics(task_statistics[TASK_IDS["user"], 5:])
            record.update(action_metrics)
            self._pending_gradient_log_metrics = {**(self._pending_gradient_log_metrics or {}), **action_metrics}
        if gradient_metrics is not None:
            record.update(gradient_metrics)
        with open(self.monitoring_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    @override
    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if self._pending_gradient_log_metrics is not None and "loss" in logs:
            logs = {**logs, **self._pending_gradient_log_metrics}
            self._pending_gradient_log_metrics = None
        return super().log(logs, start_time)

    @override
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        expected_allocation = self.macro_loader.current_local_allocation
        if not isinstance(inputs, dict) or set(inputs) != set(expected_allocation):
            raise ValueError("MultiTaskMacroSeq2SeqTrainer expected a grouped macro batch.")
        macro_started = time.perf_counter()
        next_macro_step = self.state.global_step + 1
        if self.gradient_controller is not None:
            self.gradient_controller.begin_macro_step(next_macro_step)
        logging_steps = max(1, int(self.args.logging_steps))
        write_monitor_record = (
            self.monitoring_enabled or self.gradient_controller is not None or self.user_action_auxiliary is not None
        ) and next_macro_step % logging_steps == 0
        collect_metrics = write_monitor_record or self.gradient_controller is not None
        task_statistics = (
            torch.zeros((len(TASK_IDS), self._task_stat_size), device=self.args.device) if collect_metrics else None
        )
        total_loss = torch.zeros((), device=self.args.device)
        for task_name, microbatches in inputs.items():
            if len(microbatches) != expected_allocation[task_name]:
                raise RuntimeError(f"Unexpected microbatch count for task {task_name}.")
            for microbatch in microbatches:
                if microbatch["task_name"] != task_name or microbatch["task_id"] != TASK_IDS[task_name]:
                    raise RuntimeError("Task boundary violation in multitask macro batch.")
            task_loss, task_metrics = self.compute_task_microbatches(model, task_name, microbatches, collect_metrics)
            total_loss = total_loss + task_loss
            if task_statistics is not None and task_metrics is not None:
                task_statistics[TASK_IDS[task_name]] = task_metrics
        gradient_metrics = None
        if self.gradient_controller is not None:
            local_loss_sums = {
                task: float(task_statistics[TASK_IDS[task], 0].item()) for task in self.gradient_controller.tasks
            }
            local_counts = {
                task: int(task_statistics[TASK_IDS[task], 1].item()) for task in self.gradient_controller.tasks
            }
            gradient_metrics = self.gradient_controller.finish_macro_step(local_loss_sums, local_counts)
        if self.global_microbatch_ddp:
            self.accelerator.wait_for_everyone()
        if write_monitor_record and task_statistics is not None:
            if gradient_metrics is not None:
                self._pending_gradient_log_metrics = {**(self._pending_gradient_log_metrics or {}), **gradient_metrics}
            self._write_monitor_record(
                task_statistics,
                time.perf_counter() - macro_started,
                gradient_metrics=gradient_metrics,
            )
        current_global_allocation = self.macro_loader.current_global_allocation
        self._microbatch_step = getattr(self, "_microbatch_step", 0) + sum(current_global_allocation.values())
        logger.info_rank0(
            f"multitask macro_step={self.state.global_step + 1} "
            f"microbatch_step={self._microbatch_step} optimizer_step={self.state.global_step + 1} "
            f"allocation={current_global_allocation}"
        )
        return total_loss.detach()

    @override
    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        self.accelerator.wait_for_everyone()
        checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")
        if self.global_microbatch_ddp:
            state_name = f"multitask_macro_state_rank{self.multitask_rank}.json"
        else:
            state_name = "multitask_macro_state.json"
        with open(os.path.join(checkpoint_dir, state_name), "w", encoding="utf-8") as file:
            json.dump(self.macro_loader.state_dict(), file, ensure_ascii=False)
        if self.gradient_controller is not None:
            self.gradient_controller.save_checkpoint(checkpoint_dir, self.is_world_process_zero())
        self.accelerator.wait_for_everyone()

    @override
    def _load_from_checkpoint(self, resume_from_checkpoint, *args, **kwargs):
        result = super()._load_from_checkpoint(resume_from_checkpoint, *args, **kwargs)
        state_name = (
            f"multitask_macro_state_rank{self.multitask_rank}.json"
            if self.global_microbatch_ddp
            else "multitask_macro_state.json"
        )
        state_path = os.path.join(resume_from_checkpoint, state_name)
        if os.path.isfile(state_path):
            with open(state_path, encoding="utf-8") as file:
                self.macro_loader.load_state_dict(json.load(file))
            logger.info_rank0("Restored multitask macro sampler state; resuming from the next complete macro-step.")
        if self.gradient_controller is not None and self.gradient_controller.load_checkpoint(resume_from_checkpoint):
            logger.info_rank0("Restored multitask gradient controller state on every rank.")
        return result
