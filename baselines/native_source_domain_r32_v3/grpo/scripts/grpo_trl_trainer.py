# -*- coding: utf-8 -*-
"""RecGRPOTrainer: route-aware minimal subclass of TRL GRPOTrainer.
Think G=4 / NoThink G=8 via route-homogeneous batches + dynamic num_generations.
Population-std advantage override. No training performed in this file's tests."""
import sys
sys.path.insert(0, "/data/GRPO/scripts")
import trl_import_fix  # must run before trl.trainer imports (no site-packages change)

import collections
import json
import torch
from datasets import Dataset
from torch.utils.data import Sampler

from trl import GRPOConfig, GRPOTrainer

from grpo_sid import parse_sid, final_sid, q_reward, think_reward

M_THINK = 4
M_NO = 8
ROUTE_G = {"think": M_THINK, "no_think": M_NO}
ROUTE_TEMP = {"think": 0.9, "no_think": 1.0}
ROUTE_TOP_P = {"think": 0.95, "no_think": 1.0}
# Route loss multipliers: NoThink runs ~2x generation batches per epoch
# (2 groups per global batch vs 4 for Think), so per-sample weight 0.5 makes
# single-group total weight equal: 4 samples x 1.0 == 8 samples x 0.5 == 4.
ROUTE_LOSS_W = {"think": 1.0, "no_think": 0.5}
ROUTE_ID = {"think": 0, "no_think": 1}


# ---------------- route dataset ----------------

def build_route_dataset(data_path, n_groups=20, seed=20260816, chunk=8):
    """Route-homogeneous alternating dataset.
    Order: [think x chunk][no_think x chunk][think x chunk]...
    chunk=8 aligns with RepeatSampler chunking under dynamic G (G=4 -> sampler
    chunk=4 idx, so 2 rollouts per segment; G=8 -> chunk=2 idx, 4 rollouts per
    segment): yields 3 Think + 3 NoThink rollouts with segments T,T,N,N,T,T.
    """
    import random
    rows = [json.loads(l) for l in open(data_path, encoding="utf-8")]
    by_group = collections.defaultdict(dict)
    for r in rows:
        by_group[r["recommendation_group_id"]][r["route"]] = r
    gids = sorted(by_group.keys())
    rng = random.Random(seed)
    rng.shuffle(gids)
    chosen = gids[:n_groups]
    recs = []
    for gid in chosen:
        g = by_group[gid]
        for route in ("think", "no_think"):
            rec = g[route]
            recs.append({
                "prompt": rec["prompt"],
                "route": route,
                "recommendation_group_id": gid,
                "target_domain": rec["target_domain"],
                "all_gold_sids": rec["all_gold_sids"],
            })
    # alternating chunks
    think = [r for r in recs if r["route"] == "think"]
    no_think = [r for r in recs if r["route"] == "no_think"]
    ordered = []
    for i in range(0, len(think), chunk):
        ordered += think[i:i + chunk]
        ordered += no_think[i:i + chunk]
    return Dataset.from_list(ordered)


# ---------------- route-aware sampler (true dynamic G) ----------------

class RouteAwareRepeatSampler(Sampler):
    """Route-aware RepeatSampler: real dynamic G without mutating
    num_generations after init (TRL's RepeatSampler fixes mini_repeat_count at
    construction time, so the old G=4-init hack mixed 2 prompts into G=8 groups).

    Global generation batch is ALWAYS 16:
      Think  : 4 unique prompts x 4 repeats  -> A A A A B B B B C C C C D D D D
      NoThink: 2 unique prompts x 8 repeats  -> A A A A A A A A B B B B B B B B

    The dataset must be route-homogeneous in index order (build_route_dataset
    guarantees this); chunk boundaries are computed from each index's route.
    repeat_count (= num_iterations * steps_per_generation) replays each chunk
    back-to-back so TRL reuses one rollout across policy iterations."""

    def __init__(self, data_source, generation_batch_size=16, repeat_count=1,
                 shuffle=False, seed=None):
        assert not shuffle, (
            "RouteAwareRepeatSampler requires shuffle=False to keep "
            "route-homogeneous order")
        self.data_source = data_source
        self.gen_batch = generation_batch_size
        self.repeat_count = repeat_count
        self._chunks = self._build_chunks()

    def _chunk_unique(self, route):
        # unique prompts per chunk: gen_batch / G  (Think 16//4=4, NoThink 16//8=2)
        return self.gen_batch // ROUTE_G[route]

    def _build_chunks(self):
        # Walk route-homogeneous runs; within each run cut chunks of
        # gen_batch/G unique indices (tail shorter than a chunk is dropped,
        # matching TRL RepeatSampler drop_last semantics).
        n = len(self.data_source)
        chunks = []
        i = 0
        while i < n:
            route = self.data_source[i]["route"]
            j = i
            while j < n and self.data_source[j]["route"] == route:
                j += 1
            seg = list(range(i, j))
            unique = self._chunk_unique(route)
            for k in range(0, len(seg) - unique + 1, unique):
                chunks.append((route, seg[k:k + unique]))
            i = j
        return chunks

    def __iter__(self):
        for route, indices in self._chunks:
            for _ in range(self.repeat_count):
                for idx in indices:
                    for _ in range(ROUTE_G[route]):
                        yield idx

    def __len__(self):
        return sum(len(indices) * ROUTE_G[route] * self.repeat_count
                   for route, indices in self._chunks)



# ---------------- route multiplier (pure, testable) ----------------

def route_multiplier(route_ids):
    # route_id tensor (think=0, no_think=1) -> loss multiplier;
    # asserts the rollout batch is route-homogeneous.
    assert bool((route_ids == route_ids[0]).all()), (
        "route_id mixed inside rollout batch")
    route = "think" if int(route_ids[0]) == ROUTE_ID["think"] else "no_think"
    return ROUTE_LOSS_W[route]


# ---------------- population advantage helper (pure, testable) ----------------

def group_advantages_population(rewards, G, eps=1e-4):
    """TRL group advantage with population std (correction=0).
    rewards: [B*G]. Returns advantages [B*G]; all-zero if group std==0."""
    r = torch.as_tensor(rewards, dtype=torch.float32)
    mean = r.view(-1, G).mean(dim=1)
    std = r.view(-1, G).std(dim=1, correction=0)
    mean = mean.repeat_interleave(G, dim=0)
    std = std.repeat_interleave(G, dim=0)
    adv = r - mean
    is_zero = torch.isclose(std, torch.zeros_like(std))
    adv = torch.where(is_zero, torch.zeros_like(adv), adv / (std + eps))
    return adv


# ---------------- reward funcs ----------------

def make_nothink_reward_func(tokenizer=None):
    """NoThink reward_func: computes q() ONLY for no_think samples; returns
    None for think samples (TRL converts None -> NaN, excluded by nansum).
    Route comes from the dataset column passed through TRL reward kwargs.
    SID is parsed from RAW completion_ids (tokenizer.decode with
    skip_special_tokens=False) - TRL's decoded completions use
    skip_special_tokens=True which strips <|..._begin|>/<s_a_>/<s_b_>/<s_c_>
    added tokens and would turn valid SIDs into -1."""
    if tokenizer is None:
        raise RuntimeError("make_nothink_reward_func requires tokenizer "
                           "(raw completion_ids SID parse)")
    def reward_func(prompts, completions, completion_ids, **kwargs):
        golds_list = kwargs["all_gold_sids"]
        routes = kwargs.get("route")
        if routes is None:
            raise RuntimeError("nothink_reward requires dataset column 'route' "
                               "(route-specific reward routing)")
        out = []
        for cids, golds, route in zip(completion_ids, golds_list, routes):
            if route != "no_think":
                out.append(None)  # Think sample: not this reward's route
                continue
            text = tokenizer.decode(cids, skip_special_tokens=False)
            sid = final_sid(text)
            gold_set = set()
            for s in golds:
                t = parse_sid(s)
                if t:
                    gold_set.add(t)
            out.append(q_reward(sid, gold_set))
        return out
    reward_func.__name__ = "nothink_reward"
    return reward_func


def make_think_reward_func(beam32_fn=None):
    """Think reward: completion = sampled CoT; reward via Beam32 hierarchical credits.
    beam32_fn(prompts, completions, completion_ids_list, tokenizer, gold_sets)
    -> list of rewards. completion_ids are the RAW sampled tokens (TRL decodes
    completions with skip_special_tokens=True which strips </think>/SID tokens,
    so beam32_fn must re-decode from ids)."""
    def reward_func(prompts, completions, completion_ids, **kwargs):
        golds_list = kwargs["all_gold_sids"]
        routes = kwargs.get("route")
        if routes is None:
            raise RuntimeError("think_reward requires dataset column 'route' "
                               "(route-specific reward routing)")
        think_idx = [i for i, r in enumerate(routes) if r == "think"]
        if not think_idx:
            # NoThink batch: beam32 MUST NOT run -> strict call count 0
            return [None] * len(prompts)
        gold_sets = []
        for golds in golds_list:
            gs = set()
            for s in golds:
                t = parse_sid(s)
                if t:
                    gs.add(t)
            gold_sets.append(gs)
        if beam32_fn is None:
            raise RuntimeError("think reward requires beam32_fn (GPU smoke wiring)")
        sub_prompts = [prompts[i] for i in think_idx]
        sub_completions = [completions[i] for i in think_idx]
        sub_ids = [completion_ids[i] for i in think_idx]
        sub_golds = [gold_sets[i] for i in think_idx]
        sub_rewards = beam32_fn(sub_prompts, sub_completions, sub_ids, sub_golds)
        out = [None] * len(prompts)
        for pos, r in zip(think_idx, sub_rewards):
            out[pos] = r
        return out
    reward_func.__name__ = "think_reward"
    return reward_func


# ---------------- RecGRPOTrainer ----------------

class RecGRPOTrainer(GRPOTrainer):
    """Route-aware GRPOTrainer with:
    - route-homogeneous batches (alternating Think/NoThink in dataset order)
    - dynamic num_generations (Think 4 / NoThink 8) + temperature/top_p per route
    - population-std group advantages (override _generate_and_score_completions,
      single change: std(dim=1, correction=0))
    """

    def __init__(self, *args, **kwargs):
        # transformers 5.x removed PreTrainedModel.warnings_issued which TRL 0.24
        # assumes; add it on the instance (user-code fix, site-packages untouched)
        model = kwargs.get("model")
        if model is None and args:
            model = args[0]
        if model is not None and isinstance(model, torch.nn.Module) and not hasattr(model, "warnings_issued"):
            model.warnings_issued = {}
        self._smoke_rollout_id = 0
        self._smoke_log = []
        self._smoke_policy_epoch = {}
        super().__init__(*args, **kwargs)

    def _generate_single_turn(self, prompts, images=None):
        """TRL transformers-path copy with one addition: Think-route completions
        are truncated at the </think> token (single added token 151668), keeping
        the CoT action bounded and final SIDs out of the action. No site-packages
        change; stop is applied post-hoc since TRL does not pass stop_strings."""
        from trl.extras.profiling import profiling_context
        from trl.models.utils import unwrap_model_for_generation
        from trl.data_utils import maybe_apply_chat_template
        from contextlib import nullcontext
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa

        device = self.accelerator.device
        kwargs = {}
        from grpo_model import render_prompt
        # TRL's maybe_apply_chat_template passes non-conversational (str) prompts
        # through UNCHANGED -> rollout would see bare prompt text without the SFT
        # chat template. Use the shared SFT-identical renderer instead.
        prompts_text = [render_prompt(self.processing_class, prompt) for prompt in prompts]
        forward_kwargs = {}
        generate_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True,
            padding_side="left", max_length=self.max_prompt_length,
            truncation=True, add_special_tokens=False, **kwargs,
        )
        # NOTE: use the transformers Trainer base _prepare_inputs (not the GRPO
        # route-aware override, which would recurse into rollout).
        from transformers import Trainer as _HFTrainer
        generate_inputs = _HFTrainer._prepare_inputs(self, generate_inputs)

        # ---- token-id based </think> stopping for the Think route ----
        # (stop_strings unreliable here; token-id StoppingCriteria is exact)
        stopping_criteria = None
        if getattr(self, "num_generations", 0) == 4:  # Think route
            from transformers import StoppingCriteria, StoppingCriteriaList

            class _ThinkStop(StoppingCriteria):
                def __init__(self, tid):
                    self.tid = tid

                def __call__(self, input_ids, scores, **kw):
                    # per-sample BoolTensor: each sequence stops at its own </think>.
                    # (scalar .any() would stop the whole batch at the first closure)
                    return input_ids[:, -1] == self.tid

            think_tok_id = self.processing_class.encode("</think>", add_special_tokens=False)[0]
            stopping_criteria = StoppingCriteriaList([_ThinkStop(think_tok_id)])
        with (
            profiling_context(self, "transformers.generate"),
            unwrap_model_for_generation(
                self.model_wrapped, self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
            ) as unwrapped_model,
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False)
            if getattr(self, "is_fsdp_enabled", False) else nullcontext(),
        ):
            prompt_completion_ids = unwrapped_model.generate(
                **generate_inputs, generation_config=self.generation_config,
                disable_compile=True,
                tokenizer=self.processing_class,  # required for stop_strings in transformers 5.3
                stopping_criteria=stopping_criteria,
            )
        prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # ---- Think-route truncation at </think> token ----
        is_think = getattr(self, "num_generations", 0) == 4
        completion_ids_list_raw = [completion_ids[i].tolist() for i in range(completion_ids.size(0))]
        if is_think:
            think_tok = self.processing_class.encode("</think>", add_special_tokens=False)[0]
            trimmed = []
            for c in completion_ids_list_raw:
                try:
                    pos = c.index(think_tok)
                    trimmed.append(c[: pos + 1])
                except ValueError:
                    trimmed.append(c)  # no closure; keep as-is (truncation recorded later)
            completion_ids_list_raw = trimmed

        # Mask everything after the first EOS token (original TRL logic, list form)
        completion_ids = []
        for c in completion_ids_list_raw:
            if self.eos_token_id in c:
                c = c[: c.index(self.eos_token_id) + 1]
            completion_ids.append(c)
        prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool())]
        logprobs = None
        return prompt_ids, completion_ids, logprobs, forward_kwargs

    def _get_train_sampler(self, dataset=None):
        # True dynamic G: RouteAwareRepeatSampler expands Think 4x4 / NoThink
        # 2x8 chunks; TRL's fixed mini_repeat_count is no longer relied upon.
        if dataset is None:
            dataset = self.train_dataset
        return RouteAwareRepeatSampler(
            data_source=dataset,
            generation_batch_size=self.args.generation_batch_size,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _prepare_inputs(self, generation_batch):
        # route-homogeneous batch: take route from first example
        if isinstance(generation_batch, list):
            route = generation_batch[0]["route"]
        else:
            route = generation_batch["route"][0] if isinstance(generation_batch["route"], (list, torch.Tensor)) else generation_batch["route"]
        self.num_generations = ROUTE_G[route]
        # dynamic sampling params must be synced across ALL consumers:
        # generation uses generation_config; policy logprob scoring uses
        # self.temperature/self.top_p; TRL internals read self.args.
        self.args.temperature = ROUTE_TEMP[route]
        self.args.top_p = ROUTE_TOP_P[route]
        self.temperature = ROUTE_TEMP[route]
        self.top_p = ROUTE_TOP_P[route]
        # TRL RepeatSampler already repeats each prompt num_generations times
        # (mini_repeat_count), so generate must use num_return_sequences=1 (the
        # default); do NOT touch generation_config.num_return_sequences here.
        if getattr(self, "generation_config", None) is not None:
            self.generation_config.temperature = ROUTE_TEMP[route]
            self.generation_config.top_p = ROUTE_TOP_P[route]
            # stop CoT at </think> only for the Think route (NoThink completions
            # also contain </think> inside the empty-think wrapper -> must NOT stop)
            if route == "think":
                self.generation_config.stop_strings = ["</think>"]
            else:
                self.generation_config.stop_strings = None
            if self._smoke_rollout_id <= 2:
                import os
                if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                    print(f"[dbg] route={route} G={self.num_generations} "
                          f"nrs={self.generation_config.num_return_sequences} "
                          f"stop_strings={self.generation_config.stop_strings}", flush=True)
        # rollout identity: new rollout when generation happens (buffers reset)
        generate_every = self.args.steps_per_generation * self.num_iterations
        if self._step % generate_every == 0 or self._buffered_inputs is None:
            self._smoke_rollout_id += 1
            self._smoke_log.append(dict(
                rollout_id=self._smoke_rollout_id,
                step=self._step,
                route=route,
                num_generations=self.num_generations,
                local_prompts=len(generation_batch) if isinstance(generation_batch, list) else 0,
            ))
        return super()._prepare_inputs(generation_batch)

    def _generate_and_score_completions(self, inputs):
        """Copy of TRL 0.24 _generate_and_score_completions (transformers path,
        no images/vllm/ref branches) with ONE change:
        std_rewards uses population std (correction=0)."""
        import time as _t
        _t0 = _t.time()
        from trl.trainer.grpo_trainer import (
            pad, gather, gather_object, nanstd, is_conversational,
        )
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        prompts = [x["prompt"] for x in inputs]
        images = None
        # ---- runtime assert: dynamic temperature/top_p actually in effect ----
        _route = inputs[0]["route"]
        assert abs(self.args.temperature - ROUTE_TEMP[_route]) < 1e-6, (
            f"args.temperature {self.args.temperature} != {ROUTE_TEMP[_route]} ({_route})")
        assert abs(self.temperature - ROUTE_TEMP[_route]) < 1e-6, (
            f"self.temperature {self.temperature} != {ROUTE_TEMP[_route]} ({_route})")
        assert abs(self.generation_config.temperature - ROUTE_TEMP[_route]) < 1e-6, (
            f"generation_config.temperature {self.generation_config.temperature} != {ROUTE_TEMP[_route]}")
        assert abs(self.args.top_p - ROUTE_TOP_P[_route]) < 1e-6, (
            f"args.top_p {self.args.top_p} != {ROUTE_TOP_P[_route]} ({_route})")
        assert abs(self.top_p - ROUTE_TOP_P[_route]) < 1e-6, (
            f"self.top_p {self.top_p} != {ROUTE_TOP_P[_route]} ({_route})")
        assert abs(self.generation_config.top_p - ROUTE_TOP_P[_route]) < 1e-6, (
            f"generation_config.top_p {self.generation_config.top_p} != {ROUTE_TOP_P[_route]}")
        _tgen = _t.time()
        (
            prompt_ids_list, completion_ids_list, num_items_in_batch,
            sampling_per_token_logps_list, forward_kwargs,
        ) = self._generate(prompts, images)
        gen_wall_sec = _t.time() - _tgen  # CoT generation wall (this rank)

        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None

        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model, prompt_completion_ids, attention_mask,
                    logits_to_keep, batch_size,
                )
            else:
                old_per_token_logps = None
            ref_per_token_logps = None  # beta=0

        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # ---- runtime DDP assertions (dynamic G correctness) ----
        # NOTE: rewards_per_func is GATHERED across ranks (TRL _calculate_rewards
        # gathers before returning), so n_all = generation_batch_size (global).
        n_all = rewards.numel()
        n_prompts_local = len(prompts)
        world = self.accelerator.num_processes
        gen_batch = getattr(self.args, "generation_batch_size", None) or (n_prompts_local * world)
        assert n_all == gen_batch, (
            f"gathered rewards {n_all} != generation_batch_size {gen_batch}")
        assert gen_batch % self.num_generations == 0, (
            f"generation_batch_size {gen_batch} not divisible by G={self.num_generations}")
        # local generation count sanity: each rank holds gen_batch/world examples
        assert n_prompts_local == gen_batch // world, (
            f"local prompts {n_prompts_local} != gen_batch/world {gen_batch // world}")
        assert all(example.get("route", inputs[0].get("route")) == inputs[0]["route"]
                   for example in inputs), "mixed route inside generation batch"
        # ---- HARD group_id assert: every rewards.view(-1, G) group must be the
        # G samples of ONE prompt (same recommendation_group_id). TRL gather
        # keeps rank order, same as gather_object below. ----
        group_ids_all = gather_object([x["recommendation_group_id"] for x in inputs])
        assert len(group_ids_all) == n_all, (
            f"gathered group_ids {len(group_ids_all)} != rewards {n_all}")
        for gi in range(0, n_all, self.num_generations):
            seg = group_ids_all[gi:gi + self.num_generations]
            assert len(set(seg)) == 1, (
                f"view(-1,G) group {gi // self.num_generations} mixes group_ids {seg}")
        # group-relative normalization must not cross prompts: TRL gather keeps
        # rank order, so view(-1, G) groups the G completions of each prompt.

        # group normalization (population std — the ONLY change vs upstream)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards in ["group", "none"]:
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1, correction=0)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(f"Invalid scale_rewards: {self.scale_rewards}")
        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
            # batch-aligned route_id tensor (think=0, no_think=1): scalar str
            # would be corrupted by shuffle_sequence_dict (str indexed by char)
            "route_id": torch.full(
                (completion_ids.size(0),), ROUTE_ID[inputs[0]["route"]],
                dtype=torch.long, device=device),
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps

        # ---- smoke monitoring: rollout-level stats ----
        import statistics as _st
        _plens = sorted(len(x) for x in prompt_ids_list)
        entry = dict(
            rollout_id=self._smoke_rollout_id,
            route=self._smoke_log[-1]["route"] if self._smoke_log else "?",
            num_generations=self.num_generations,
            reward_mean=float(mean_grouped_rewards.mean()),
            reward_std=float(std_rewards.mean()),
            zero_std_ratio=float(is_std_zero.float().mean()),
            advantage_mean=float(all_process_advantages.mean()),
            advantage_std=float(all_process_advantages.std(unbiased=False)),
            abs_advantage_mean=float(all_process_advantages.abs().mean()),
            rollout_sec=round(_t.time() - _t0, 2),
            gen_wall_sec=round(gen_wall_sec, 2),
            prompt_len_mean=round(_st.mean(_plens), 1),
            prompt_len_p95=_plens[min(len(_plens) - 1, int(0.95 * len(_plens)))],
            prompt_len_max=_plens[-1],
        )
        # NoThink six-level reward distribution (local rewards, pre-gather slice)
        if entry["route"] == "no_think":
            local_r = rewards[process_slice].tolist()
            entry["reward_level_dist"] = {
                str(k): local_r.count(k) for k in (-1.0, -0.25, 0.0, 0.5, 2.0, 8.0)
            }
        self._smoke_log[-1].update(entry)
        return output

    def _compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Copy of TRL 0.24 _compute_loss with smoke instrumentation only
        (ratio/clip/KL/timing). Loss math identical to upstream."""
        import time as _t
        t0 = _t.time()
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=True,
        )

        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if self.loss_type == "grpo":
            per_sample_loss = (per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            # route-equal weighting: NoThink has 2x batches per epoch (2 groups
            # per 16-sample batch vs 4), so per-sample multiplier 0.5 makes
            # single-group total weight equal (4 == 8*0.5) and epoch totals equal.
            route_ids = inputs["route_id"]
            multiplier = route_multiplier(route_ids)
            loss = (per_sample_loss * multiplier).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # ---- smoke instrumentation (no gradient impact) ----
        with torch.no_grad():
            flat = log_ratio[completion_mask.bool()].float()
            ratio_flat = torch.exp(flat)
            pe = self._smoke_policy_epoch.get(self._smoke_rollout_id, 0)
            entry = {
                f"policy_epoch_{pe+1}": 1,
                f"step_{pe+1}": self.state.global_step,
                f"ratio_mean_ep{pe+1}": float(ratio_flat.mean()),
                f"ratio_std_ep{pe+1}": float(ratio_flat.std(unbiased=False)),
                f"ratio_p95_ep{pe+1}": float(ratio_flat.quantile(0.95)),
                f"ratio_p99_ep{pe+1}": float(ratio_flat.quantile(0.99)),
                f"max_abs_log_ratio_ep{pe+1}": float(flat.abs().max()),
                f"clip_fraction_ep{pe+1}": float(((ratio_flat - 1.0).abs() > self.epsilon_low).float().mean()),
                f"approx_kl_ep{pe+1}": float((ratio_flat - 1.0 - flat).mean()),
                f"action_tokens_ep{pe+1}": float(completion_mask.sum()),
                f"policy_fwb_sec_ep{pe+1}": round(_t.time() - t0, 2),
            }
            self._smoke_policy_epoch[self._smoke_rollout_id] = pe + 1
            self._smoke_log[-1].update(entry)
        return loss
