"""CPU regressions for alpha recommendation monitor-only metrics."""

from __future__ import annotations

import torch

from rec_pu.alpha_recommendation_monitor import (
    ALL_METRIC_NAMES,
    collect_alpha_recommendation_monitor,
)
from rec_pu.recommendation_pu_phase2 import PackedSegment, locate_packed_rec_pu_targets
from rec_pu.sid8_rec_pu_integration import RecPUConfig, SIDComponentVocab, compute_native_sid8_loss


VOCAB = SIDComponentVocab(a=(1, 2, 7), b=(3, 4, 8), c=(5, 6, 9))
INDEX = {name: index for index, name in enumerate(ALL_METRIC_NAMES)}


def _target(route, start, end, a_label, b_label, c_label):
    return {
        "segment_index": 0,
        "segment_start": start,
        "segment_end": end,
        "a_label_position": a_label,
        "b_label_position": b_label,
        "c_label_position": c_label,
        "a_logit_position": a_label - 1,
        "b_logit_position": b_label - 1,
        "c_logit_position": c_label - 1,
        "positives": {"a": (1,), "b": (3,), "c": (5,)},
        "source_segment": route,
    }


def _fixture():
    labels = torch.tensor([[-100, 0, 0, 1, 3, 5, -100, 0, 2, 4, 6, -100]], dtype=torch.long)
    # Shift positions 0..10 directly correspond to labels 1..11.
    base_ce = torch.arange(1, 12, dtype=torch.float32).view(1, -1)
    logits = torch.zeros((1, 12, 16), dtype=torch.float32)
    # current a/b/c values are highest within each semantic vocabulary.
    for pos, value in ((2, 1), (3, 3), (4, 5), (7, 2), (8, 4), (9, 6)):
        logits[0, pos, value] = 9.0
    targets = [[
        _target("recommendation_cot", 1, 6, 3, 4, 5),
        _target("recommendation_nocot", 6, 11, 8, 9, 10),
    ]]
    return labels, base_ce, logits, targets


def _mean(stats, name):
    pair = stats[INDEX[name]]
    return (pair[0] / pair[1]).item() if pair[1].item() else None


def test_ce_partition_route_and_current_gold_teacher_forced_metrics():
    labels, base_ce, logits, targets = _fixture()
    stats = collect_alpha_recommendation_monitor(
        base_per_token_ce=base_ce, logits=logits, labels=labels, rec_targets=targets,
        sid_component_vocab=VOCAB, collect_tf=True,
    )
    assert _mean(stats, "a_rec_cot_body_ce") == 1.5
    assert _mean(stats, "b_rec_cot_gold_sid_ce") == 4.0
    assert _mean(stats, "c_rec_nocot_gold_sid_ce") == 9.0
    assert _mean(stats, "d_rec_gold_a_ce") == 5.5
    assert _mean(stats, "e_rec_gold_b_ce") == 6.5
    assert _mean(stats, "f_rec_gold_c_ce") == 7.5
    for name in (
        "g_rec_tf_a_hit8", "h_rec_tf_a_hit32", "i_rec_tf_b_hit8", "j_rec_tf_c_hit8",
        "k_rec_tf_chain_32_8_8", "l_rec_cot_tf_a_hit32", "m_rec_nocot_tf_a_hit32",
        "n_rec_cot_tf_chain_32_8_8", "o_rec_nocot_tf_chain_32_8_8",
    ):
        assert _mean(stats, name) == 1.0
    assert stats[INDEX["rec_monitor_cot_segments"], 0].item() == 1
    assert stats[INDEX["rec_monitor_nocot_segments"], 0].item() == 1
    assert stats[INDEX["rec_monitor_cot_gold_positions"], 0].item() == 3
    assert stats[INDEX["rec_monitor_nocot_gold_positions"], 0].item() == 3
    assert stats[INDEX["rec_monitor_missing_gold"], 0].item() == 0
    assert stats[INDEX["rec_monitor_invalid_route"], 0].item() == 0


def test_monitor_collection_cannot_change_native_loss_or_logits_gradient():
    labels = torch.tensor([[-100, 0, 1, 3, 5]], dtype=torch.long)
    weights = torch.tensor([[0.0, 1.0, 8.0, 8.0, 8.0]])
    sample_ids = torch.zeros_like(labels)
    tasks = torch.ones_like(labels)
    domains = torch.ones_like(weights)
    torch.manual_seed(7)
    monitored_logits = torch.randn(1, 5, 16, requires_grad=True)
    plain_logits = monitored_logits.detach().clone().requires_grad_(True)
    common = dict(
        labels=labels, loss_weights=weights, sample_ids=sample_ids, sample_task_ids=tasks,
        sample_domain_weights=domains, rec_pu_targets=[[ _target("recommendation_nocot", 0, 5, 2, 3, 4) ]],
        rec_pu_config=RecPUConfig(False, 0.05), sid_component_vocab=VOCAB,
    )
    monitored_loss, details = compute_native_sid8_loss(logits=monitored_logits, **common)
    collect_alpha_recommendation_monitor(
        base_per_token_ce=details.base_per_token_ce.detach(), logits=monitored_logits.detach(), labels=labels,
        rec_targets=common["rec_pu_targets"], sid_component_vocab=VOCAB, collect_tf=True,
    )
    plain_loss, _ = compute_native_sid8_loss(logits=plain_logits, **common)
    monitored_loss.backward()
    plain_loss.backward()
    assert torch.equal(monitored_loss, plain_loss)
    assert torch.equal(monitored_logits.grad, plain_logits.grad)


class _Tokenizer:
    def __init__(self):
        self.tokens = {"<|prod_begin|>": 11, "<s_a_1>": 1, "<s_b_3>": 3, "<s_c_5>": 5}
        self.reverse = {value: key for key, value in self.tokens.items()}
        self.unk_token_id = -1

    def convert_tokens_to_ids(self, token): return self.tokens.get(token, -1)
    def convert_ids_to_tokens(self, token_id): return self.reverse.get(token_id, "<unk>")
    def encode(self, token, add_special_tokens=False): return [self.convert_tokens_to_ids(token)]


def test_final_occurrence_reuses_mature_locator_not_think_occurrence():
    metadata = {
        "recommendation_group_id": "g", "recommendation_group_size": 1,
        "recommendation_all_gold_sids": ["<|prod_begin|><s_a_1><s_b_3><s_c_5>"],
        "recommendation_current_gold_sid": "<|prod_begin|><s_a_1><s_b_3><s_c_5>",
    }
    labels = [-100, 11, 1, 3, 5, 99, 11, 1, 3, 5]
    target = locate_packed_rec_pu_targets(
        labels, [PackedSegment(0, 10, "recommendation", metadata, "recommendation_cot")], _Tokenizer(),
        final_occurrence=True,
    )[0]
    assert (target.a_label_position, target.b_label_position, target.c_label_position) == (7, 8, 9)
    assert target.source_segment == "recommendation_cot"


if __name__ == "__main__":
    test_ce_partition_route_and_current_gold_teacher_forced_metrics()
    test_monitor_collection_cannot_change_native_loss_or_logits_gradient()
    test_final_occurrence_reuses_mature_locator_not_think_occurrence()
    print("alpha recommendation monitor tests: PASS")
