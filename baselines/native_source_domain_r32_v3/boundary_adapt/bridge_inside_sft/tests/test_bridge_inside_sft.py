from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT.parent))
from bridge_inside_sft import build_dataset as builder  # noqa: E402
from bridge_inside_sft import common  # noqa: E402


def test_extract_response_preserves_exact_bridge():
    response = "<think>cot</think>\n精确桥: <|video_begin|><s_a_1><s_b_2><s_c_3>"
    cot, bridge, abc = common.extract_response(response, "video")
    assert cot == "<think>cot"
    assert bridge == "\n精确桥: "
    assert abc == "<s_a_1><s_b_2><s_c_3>"


def test_prompt_family_has_three_standard_and_other():
    for index, ending in enumerate(common.STANDARD_USER_ENDINGS):
        assert common.prompt_family("prefix\n" + ending) == f"standard_{index}"
    assert common.prompt_family("unusual /think") == "other"


def test_group_mean_token_ce_is_not_flat_token_mean():
    # Row 0 has two labels and row 1 has four. Each row must retain weight 1/2.
    torch.manual_seed(3)
    logits = torch.randn(2, 6, 13, dtype=torch.float64, requires_grad=True)
    labels = torch.full((2, 6), -100, dtype=torch.long)
    labels[0, 4:] = torch.tensor([1, 2])
    labels[1, 2:] = torch.tensor([3, 4, 5, 6])
    actual = common.transition_group_loss(logits, labels)
    rows = []
    for index in range(2):
        mask = labels[index, 1:].ne(-100)
        rows.append(F.cross_entropy(logits[index, :-1][mask], labels[index, 1:][mask], reduction="mean"))
    expected = torch.stack(rows).mean()
    flat = F.cross_entropy(logits[:, :-1].reshape(-1, 13), labels[:, 1:].reshape(-1), ignore_index=-100)
    assert torch.allclose(actual, expected, rtol=0, atol=1e-12)
    assert not torch.allclose(actual, flat, rtol=0, atol=1e-5)
    assert torch.allclose(torch.autograd.grad(actual, logits, retain_graph=True)[0], torch.autograd.grad(expected, logits)[0], rtol=0, atol=1e-12)


def test_emission_requires_immediate_bridge_close_adjacency():
    exact = [4, 5]
    known = {"video": exact, "prod": [7]}
    good = common.emission_metrics([1, 4, 5, 9], 9, exact, known, "video")
    gap = common.emission_metrics([1, 4, 5, 6, 9], 9, exact, known, "video")
    wrong = common.emission_metrics([1, 7, 9], 9, exact, known, "video")
    assert good["exact_bridge_before_close"]
    assert not gap["exact_bridge_before_close"] and gap["bridge_not_adjacent_to_close"]
    assert wrong["wrong_domain_bridge"]


def test_split_forces_historical_without_overlap():
    rows = []
    for domain in common.DOMAIN_ORDER:
        for index in range(100):
            rows.append({"group_id": f"{domain}-{index}", "target_domain": domain})
    historical = {f"{domain}-99" for domain in common.DOMAIN_ORDER}
    train, holdout = builder.split_rows(rows, historical)
    train_ids, holdout_ids = {row["group_id"] for row in train}, {row["group_id"] for row in holdout}
    assert len(holdout) == 40
    assert not train_ids & holdout_ids
    assert historical <= holdout_ids
