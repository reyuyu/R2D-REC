from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).parents[1]
TRAINER = ROOT.parents[0] / "trainer"
sys.path[:0] = [str(ROOT), str(TRAINER)]
from think_rescue_v1 import A_RESCUE_REDUCTION, uniform_positive_soft_ce  # noqa: E402
from truerec_loss_v1 import multi_positive_log_mass_loss  # noqa: E402


@pytest.mark.parametrize("targets", [(0, 2), (0, 2, 4)])
def test_uniform_positive_soft_ce_matches_manual_mean(targets):
    logits = torch.tensor([0.2, -0.4, 1.1, 0.7, -0.3], dtype=torch.float64)
    expected = -torch.log_softmax(logits, dim=-1)[list(targets)].mean()
    torch.testing.assert_close(uniform_positive_soft_ce(logits, targets), expected)


def test_uniform_soft_ce_is_not_positive_log_mass():
    logits = torch.tensor([2.0, 0.0, -1.0, 1.0], dtype=torch.float64)
    soft_ce = uniform_positive_soft_ce(logits, [0, 2])
    log_mass = multi_positive_log_mass_loss(logits, [0, 2])
    assert not torch.isclose(soft_ce, log_mass)


def test_targets_are_deduplicated_and_reduction_is_group_level():
    logits = torch.tensor([0.0, 1.0, 2.0])
    torch.testing.assert_close(
        uniform_positive_soft_ce(logits, [0, 0, 2]),
        uniform_positive_soft_ce(logits, [0, 2]),
    )
    assert A_RESCUE_REDUCTION == "one_group_level_A_distribution"


@pytest.mark.parametrize("targets,error", [([], ValueError), ([-1], IndexError), ([3], IndexError)])
def test_invalid_target_contract(targets, error):
    with pytest.raises(error):
        uniform_positive_soft_ce(torch.zeros(3), targets)
