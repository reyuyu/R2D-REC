from __future__ import annotations

import inspect
import math
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

try:
    from .composite_interest_adapter import adapt_events, is_composite_manifest
    from .generate_composite_demo_run import generate
    from .server import create_app, monitor_advantage_formula, read_jsonl
except ImportError:
    from composite_interest_adapter import adapt_events, is_composite_manifest
    from generate_composite_demo_run import generate
    from server import create_app, monitor_advantage_formula, read_jsonl


def demo_client():
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name)
    generate(root)
    return temporary, TestClient(create_app(run_dir=root / "GR-REC-THINK-COMPOSITE-DEMO"))


def test_exact_manifest_capability_and_captured_advantage():
    temporary, client = demo_client()
    try:
        capability = client.get('/api/capabilities').json()
        assert capability['composite_interest'] is True
        assert capability['advantage_source'] == 'captured'
        payload = client.get('/api/advantages').json()
        assert payload['formula'] == 'composite_interest_v1'
        assert payload['provenance']['mode'] == 'captured'
        assert len(payload['groups']) == 4
    finally:
        temporary.cleanup()


def test_composite_filters_and_alignment():
    temporary, client = demo_client()
    try:
        payload = client.get('/api/composite-interest?from_step=200&to_step=400').json()
        assert [group['step'] for group in payload['groups']] == [200, 400]
        selected = client.get('/api/composite-interest?group_id=demo-rescued').json()['groups']
        assert len(selected) == 1 and len(selected[0]['candidates']) == 4
        for candidate in selected[0]['candidates']:
            assert math.isclose(candidate['beam_contribution'] + candidate['cot_contribution'],
                                candidate['composite_reward'], abs_tol=1e-9)
    finally:
        temporary.cleanup()


def test_matching_indices_unmatched_and_demo_cases():
    temporary, client = demo_client()
    try:
        groups = client.get('/api/composite-interest').json()['groups']
        assert any(group['beam_all_equal'] and not group['composite_all_equal'] for group in groups)
        assert any(group['composite_all_equal'] for group in groups)
        assert any(group['top_set_tie_break'] for group in groups)
        assert any(group['strict_beam_reversal'] for group in groups)
        for group in groups:
            for candidate in group['candidates']:
                assert all(detail['pred_index'] < len(candidate['pred_interest_units'])
                           and detail['gold_index'] < len(group['gold_interest_units'])
                           for detail in candidate['match_details'])
                assert all(index < len(candidate['pred_interest_units'])
                           for index in candidate['unmatched_pred_indices'])
                assert all(index < len(group['gold_interest_units'])
                           for index in candidate['unmatched_gold_indices'])
    finally:
        temporary.cleanup()


def test_probe_api_has_twelve_groups_and_milestones():
    temporary, client = demo_client()
    try:
        rows = client.get('/api/probes').json()
        assert len({row['group_id'] for row in rows}) == 12
        assert {row['step'] for row in rows} == {0, 200, 400, 600, 716}
        assert all(row['probe_route'] == 'think_only' for row in rows)
    finally:
        temporary.cleanup()


def test_legacy_formula_selection_remains_reconstructed():
    assert monitor_advantage_formula({'experiment': 'GR_REC_NoThinkOnly_Frontier_v1'}) == 'frontier_v1'
    assert monitor_advantage_formula({'runner': 'x/gr_rec_think_exact_clamp_v1/run_think_exact_clamp_train.py'}) == 'clamp_bridge_v1'
    assert monitor_advantage_formula({'experiment': 'GR_REC_DSR_Ablation_v1'}) is None
    assert is_composite_manifest({'experiment': 'almost-composite'}) is False


def test_server_has_no_gold_source_path_reader():
    try:
        from . import server
    except ImportError:
        import server
    source = inspect.getsource(server)
    assert 'gold_source_path' not in source


def test_frontend_composite_contract_and_no_js_reward_math():
    source = (Path(__file__).parent / 'static' / 'composite_dashboard.js').read_text(encoding='utf-8')
    for marker in ('renderCompositeThinkAdvantage', '暂无实采 Composite 数据', '实采',
                   'CoT Reward 救活零方差组', 'Beam 并列第一', 'Composite 改变 Beam 排名',
                   'Candidate 额外兴趣', 'Gold 兴趣未恢复', 'Think-only 12 probes',
                   '0 / 200 / 400 / 600 / 716', 'compositeOverviewPanels'):
        assert marker in source
    for forbidden in ('S_text', 'S_evidence', 'maximum_weight_matching', 'population_advantages'):
        assert forbidden not in source
