from __future__ import annotations

import inspect
import json
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
        first_match = next(detail for group in groups for candidate in group["candidates"] for detail in candidate["match_details"])
        assert first_match["pred_index"] == 1 and first_match["gold_index"] == 1
        assert any(group['beam_all_equal'] and not group['composite_all_equal'] for group in groups)
        assert any(group['composite_all_equal'] for group in groups)
        assert any(group['top_set_tie_break'] for group in groups)
        assert any(group['strict_beam_reversal'] for group in groups)
        for group in groups:
            for candidate in group['candidates']:
                pred_indices = {unit['index'] for unit in candidate['pred_interest_units']}
                gold_indices = {unit['index'] for unit in group['gold_interest_units']}
                assert all(detail['pred_index'] in pred_indices
                           and detail['gold_index'] in gold_indices
                           for detail in candidate['match_details'])
                assert set(candidate['unmatched_pred_indices']) <= pred_indices
                assert set(candidate['unmatched_gold_indices']) <= gold_indices
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


def test_fixed_probe_ids_and_effective_max_steps_contract():
    with tempfile.TemporaryDirectory() as directory:
        run = Path(directory) / 'formal'
        run.mkdir()
        probe_ids = [f'p{index}' for index in range(1, 13)]
        (run / "manifest.json").write_text(
            json.dumps({
                "run_id": "formal",
                "experiment": "GR_REC_Think_CompositeInterest_v1",
                "fixed_probe_ids": probe_ids,
                "effective_max_steps": 716,
            }),
            encoding="utf-8",
        )
        client = TestClient(create_app(run_dir=run))
        assert client.get('/api/capabilities').json()['probes'] is True
        manifest = client.get('/api/manifest').json()
        assert manifest['effective_max_steps'] == 716
        assert manifest['max_steps'] == 716
        assert client.get('/api/runs').json()[0]['max_steps'] == 716

    runner = (Path(__file__).parents[2] / 'ablations' /
              'gr_rec_think_composite_interest_v1' /
              'run_gr_rec_think_composite_interest_v1.py').read_text(encoding='utf-8')
    formal_manifest = runner.split('monitor.write_manifest({', 1)[1].split('beam32 =', 1)[0]
    assert '"effective_max_steps": args.max_steps' in formal_manifest


def test_frontend_composite_contract_and_no_js_reward_math():
    source = (Path(__file__).parent / 'static' / 'composite_dashboard.js').read_text(encoding='utf-8')
    for marker in ('renderCompositeThinkAdvantage', '暂无实采 Composite 数据', '实采',
                   'CoT Reward 救活零方差组', 'Beam 并列第一', 'Composite 改变 Beam 排名',
                   'Candidate 额外兴趣', 'Gold 兴趣未恢复', 'Think-only 12 probes',
                   '0 / 200 / 400 / 600 / 716', 'compositeOverviewPanels',
                   'Beam 命中得分', 'CoT 兴趣命中得分', '最终 Composite 奖励',
                   'Beam 命中均值', 'CoT 兴趣命中均值（U_cot）',
                   '最终 Composite 均值 / 标准差'):
        assert marker in source
    shell = (Path(__file__).parent / 'static' / 'index.html').read_text(encoding='utf-8')
    assert "id==='probeOverview'&&!isSimpleDsr()&&state.manifest?.experiment!==" in shell
    assert "GR_REC_Think_CompositeInterest_v1" in shell
    for forbidden in ('S_text', 'S_evidence', 'maximum_weight_matching', 'population_advantages'):
        assert forbidden not in source


def test_frontend_interest_indices_provenance_and_probe_manifest_contract():
    source = (Path(__file__).parent / 'static' / 'composite_dashboard.js').read_text(encoding='utf-8')
    assert 'Pred #${m.pred_index}' in source and 'Gold #${m.gold_index}' in source
    assert '`Pred #${i}`' in source and '`Gold #${i}`' in source
    assert 'pred_index+1' not in source and 'gold_index+1' not in source
    assert 'intro.textContent="实采"' in source
    assert 'intro.textContent="复算"' in source
    assert '由已落盘 trace 只读重构，非训练时直接采集' in source
    assert '复head' not in source
    assert 'state.manifest.fixed_probe_ids||state.manifest.fixed_probe_group_ids' in source
