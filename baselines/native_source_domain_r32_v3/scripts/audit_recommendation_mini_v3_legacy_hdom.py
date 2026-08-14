#!/usr/bin/env python3
"""Read-only legacy CoT audit using ordered SID-history + target domain.

This is an intentionally conservative *candidate* search, not a conversion:
legacy rows lack recommendation_group_id, so any collision/variant is reported
as ambiguous and is never considered auto-recoverable.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

MINI = Path('/data/lf_data_versions/task_pools/懂推荐/mini_v2/recommendation_mini_v2.jsonl')
REPORT = MINI.parent / 'mini_v3_legacy_hdom_audit.json'
SID = re.compile(r'<\|(video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>')
LEGACY = [
    Path('/data/lf_data_versions/alltrain/v1_thought_prompt_all/onereason_recommendation_cot.jsonl'),
    Path('/data/lf_data_versions/alltrain/v2_recommendation_dual/onereason_recommendation_cot_v2_dual.jsonl'),
    Path('/data/lf_data_versions/alltrain/v2_recommendation_cot_complete_all/onereason_recommendation_cot.jsonl'),
    Path('/data/lf_data_versions/alltrain/v3_recommendation_multi_positive/onereason_recommendation_cot_v3_multi_positive.jsonl'),
    Path('/data/lf_data_versions/alltrain/BETA/onereason_recommendation_cot.jsonl'),
]


def target_domain(row: dict) -> str | None:
    try:
        aux = json.loads(row.get('aux_metadata_json', ''))
        value = aux.get('recommendation_current_gold_sid')
    except (TypeError, json.JSONDecodeError):
        value = row.get('recommendation_current_gold_sid')
    if not isinstance(value, str):
        matches = list(SID.finditer(str(row.get('output', ''))))
        value = matches[-1].group(0) if matches else ''
    match = SID.fullmatch(value)
    return match.group(1) if match else None


def hdom(row: dict, domain: str) -> str | None:
    text = '\n'.join(str(row.get(key, '')) for key in ('system', 'instruction', 'input', 'history'))
    history = [m.group(0) for m in SID.finditer(text)]
    if not history:
        return None
    data = json.dumps({'sid_history': history, 'target_domain': domain}, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(data.encode()).hexdigest()


def cot_body(row: dict) -> str | None:
    output = row.get('output')
    if not isinstance(output, str) or not output.startswith('<think>'):
        return None
    end = output.find('</think>')
    if end < 0:
        return None
    body = output[:end]
    return body if body.removeprefix('<think>').strip() else None


def main() -> None:
    desired: dict[str, list[dict]] = defaultdict(list)
    with MINI.open(encoding='utf-8') as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if row.get('source_segment') != 'recommendation_nocot':
                continue
            domain = target_domain(row)
            key = hdom(row, domain) if domain else None
            if key:
                aux = json.loads(row['aux_metadata_json'])
                desired[key].append({'row_index': index, 'group_id': aux['recommendation_group_id']})
    bodies: dict[str, set[str]] = defaultdict(set)
    sources: dict[str, set[str]] = defaultdict(set)
    scan = []
    for path in LEGACY:
        rows = cot = matched = 0
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                rows += 1
                row = json.loads(line)
                body = cot_body(row)
                domain = target_domain(row)
                key = hdom(row, domain) if domain else None
                if body is None or key is None:
                    continue
                cot += 1
                if key in desired:
                    bodies[key].add(body)
                    sources[key].add(str(path))
                    matched += 1
        scan.append({'path': str(path), 'rows': rows, 'valid_cot_rows': cot, 'matching_rows': matched})
    one_body = {k for k, value in bodies.items() if len(value) == 1}
    multi_body = {k for k, value in bodies.items() if len(value) > 1}
    # A hdom key mapped to more than one strict current group is not safe enough
    # to infer full-prompt group equality.
    collision = {k for k, rows in desired.items() if len({r['group_id'] for r in rows}) > 1}
    safe = one_body - collision
    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'operation': 'read_only_legacy_history_domain_candidate_search',
        'remaining_nocot_rows': sum(len(v) for v in desired.values()),
        'history_domain_keys': len(desired),
        'single_body_candidate_keys': len(one_body),
        'multi_body_ambiguous_keys': len(multi_body),
        'full_group_collision_keys': len(collision),
        'strictly_safe_candidate_keys': len(safe),
        'strictly_safe_candidate_rows': sum(len(desired[k]) for k in safe),
        'scan': scan,
        'safe_examples': [{'key': k, 'rows': desired[k], 'sources': sorted(sources[k])} for k in sorted(safe)[:20]],
        'ambiguous_examples': [{'key': k, 'rows': desired[k], 'body_variants': len(bodies[k]), 'sources': sorted(sources[k])} for k in sorted(multi_body | collision)[:20]],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
