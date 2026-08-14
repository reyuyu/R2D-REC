#!/usr/bin/env python3
"""Read-only search for CoT bodies for mini_v2's remaining NoThink rows."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


POOL = Path('/data/lf_data_versions/task_pools/懂推荐')
MINI_V2 = POOL / 'mini_v2' / 'recommendation_mini_v2.jsonl'
REPORT = POOL / 'mini_v2' / 'mini_v3_cot_source_audit.json'

# Ordered from narrower recommendation-only pools to historical combined datasets.
CANDIDATES = [
    POOL / '不完整过滤' / 'recommendation_cot_complete.jsonl',
    POOL / '不完整过滤' / 'recommendation_all_after_incomplete_cot_filter.jsonl',
    POOL / 'beta版' / 'recommendation_beta.jsonl',
    POOL / 'alpha_mini' / 'recommendation_multipositive_video_top550_other_domains_all.jsonl',
    Path('/data/lf_data_versions/alltrain/v1_thought_prompt_all/onereason_recommendation_cot.jsonl'),
    Path('/data/lf_data_versions/alltrain/v2_recommendation_dual/onereason_recommendation_cot_v2_dual.jsonl'),
    Path('/data/lf_data_versions/alltrain/v2_recommendation_cot_complete_all/onereason_recommendation_cot.jsonl'),
    Path('/data/lf_data_versions/alltrain/v3_recommendation_multi_positive/onereason_recommendation_cot_v3_multi_positive.jsonl'),
    Path('/data/lf_data_versions/alltrain/BETA/onereason_recommendation_cot.jsonl'),
    Path('/data/lf_data_versions/alltrain/BETA_invalid_source_label_split_20260810/onereason_recommendation_cot.jsonl'),
    Path('/data/lf_data_versions/alltrain/bata_baseline_v1/onereason_bata_baseline.jsonl'),
    Path('/data/lf_data_versions/alltrain/alpha-jiankong/onereason_alpha_jiankong.jsonl'),
]

REQ = {
    'recommendation_group_id', 'recommendation_group_size',
    'recommendation_current_gold_sid', 'recommendation_all_gold_sids',
}


def parse_meta(row: dict) -> dict | None:
    if row.get('data_source') != 'recommend':
        return None
    try:
        value = json.loads(row.get('aux_metadata_json', ''))
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) and REQ.issubset(value) else None


def body(output: str) -> str | None:
    if not isinstance(output, str) or not output.startswith('<think>'):
        return None
    end = output.find('</think>')
    if end < 0:
        return None
    candidate = output[:end]
    return candidate if candidate.removeprefix('<think>').strip() else None


def main() -> None:
    remaining: list[tuple[int, str]] = []
    by_group: dict[str, list[int]] = defaultdict(list)
    with MINI_V2.open(encoding='utf-8') as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if row.get('source_segment') != 'recommendation_nocot':
                continue
            metadata = parse_meta(row)
            if metadata is None:
                raise ValueError(f'invalid mini_v2 metadata at row {index}')
            group = metadata['recommendation_group_id']
            remaining.append((index, group))
            by_group[group].append(index)

    wanted = set(by_group)
    bodies: dict[str, set[str]] = defaultdict(set)
    sources: dict[str, set[str]] = defaultdict(set)
    scan_reports = []
    for path in CANDIDATES:
        if not path.is_file():
            scan_reports.append({'path': str(path), 'status': 'missing'})
            continue
        rows = rec_rows = cot_rows = matched = 0
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                rows += 1
                row = json.loads(line)
                metadata = parse_meta(row)
                if metadata is None:
                    continue
                rec_rows += 1
                cot = body(row.get('output', ''))
                if cot is None:
                    continue
                cot_rows += 1
                group = metadata['recommendation_group_id']
                if group in wanted:
                    bodies[group].add(cot)
                    sources[group].add(str(path))
                    matched += 1
        scan_reports.append({'path': str(path), 'status': 'scanned', 'rows': rows, 'recommendation_rows': rec_rows, 'cot_rows': cot_rows, 'matching_cot_rows': matched})

    found_groups = {g for g, v in bodies.items() if len(v) == 1}
    ambiguous_groups = {g for g, v in bodies.items() if len(v) > 1}
    found_rows = sum(len(by_group[g]) for g in found_groups)
    ambiguous_rows = sum(len(by_group[g]) for g in ambiguous_groups)
    missing_groups = wanted - found_groups - ambiguous_groups
    missing_rows = sum(len(by_group[g]) for g in missing_groups)
    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'operation': 'read_only_cot_source_search',
        'mini_v2_path': str(MINI_V2),
        'remaining_nocot_rows': len(remaining),
        'remaining_nocot_groups': len(wanted),
        'recoverable_unique_cot_groups': len(found_groups),
        'recoverable_rows': found_rows,
        'ambiguous_cot_groups': len(ambiguous_groups),
        'ambiguous_rows': ambiguous_rows,
        'still_missing_groups': len(missing_groups),
        'still_missing_rows': missing_rows,
        'source_scan': scan_reports,
        'found_group_source_count': Counter(len(sources[g]) for g in found_groups),
        'ambiguous_examples': [
            {'group_id': g, 'row_indices': by_group[g], 'body_variants': len(bodies[g]), 'sources': sorted(sources[g])}
            for g in sorted(ambiguous_groups)[:20]
        ],
        'missing_examples': [
            {'group_id': g, 'row_indices': by_group[g]}
            for g in sorted(missing_groups)[:20]
        ],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
