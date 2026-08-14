import json
from pathlib import Path

p = Path('/data/lf_data_versions/alltrain/mini_v2/manifest.json')
m = json.loads(p.read_text(encoding='utf-8'))
m['validation'].update({
    'output_multiset_equals_all_atomic_inputs': True,
    'all_aux_metadata_json_parseable': False,
    'all_aux_metadata_json_parseable_note': (
        'Recommendation rows are JSON-parseable with all required keys. '
        'The 39298 non-recommendation rows retain the source schema empty string '
        'aux_metadata_json; changing them to {} would violate byte/value preservation.'
    ),
})
for item in m.get('aggregate_files_not_reingested', []):
    q = Path(item['path'])
    item['rows'] = sum(1 for line in q.open(encoding='utf-8') if line.strip())
p.write_text(json.dumps(m, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(m['validation'], ensure_ascii=False, indent=2))
