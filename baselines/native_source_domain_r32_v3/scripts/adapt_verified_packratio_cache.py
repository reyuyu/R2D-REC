from pathlib import Path
from datasets import Dataset, DatasetDict, concatenate_datasets

source = Path('/root/.cache/huggingface/datasets/json/default-e95229b07fd4770d/0.0.0/f4e89e8750d5d5ffbef2c078bf0ddfedef29dc2faff52a6255cf513c05eb1092')
target = Path('/data/tokenized/onereason_beta_material_aligned_packratio_verified_v1')
shards = sorted(source.glob('cache-e90f0006044933d3_*.arrow'))
dataset = concatenate_datasets([Dataset.from_file(str(shard)) for shard in shards])
required = {'input_ids','attention_mask','position_ids','labels','loss_weights','sample_ids','sample_task_ids','sample_domain_weights','pack_task_id','rec_pu_targets_json'}
if len(dataset) != 33616 or not required.issubset(dataset.column_names):
    raise RuntimeError(f'Unexpected source packed cache: {len(dataset)} {dataset.column_names}')
if target.exists():
    raise RuntimeError(f'Refusing to overwrite existing adapter: {target}')
DatasetDict({'train': dataset}).save_to_disk(str(target))
loaded = DatasetDict.load_from_disk(str(target))['train']
if len(loaded) != len(dataset) or loaded.column_names != dataset.column_names:
    raise RuntimeError('Adapter schema or row count mismatch')
for index in (0, len(dataset)//2, len(dataset)-1):
    for column in ('input_ids','labels','sample_ids','sample_task_ids','loss_weights','pack_task_id','rec_pu_targets_json'):
        if loaded[index][column] != dataset[index][column]:
            raise RuntimeError(f'Adapter content mismatch at row={index}, column={column}')
print({'path': str(target), 'rows': len(loaded), 'columns': loaded.column_names, 'fingerprint': loaded._fingerprint})
