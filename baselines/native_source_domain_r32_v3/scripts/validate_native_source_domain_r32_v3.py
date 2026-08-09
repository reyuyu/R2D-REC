import inspect
import json
from importlib.metadata import version
from pathlib import Path

import liger_kernel
from liger_kernel.transformers import apply_liger_kernel_to_qwen3

from llamafactory.hparams import get_train_args


root = Path("/data/baselines/native_source_domain_r32_v3")
manifest = json.loads((root / "dataset" / "manifest.json").read_text(encoding="utf-8"))
assert manifest["records"] == sum(manifest["source_counts"].values())
assert manifest["world_included"] is False
assert set(manifest["material_domain_weights"]) == {"video", "prod", "ad", "living"}
for config_name in ("train_native_source_domain_r32_v3_2epoch.yaml", "smoke_2gpu_5steps.yaml"):
    config = root / "config" / config_name
    try:
        get_train_args(json.loads(json.dumps(__import__("yaml").safe_load(config.read_text(encoding="utf-8")))))
    except ValueError as error:
        if "Please launch distributed training" not in str(error):
            raise
print(f"PASS source-domain manifest records={manifest['records']}")
print(f"PASS liger={version('liger-kernel')} qwen3_signature={inspect.signature(apply_liger_kernel_to_qwen3)}")
print("PASS formal and 2-GPU smoke config parse")
