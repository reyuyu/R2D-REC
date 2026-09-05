"""CPU-only end-to-end test for the isolated merge/upload worker."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    stubs = root / "stubs"
    (stubs / "modelscope" / "hub").mkdir(parents=True)
    (stubs / "torch.py").write_text("bfloat16 = 'bfloat16'\n", encoding="utf-8")
    (stubs / "transformers.py").write_text(
        """
from pathlib import Path
class Model:
    def merge_and_unload(self, **kwargs): return self
    def save_pretrained(self, path, **kwargs):
        Path(path, 'model-00001-of-00001.safetensors').write_bytes(b'merged')
class AutoModelForCausalLM:
    @classmethod
    def from_pretrained(cls, *args, **kwargs): return Model()
class Tokenizer:
    def save_pretrained(self, path): Path(path, 'tokenizer.json').write_text('{}')
class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs): return Tokenizer()
""".strip() + "\n",
        encoding="utf-8",
    )
    (stubs / "peft.py").write_text(
        "class PeftModel:\n    @classmethod\n    def from_pretrained(cls, model, *args, **kwargs): return model\n",
        encoding="utf-8",
    )
    (stubs / "modelscope" / "__init__.py").write_text("", encoding="utf-8")
    (stubs / "modelscope" / "hub" / "__init__.py").write_text("", encoding="utf-8")
    (stubs / "modelscope" / "hub" / "api.py").write_text(
        """
import json, os
from pathlib import Path
class HubApi:
    def repo_exists(self, repo_id, **kwargs):
        assert kwargs['token'] == 'secret-test-token'
        return False
    def create_model(self, model_id, **kwargs):
        assert kwargs['visibility'] == 1 and kwargs['token'] == 'secret-test-token'
    def upload_folder(self, **kwargs):
        assert kwargs['token'] == 'secret-test-token'
        Path(os.environ['FAKE_UPLOAD_RECORD']).write_text(json.dumps({
            'repo_id': kwargs['repo_id'],
            'files': sorted(p.name for p in Path(kwargs['folder_path']).iterdir()),
        }))
""".strip() + "\n",
        encoding="utf-8",
    )

    base = root / "base"
    adapter = root / "adapter"
    base.mkdir()
    adapter.mkdir()
    (base / "model.safetensors").write_bytes(b"full-sft-parent")
    (base / "config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"grpo-adapter")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    base_sha = sha256(base / "model.safetensors")
    adapter_sha = sha256(adapter / "adapter_model.safetensors")
    (adapter / "lineage.json").write_text(json.dumps({
        "schema": "grpo_adapter_lineage_v1",
        "parent_mode": "full_model",
        "parent_base_sha256": base_sha,
    }), encoding="utf-8")
    token_file = root / "token"
    token_file.write_text("secret-test-token", encoding="utf-8")
    if os.name != "nt":
        token_file.chmod(0o600)
    status = root / "status.json"
    upload_record = root / "upload.json"
    environment = {
        **os.environ,
        "PYTHONPATH": str(stubs),
        "FAKE_UPLOAD_RECORD": str(upload_record),
    }
    command = [
        sys.executable, str(HERE / "publish_merged_model.py"),
        "--base-model", str(base),
        "--adapter", str(adapter),
        "--expected-base-sha256", base_sha,
        "--expected-adapter-sha256", adapter_sha,
        "--run-id", "formal-run",
        "--checkpoint", "checkpoint-100",
        "--model-id", "owner/model",
        "--visibility", "private",
        "--token-file", str(token_file),
        "--work-dir", str(root / "work"),
        "--status-file", str(status),
    ]
    completed = subprocess.run(command, env=environment, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    final_status = json.loads(status.read_text(encoding="utf-8"))
    assert final_status["state"] == "completed" and final_status["progress"] == 100
    assert final_status["model_url"] == "https://modelscope.cn/models/owner/model"
    assert not (root / "work" / "merged-model").exists()
    uploaded = json.loads(upload_record.read_text(encoding="utf-8"))
    assert uploaded["repo_id"] == "owner/model"
    assert {"configuration.json", "MERGE_MANIFEST.json", "model-00001-of-00001.safetensors"} <= set(uploaded["files"])
    assert "secret-test-token" not in status.read_text(encoding="utf-8")

    continued_adapter = root / "continued-adapter"
    continued_adapter.mkdir()
    (continued_adapter / "adapter_model.safetensors").write_bytes(b"grpo1-plus-grpo2-adapter")
    (continued_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    continued_sha = sha256(continued_adapter / "adapter_model.safetensors")
    (continued_adapter / "lineage.json").write_text(json.dumps({
        "schema": "grpo2_continued_adapter_lineage_v1",
        "adapter_only": True,
        "adapter_continuation": True,
        "adapter_semantics": "CONTINUED_SINGLE_ADAPTER",
        "contains_grpo1_and_grpo2_effect": True,
        "base_full_model_sha256": base_sha,
        "adapter_initialization_source_stage": "GRPO1_REC_BILATERAL",
    }), encoding="utf-8")
    continued_status = root / "continued-status.json"
    continued_upload_record = root / "continued-upload.json"
    continued_command = [
        sys.executable, str(HERE / "publish_merged_model.py"),
        "--base-model", str(base),
        "--adapter", str(continued_adapter),
        "--expected-base-sha256", base_sha,
        "--expected-adapter-sha256", continued_sha,
        "--run-id", "grpo2-continued-run",
        "--checkpoint", "checkpoint-200",
        "--model-id", "owner/grpo2-continued",
        "--visibility", "private",
        "--token-file", str(token_file),
        "--work-dir", str(root / "continued-work"),
        "--status-file", str(continued_status),
    ]
    continued_environment = {**environment, "FAKE_UPLOAD_RECORD": str(continued_upload_record)}
    continued = subprocess.run(continued_command, env=continued_environment, text=True, capture_output=True)
    assert continued.returncode == 0, continued.stderr
    continued_result = json.loads(continued_status.read_text(encoding="utf-8"))
    assert continued_result["state"] == "completed"
    continued_uploaded = json.loads(continued_upload_record.read_text(encoding="utf-8"))
    assert continued_uploaded["repo_id"] == "owner/grpo2-continued"
    assert "secret-test-token" not in continued_status.read_text(encoding="utf-8")
    print("ALL MODEL PUBLISH WORKER CPU TESTS PASSED")
