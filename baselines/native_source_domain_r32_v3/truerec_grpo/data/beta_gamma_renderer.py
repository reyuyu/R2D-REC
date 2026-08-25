"""Pinned CPU tokenizer and SFT renderer for the Beta-Gamma run."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
from typing import Any


BETA_GAMMA_RUN = Path(
    "/data/outputs/baselines/native_source_domain_r32_v3/"
    "BETA-GAMMA-R32-2E-GC04-4GPU-20260825-054127"
)
LLAMAFACTORY_REFERENCE = Path("/data/reference/llamafactory-01398eb")
LLAMAFACTORY_COMMIT = "01398eb18dd475a6e27c36f15b970aeacf0d4a60"
TEMPLATE_RELATIVE = Path("src/llamafactory/data/template.py")
CONVERTER_RELATIVE = Path("src/llamafactory/data/converter.py")
TEMPLATE_SHA256 = "68f289bb28c434e7d26cb36bd4351ea4636f9e102bf374d9a84f642565d9f84e"
CONVERTER_SHA256 = "c596efb447fb50fc915f6aa5c4f228d5409fba83808e23015a1bcf3387a3ffbf"
TOKENIZER_JSON_SHA256 = "0be9ef1be1eac715f9a14a78277f70b3f0596d0cb61a8dbc8fc4aad2abb20a9a"
TOKENIZER_CONFIG_SHA256 = "47fb9a938aadb92d2023e81b4b4eca16772853d8a4da9654f1444e0e2c89373a"
CHAT_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
SOURCE_CONFIG_SHA256 = "b6286f6e96f6dd8a34c816ffa93aa0ddfd7bec7c927fa14aec3c2ab2f83d7b5a"
TEMPLATE_NAME = "qwen3_nothink"
EMPTY_THINK = "<think>\n\n</think>\n"


class RendererError(RuntimeError):
    pass


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def join_alpaca_user(instruction: str, input_text: str) -> str:
    """Match AlpacaDatasetConverter: join non-empty prompt/query with one newline."""
    return "\n".join(value for value in (instruction, input_text) if value)


class BetaGammaRenderer:
    def __init__(self) -> None:
        source_dir = LLAMAFACTORY_REFERENCE / "src"
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        commit = subprocess.check_output(
            ["git", "-C", str(LLAMAFACTORY_REFERENCE), "rev-parse", "HEAD"],
            text=True,
            encoding="utf-8",
        ).strip()
        template_sha = file_sha(LLAMAFACTORY_REFERENCE / TEMPLATE_RELATIVE)
        converter_sha = file_sha(LLAMAFACTORY_REFERENCE / CONVERTER_RELATIVE)
        if commit != LLAMAFACTORY_COMMIT:
            raise RendererError(f"LLAMAFACTORY_COMMIT_FAIL={commit}")
        if template_sha != TEMPLATE_SHA256 or converter_sha != CONVERTER_SHA256:
            raise RendererError(
                f"LLAMAFACTORY_SOURCE_SHA_FAIL={template_sha},{converter_sha}"
            )
        run_files = {
            "tokenizer.json": TOKENIZER_JSON_SHA256,
            "tokenizer_config.json": TOKENIZER_CONFIG_SHA256,
            "chat_template.jinja": CHAT_TEMPLATE_SHA256,
            "metadata/source_config.yaml": SOURCE_CONFIG_SHA256,
        }
        actual_run_files = {
            name: file_sha(BETA_GAMMA_RUN / name) for name in run_files
        }
        if actual_run_files != run_files:
            raise RendererError(
                f"BETA_GAMMA_RENDERER_FILE_SHA_FAIL={actual_run_files}"
            )
        from transformers import AutoTokenizer
        from llamafactory.data.template import TEMPLATES

        self.tokenizer = AutoTokenizer.from_pretrained(
            BETA_GAMMA_RUN, local_files_only=True, trust_remote_code=True
        )
        self.template = TEMPLATES[TEMPLATE_NAME]
        self.template.fix_special_tokens(self.tokenizer)
        self.provenance = {
            "beta_gamma_run": str(BETA_GAMMA_RUN),
            "tokenizer_json": str(BETA_GAMMA_RUN / "tokenizer.json"),
            "tokenizer_json_sha256": actual_run_files["tokenizer.json"],
            "tokenizer_config_sha256": actual_run_files["tokenizer_config.json"],
            "exported_chat_template": str(BETA_GAMMA_RUN / "chat_template.jinja"),
            "exported_chat_template_sha256": actual_run_files["chat_template.jinja"],
            "source_config_sha256": actual_run_files["metadata/source_config.yaml"],
            "training_renderer": f"LLaMA-Factory {TEMPLATE_NAME}",
            "llamafactory_reference": str(LLAMAFACTORY_REFERENCE),
            "llamafactory_commit": commit,
            "template_py_sha256": template_sha,
            "converter_py_sha256": converter_sha,
            "alpaca_user_join": "newline join of non-empty instruction and input",
        }

    def prompt_ids(self, system: str, user_content: str) -> list[int]:
        prompt_ids, _ = self.template.encode_oneturn(
            self.tokenizer,
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": ""},
            ],
            system,
        )
        return prompt_ids

    def assistant_ids(self, response: str) -> list[int]:
        elements = self.template.format_assistant.apply(content=response)
        return self.template._convert_elements_to_ids(self.tokenizer, elements)

    def full_sft_ids(self, system: str, user_content: str, response: str) -> list[int]:
        return self.prompt_ids(system, user_content) + self.assistant_ids(response)

    def rl_context_ids(self, system: str, user_content: str, fixed_domain_token: str) -> list[int]:
        return self.prompt_ids(system, user_content) + self.tokenizer.encode(
            EMPTY_THINK + fixed_domain_token, add_special_tokens=False
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def audit(self) -> dict[str, Any]:
        return dict(self.provenance)
