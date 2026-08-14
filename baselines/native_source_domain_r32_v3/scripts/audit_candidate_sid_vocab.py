"""Read tokenizer once to report legal a/b/c component-vocabulary sizes."""

from __future__ import annotations

from pathlib import Path
import sys

BASELINE_ROOT = Path("/data/baselines/native_source_domain_r32_v3")
REFERENCE_SRC = Path("/data/reference/llamafactory-01398eb/src")
sys.path[:0] = [str(BASELINE_ROOT), str(REFERENCE_SRC)]

from llamafactory.hparams import DataArguments, ModelArguments
from llamafactory.model import load_tokenizer
from rec_pu.sid8_rec_pu_integration import build_sid_component_vocab


def main():
    tokenizer = load_tokenizer(ModelArguments(model_name_or_path="/data/models/onereason-8b-pretrain-competition"))["tokenizer"]
    vocab = build_sid_component_vocab(tokenizer)
    print(f"SID_COMPONENT_VOCAB_SIZES=a={len(vocab.a)} b={len(vocab.b)} c={len(vocab.c)}")


if __name__ == "__main__":
    main()
