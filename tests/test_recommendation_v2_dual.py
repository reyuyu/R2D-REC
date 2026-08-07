import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path("/app/LLaMA-Factory")
SCRIPT = ROOT / "scripts/create_recommendation_v2_dual.py"


def sid(domain, a, b, c):
    return f"<|{domain}_begin|><s_a_{a}><s_b_{b}><s_c_{c}>"


def row(prompt, value, analysis="analysis"):
    return {
        "instruction": "recommendation task",
        "input": prompt + "\n/think",
        "output": f"<think>{analysis}</think>answer: {value}",
        "history": [],
    }


with tempfile.TemporaryDirectory() as directory:
    temp_root = Path(directory)
    source = temp_root / "source.jsonl"
    rows = [
        row("product-history-A", sid("prod", 1, 2, 3)),
        row("product-history-A", sid("prod", 4, 5, 6)),
        row("product-history-A", sid("prod", 1, 2, 3)),
        row("video-history-B", sid("video", 7, 8, 9)),
        row("live-history-C", sid("living", 10, 11, 12)),
        row("live-history-C", sid("living", 13, 14, 15)),
        row("live-history-C", sid("living", 16, 17, 18)),
    ]
    with source.open("w", encoding="utf-8") as file:
        for item in rows:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    base_entry = {
        "file_name": str(source),
        "formatting": "alpaca",
        "columns": {"prompt": "instruction", "query": "input", "response": "output", "history": "history"},
    }
    dataset_info = temp_root / "dataset_info.json"
    dataset_info.write_text(json.dumps({"onereason_recommendation_cot": base_entry}), encoding="utf-8")
    version_manifest = temp_root / "versions.json"
    version_manifest.write_text(
        json.dumps({"logical_datasets": [], "versions": {"parent": {"parent": None, "overrides": {}}}}),
        encoding="utf-8",
    )
    output_root = temp_root / "output"
    command = [
        "python3",
        str(SCRIPT),
        "--source",
        str(source),
        "--output-root",
        str(output_root),
        "--dataset-info",
        str(dataset_info),
        "--version-manifest",
        str(version_manifest),
        "--version",
        "dual",
        "--parent",
        "parent",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout)
    if result.returncode:
        raise SystemExit(result.returncode)

    manifest = json.loads((output_root / "DUAL_MANIFEST.json").read_text(encoding="utf-8"))
    stats = manifest["statistics"]
    assert stats["raw_samples"] == 7
    assert stats["unique_prompt_domain_groups"] == 3
    assert stats["deduplicated_gold_total"] == 6
    assert stats["duplicate_gold_removed"] == 1
    assert stats["single_gold_groups"] == 1
    assert stats["multi_gold_groups"] == 2
    assert stats["cot_inconsistent_groups"] == 0
    assert stats["cot_samples"] + stats["nocot_samples"] == 6
    assert stats["completeness_check"] is True

    outputs = []
    for filename, mode in [
        ("onereason_recommendation_cot_v2_dual.jsonl", "cot"),
        ("onereason_recommendation_nocot_v2_dual.jsonl", "nocot"),
    ]:
        with (output_root / filename).open(encoding="utf-8") as file:
            for line in file:
                item = json.loads(line)
                outputs.append((item, mode))
                if mode == "cot":
                    assert "<think>" in item["output"] and "</think>" in item["output"]
                    assert item["input"].endswith("/think")
                else:
                    assert "<think>" not in item["output"] and "</think>" not in item["output"]
                    assert item["output"].startswith("<|")
                    assert item["input"].endswith("/no_think")
    assert len(outputs) == 6
    assert len({item["output"].split("</think>", 1)[-1].strip() for item, _ in outputs}) == 6
    print("PASS recommendation dual generation")
