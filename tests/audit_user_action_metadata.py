import argparse
import copy
import json
import time

import yaml

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.data.action_select import ActionSelectMetadataParser
from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as file:
        model_args, data_args, training_args, _, _ = get_train_args(yaml.safe_load(file))
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    action_args = copy.deepcopy(data_args)
    action_args.dataset = ["onereason_user_action_nocot_train98"]
    action_args.eval_dataset, action_args.val_size = None, 0.0
    action_args.packing, action_args.neat_packing, action_args.tokenized_path = False, False, None
    module = get_dataset(
        template,
        model_args,
        action_args,
        training_args,
        stage="sft",
        **tokenizer_module,
    )
    dataset = module["train_dataset"]
    metadata_parser = ActionSelectMetadataParser(tokenizer)
    started = time.perf_counter()
    totals = {
        "samples": 0,
        "parse_ok": 0,
        "history_sids": 0,
        "gold_sids": 0,
        "gold_in_history": 0,
        "gold_total": 0,
        "gold_duplicate_segments": 0,
    }
    errors = {}
    parse_ms = 0.0
    for item in dataset:
        metadata = metadata_parser.parse(item["input_ids"], item["labels"])
        totals["samples"] += 1
        parse_ms += float(metadata["parse_ms"])
        if not metadata["parse_valid"]:
            reason = metadata["parse_error"]
            errors[reason] = errors.get(reason, 0) + 1
            continue
        totals["parse_ok"] += 1
        totals["history_sids"] += len(metadata["history_sids"])
        totals["gold_sids"] += len(metadata["answer_sid_units"])
        totals["gold_duplicate_segments"] += int(metadata["gold_duplicate"])
        history = {tuple(sid) for sid in metadata["history_sids"]}
        for unit in metadata["answer_sid_units"]:
            totals["gold_total"] += 1
            totals["gold_in_history"] += tuple(unit["value"]) in history
    valid = totals["parse_ok"]
    result = {
        **totals,
        "parse_ok_rate": valid / totals["samples"] if totals["samples"] else 0.0,
        "history_sid_avg": totals["history_sids"] / valid if valid else 0.0,
        "gold_sid_avg": totals["gold_sids"] / valid if valid else 0.0,
        "gold_in_history_rate": (
            totals["gold_in_history"] / totals["gold_total"] if totals["gold_total"] else 0.0
        ),
        "gold_duplicate_rate": totals["gold_duplicate_segments"] / valid if valid else 0.0,
        "parser_reported_ms_per_sample": parse_ms / totals["samples"] if totals["samples"] else 0.0,
        "audit_wall_seconds": time.perf_counter() - started,
        "parse_errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
