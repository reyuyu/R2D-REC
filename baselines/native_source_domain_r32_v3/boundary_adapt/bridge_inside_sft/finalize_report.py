"""Finalize the 50-step heldout gate without launching continuation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def checkpoint_view(value: dict) -> dict:
    self_cot = value["self_cot"]
    bare = value["modes"]["self_cot_bare"]
    injected = value["modes"]["injected_before"]
    return {
        "bridge_rate": self_cot["exact_bridge_before_close_rate"],
        "closed_rate": self_cot["closed_rate"],
        "self_cot_bare_raw": bare["beam_raw"],
        "self_cot_exact": bare["exact"],
        "self_cot_ab": bare["ab"],
        "self_cot_a": bare["a"],
        "self_cot_invalid": bare["invalid"],
        "injected_before_raw": injected["beam_raw"],
        "injected_gold_nll": injected["mean_gold_abc_nll"],
        "transition_nll": value["proposed_transition_nll"],
        "history_exact_copy": bare["history_exact_copy"],
        "history_ab_copy": bare["history_ab_copy"],
        "history_a_copy": bare["history_a_copy"],
        "novel": bare["novel"],
        "gold_and_history": bare["gold_and_history"],
        "history_not_gold": bare["history_not_gold"],
        "gold_not_history": bare["gold_not_history"],
        "per_domain": value["per_domain"],
    }


def finalize(args) -> dict:
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    preflight = json.loads(Path(args.preflight).read_text(encoding="utf-8"))
    training = json.loads(Path(args.training).read_text(encoding="utf-8"))
    evaluations = {
        label: json.loads(Path(path).read_text(encoding="utf-8"))
        for label, path in (("0", args.eval0), ("10", args.eval10), ("25", args.eval25), ("50", args.eval50))
    }
    views = {label: checkpoint_view(value) for label, value in evaluations.items()}
    d0, t50 = views["0"], views["50"]
    gates = {
        "A_bridge_emission": t50["bridge_rate"] - d0["bridge_rate"] >= .20 or t50["bridge_rate"] >= .50,
        "B_closure": t50["closed_rate"] >= d0["closed_rate"] - .03,
        "C_decoder_preservation": t50["injected_before_raw"] >= .95 * d0["injected_before_raw"],
        "D_history_copy": t50["history_exact_copy"] >= d0["history_exact_copy"] - .10,
        "E_self_cot_bare": t50["self_cot_bare_raw"] >= d0["self_cot_bare_raw"] - .10,
    }
    continuation_gate = all(gates.values())
    candidates = ("10", "25", "50")
    best = max(candidates, key=lambda label: (
        views[label]["self_cot_bare_raw"],
        views[label]["bridge_rate"],
        views[label]["injected_before_raw"] / d0["injected_before_raw"],
        views[label]["closed_rate"],
    ))
    emission_signal = gates["A_bridge_emission"]
    decoder_preservation = t50["injected_before_raw"] / d0["injected_before_raw"]
    nll_declined = t50["transition_nll"] < .9 * d0["transition_nll"]
    if emission_signal and decoder_preservation < .95:
        root_class = "TRANSITION_TRAINING_DAMAGES_DECODER"
        conclusion = "Exact bridge emission increased, but the injected Fresh-D0-CoT decoder preservation gate failed."
    elif emission_signal and t50["self_cot_bare_raw"] > d0["self_cot_bare_raw"] and decoder_preservation >= .95:
        root_class = "TRANSITION_MIGRATION_WORKS"
        conclusion = "The model emits bridge-before-close more often, heldout official-like BareRaw improves, and injected decoder behavior is preserved."
    elif emission_signal and nll_declined:
        root_class = "TRANSITION_LEARNED_BUT_REC_NOT_IMPROVED"
        conclusion = "Transition NLL and exact bridge emission improve, but adaptation-heldout official-like BareRaw does not improve."
    elif nll_declined:
        root_class = "TEACHER_FORCED_TRANSITION_DOES_NOT_GENERALIZE_TO_SELF_COT"
        conclusion = "Teacher-forced transition NLL declines without enough exact bridge-before-close emission in sampled Self-CoT."
    else:
        root_class = "INSUFFICIENT_TRANSITION_LEARNING"
        conclusion = "Fifty optimizer steps do not produce enough transition-learning signal under the fixed gates."

    train_ranks = training["ranks"]
    preflight_pass = all(
        rank["init_parity"]["max_abs_diff"] == 0
        and rank["init_parity"]["mean_abs_diff"] == 0
        and rank["manual_loss_parity"]["pass"]
        and rank["zero_update"]["pass"]
        and rank["system_preserve_pass"]
        and rank["train_holdout_intersection"] == 0
        for rank in preflight["ranks"]
    )
    payload = {
        "type": "bridge_inside_transition_sft_v1",
        "source_commit": args.source_commit,
        "implementation_commit": args.implementation_commit,
        "split": split,
        "preflight_pass": preflight_pass,
        "preflight": preflight,
        "training": training,
        "evaluations": evaluations,
        "checkpoint_views": views,
        "step50_continuation_gates": gates,
        "step50_continuation_gate": continuation_gate,
        "continued_to_200": False,
        "best_heldout_checkpoint": best,
        "best_heldout_adapter": str(Path(args.checkpoint_root) / f"checkpoint-{best}"),
        "best_heldout_self_cot_bare_raw": views[best]["self_cot_bare_raw"],
        "best_exact_bridge_rate": views[best]["bridge_rate"],
        "best_injected_decoder_preservation": views[best]["injected_before_raw"] / d0["injected_before_raw"],
        "base_changed": any(rank["base_changed"] for rank in train_ranks),
        "only_existing_lora_trained": all(rank["only_existing_lora_trainable"] for rank in train_ranks),
        "second_lora_created": any(rank["second_lora_created"] for rank in train_ranks),
        "old_optimizer_resumed": any(rank["old_optimizer_resumed"] for rank in train_ranks),
        "training_started": True,
        "optimizer_steps": 50,
        "external_eval_started": False,
        "root_class": root_class,
        "root_conclusion": conclusion,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Bridge-Inside Transition SFT V1",
        "",
        "This is an adaptation-heldout mechanism/training result, not a globally-unseen or external benchmark result.",
        "",
        "| Step | Bridge rate | Closed rate | Self-CoT BareRaw | Injected Raw | Transition NLL | History exact copy |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("0", "10", "25", "50"):
        row = views[label]
        lines.append(f"| {label} | {row['bridge_rate']:.6f} | {row['closed_rate']:.6f} | {row['self_cot_bare_raw']:.6f} | {row['injected_before_raw']:.6f} | {row['transition_nll']:.6f} | {row['history_exact_copy']:.6f} |")
    lines += [
        "", "## Step-50 gates", "", "```json", json.dumps(gates, indent=2), "```",
        "", "## Result", "", f"`{root_class}`", "", conclusion,
        "", "Continuation was not launched. External evaluation was not launched.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    review = [
        "BRIDGE-INSIDE TRANSITION SFT V1 REVIEW",
        "",
        "SCOPE=ADAPTATION_HELDOUT; NOT GLOBALLY UNSEEN; NO EXTERNAL EVAL",
        "",
        "[SPLIT]", json.dumps(split, ensure_ascii=False, indent=2),
        "", "[PREFLIGHT]", json.dumps(preflight, ensure_ascii=False, indent=2),
        "", "[TRAINING]", json.dumps(training, ensure_ascii=False, indent=2),
        "", "[CHECKPOINT VIEWS]", json.dumps(views, ensure_ascii=False, indent=2),
        "", "[FULL EVALUATIONS]", json.dumps(evaluations, ensure_ascii=False, indent=2),
        "", "[STEP50 CONTINUATION GATE]", json.dumps(gates, indent=2),
        "", f"ROOT_CLASS={root_class}", f"ROOT_CONCLUSION={conclusion}",
        "CONTINUED_TO_200=NO", "EXTERNAL_EVAL_STARTED=NO",
    ]
    (output / "CHATGPT_BRIDGE_INSIDE_SFT_REVIEW.txt").write_text("\n".join(review) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--training", required=True)
    parser.add_argument("--eval0", required=True)
    parser.add_argument("--eval10", required=True)
    parser.add_argument("--eval25", required=True)
    parser.add_argument("--eval50", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    value = finalize(args)
    print(f"ROOT_CLASS={value['root_class']}")
    print(f"STEP50_CONTINUATION_GATE={'PASS' if value['step50_continuation_gate'] else 'FAIL'}")
    print("CONTINUED_TO_200=NO")


if __name__ == "__main__":
    main()
