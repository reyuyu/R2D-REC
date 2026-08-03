import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_sid_action_predictions.py"
SPEC = importlib.util.spec_from_file_location("action_evaluation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


SID_1 = "<|video_begin|><s_a_1><s_b_2><s_c_3>"
SID_2 = "<|video_begin|><s_a_1><s_b_2><s_c_4>"
SID_3 = "<|prod_begin|><s_a_5><s_b_6><s_c_7>"


def test_full_sid_parser_keeps_domain_in_identity():
    assert MODULE.full_sids(SID_1) == [("<|video_begin|>", "<s_a_1>", "<s_b_2>", "<s_c_3>")]


def test_evaluation_distinguishes_hallucination_duplication_and_wrong_legal_sid():
    result = MODULE.score_records(
        [{"prediction": SID_1 + SID_1 + SID_3, "reference": SID_2, "history": SID_1 + SID_2}]
    )
    errors = result["action_error_counts"]
    assert errors["duplicate_sid"] == 1
    assert errors["history_out_sid"] == 1
    assert errors["legal_but_wrong_sid"] == 1
    assert errors["missed_sid"] == 1


def test_action_set_precision_recall_and_count_error_are_exact():
    result = MODULE.score_records(
        [{"prediction": SID_1 + SID_2, "reference": SID_1 + SID_3, "history": SID_1 + SID_2 + SID_3}]
    )
    assert result["action_set_precision"] == 0.5
    assert result["action_set_recall"] == 0.5
    assert result["action_set_f1"] == 0.5
    assert result["action_prediction_count_error"] == 0


def test_format_and_unfinished_signals_are_reported_separately():
    result = MODULE.score_records(
        [{"prediction": "<s_a_1>", "reference": SID_1, "history": SID_1, "finish_reason": "length"}]
    )
    assert result["action_error_counts"]["format_error"] == 1
    assert result["action_error_counts"]["not_stopped"] == 1


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} Action Select evaluation tests passed.")
