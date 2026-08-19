import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from user_generated_token_projection import project_compiled_mask_to_generated


def _compiled(input_ids, masks, records=None):
    union = [any(mask[index] for mask in masks.values()) for index in range(len(input_ids))]
    return {
        "input_ids": input_ids,
        "token_count": len(input_ids),
        "per_kind_masks": masks,
        "penalty_mask": union,
        "records": records or [],
        "masked_token_count": sum(union),
        "masked_token_fraction": sum(union) / len(input_ids),
    }


def test_identity_projection_preserves_masks():
    compiled = _compiled([1, 2, 3], {"duplicate_sid": [False, True, False]})
    output = project_compiled_mask_to_generated(compiled, [1, 2, 3])
    assert output["penalty_mask"] == [False, True, False]
    assert output["tokenization_projection"]["required"] is False


def test_split_projection_masks_all_replacement_tokens():
    compiled = _compiled([1, 20, 3], {"hallucinated_sid": [False, True, False]})
    output = project_compiled_mask_to_generated(compiled, [1, 21, 22, 3])
    assert output["penalty_mask"] == [False, True, True, False]
    assert output["tokenization_projection"]["required"] is True


def test_merge_projection_preserves_overlap_and_record_indices():
    records = [
        {
            "included": True,
            "masked_token_indices": [1, 2],
            "token_spans": [{"token_start": 1, "token_end": 3, "token_ids": [21, 22]}],
        }
    ]
    compiled = _compiled(
        [1, 21, 22, 3],
        {
            "hallucinated_sid": [False, True, True, False],
            "duplicate_sid": [False, False, True, False],
        },
        records,
    )
    output = project_compiled_mask_to_generated(compiled, [1, 20, 3])
    assert output["per_kind_masks"]["hallucinated_sid"] == [False, True, False]
    assert output["per_kind_masks"]["duplicate_sid"] == [False, True, False]
    assert output["penalty_mask"] == [False, True, False]
    assert output["records"][0]["masked_token_indices"] == [1]
    assert output["records"][0]["token_spans"][0]["token_ids"] == [20]
