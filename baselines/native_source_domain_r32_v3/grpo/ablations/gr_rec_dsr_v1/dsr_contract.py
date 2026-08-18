"""Frozen data and probe contract for the DSR formal runner."""
from __future__ import annotations


TRAIN_SEED = "20260816"
PROBE_SEED = "20260818"
PROBE_EVERY_STEPS = "200"
PROBE_GROUP_IDS = (
    "fc6e5676c19873ebadd3deed3ed25679d986fe7a7201d02aff860e58a508e82e",
    "6068defdb009836ada15a9f22c47d97934801e795cf8c33b10587491b824ba6f",
    "281f3fe03e1ad3b8919df9660a89360561987a44118ba6f0355b78fe7c172700",
    "2cb88d8ec6d6fcce66385b20be6885b57a84a607cc0984d99efb1561f8de86c8",
)


def _single_value(argv: list[str], option: str, expected: str) -> None:
    positions = [index for index, value in enumerate(argv) if value == option]
    if not positions:
        argv.extend([option, expected])
        return
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"DSR requires exactly one {option} {expected}")
    actual = argv[positions[0] + 1]
    if actual != expected:
        raise ValueError(f"DSR requires {option} {expected}, got {actual}")


def enforce_formal_contract(argv) -> list[str]:
    """Return formal-run arguments with the frozen fairness contract applied."""
    values = list(argv)
    if "--resume-from-checkpoint" in values:
        raise ValueError("DSR must start from the original BATA adapter; resume is forbidden")
    _single_value(values, "--seed", TRAIN_SEED)
    _single_value(values, "--n-groups", "all")
    _single_value(values, "--probe-groups", "4")
    _single_value(values, "--probe-seed", PROBE_SEED)
    _single_value(values, "--probe-every-steps", PROBE_EVERY_STEPS)

    positions = [index for index, value in enumerate(values) if value == "--probe-group-id"]
    if positions:
        actual = []
        for position in positions:
            if position + 1 >= len(values):
                raise ValueError("--probe-group-id requires a value")
            actual.append(values[position + 1])
        if tuple(actual) != PROBE_GROUP_IDS:
            raise ValueError("DSR probe IDs must be ordered video/living/prod/ad")
    else:
        for group_id in PROBE_GROUP_IDS:
            values.extend(["--probe-group-id", group_id])
    return values
