"""CPU safety checks for the formal commitment fast preflight."""

from __future__ import annotations

import ast
import inspect

import audit_gpu_fast_commitment_preflight as audit


credits = [
    (0.1, -0.2, 0.0, 0.0),
    (-0.1, 0.2, -0.3, 0.0),
]
reference = {
    "abc_credits_reconstructed_from_immutable_rollout": [
        [-0.2, 0.0, 0.0],
        [0.2, -0.3, 0.0],
    ]
}
assert audit.abc_parity(reference, credits) is True
reference["abc_credits_reconstructed_from_immutable_rollout"][1][1] = 99
assert audit.abc_parity(reference, credits) is False

tree = ast.parse(inspect.getsource(audit))
assert not [
    node for node in ast.walk(tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and node.func.attr == "step"
]

print("FAST COMMITMENT PREFLIGHT CPU TESTS PASSED")
