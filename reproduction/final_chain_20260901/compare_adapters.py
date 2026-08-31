#!/usr/bin/env python3
"""Optional CPU-only numerical comparison of historical and reproduced adapters."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from repro_quality import sha256_file


def compare_adapter_files(reference: Path, reproduced: Path) -> dict[str, object]:
    from safetensors import safe_open
    import torch

    reference = reference.resolve(strict=True)
    reproduced = reproduced.resolve(strict=True)
    reference_sha = sha256_file(reference)
    reproduced_sha = sha256_file(reproduced)
    with safe_open(reference, framework="pt", device="cpu") as left, safe_open(reproduced, framework="pt", device="cpu") as right:
        left_keys = set(left.keys())
        right_keys = set(right.keys())
        if left_keys != right_keys:
            return {
                "status": "structure_mismatch",
                "missing_in_reproduced": sorted(left_keys - right_keys),
                "extra_in_reproduced": sorted(right_keys - left_keys),
                "reference_sha256": reference_sha,
                "reproduced_sha256": reproduced_sha,
            }
        dot = left_norm = right_norm = diff_norm = 0.0
        max_abs = 0.0
        element_count = 0
        tensor_count = 0
        dtype_pairs: set[str] = set()
        for key in sorted(left_keys):
            a = left.get_tensor(key)
            b = right.get_tensor(key)
            if tuple(a.shape) != tuple(b.shape):
                return {"status": "shape_mismatch", "tensor": key, "reference_shape": list(a.shape), "reproduced_shape": list(b.shape)}
            dtype_pairs.add(f"{a.dtype}/{b.dtype}")
            a64 = a.to(torch.float64)
            b64 = b.to(torch.float64)
            delta = b64 - a64
            dot += float(torch.sum(a64 * b64))
            left_norm += float(torch.sum(a64 * a64))
            right_norm += float(torch.sum(b64 * b64))
            diff_norm += float(torch.sum(delta * delta))
            max_abs = max(max_abs, float(torch.max(torch.abs(delta))) if delta.numel() else 0.0)
            element_count += a.numel()
            tensor_count += 1
        cosine = dot / math.sqrt(left_norm * right_norm) if left_norm and right_norm else 1.0
        relative_l2 = math.sqrt(diff_norm) / math.sqrt(left_norm) if left_norm else 0.0
        return {
            "status": "bitwise_match" if reference_sha == reproduced_sha else "numerically_compared",
            "reference_sha256": reference_sha,
            "reproduced_sha256": reproduced_sha,
            "tensor_count": tensor_count,
            "element_count": element_count,
            "dtype_pairs": sorted(dtype_pairs),
            "cosine_similarity": cosine,
            "relative_l2": relative_l2,
            "max_abs_delta": max_abs,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reproduced", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_adapter_files(args.reference, args.reproduced)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
