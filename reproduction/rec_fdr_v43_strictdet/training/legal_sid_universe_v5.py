#!/usr/bin/env python3
"""Build and query the V5.1 immutable legal-SID universe.

The on-disk representation is one sorted uint64 array.  Python containers are
used only for the at-most-64 sampled results, never for the 21M-row catalog.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq

DOMAIN_IDS = {"video": 0, "product": 1, "ad": 2, "live": 3}
DOMAIN_ALIASES = {"video": "video", "product": "product", "prod": "product", "goods": "product", "ad": "ad", "live": "live", "living": "live"}
SOURCE_DOMAINS = {"video/video": "video", "goods": "product", "video/ad": "ad", "live": "live"}
PARSER_VERSION = "v5.1-pid2sid-float-integral-1"
MASK13 = (1 << 13) - 1


def encode_key(domain_id: int, a: int, b: int, c: int) -> np.uint64:
    values = (domain_id, a, b, c)
    if not 0 <= domain_id < 4 or any(not 0 <= value <= MASK13 for value in values[1:]):
        raise ValueError(f"SID component out of range: {values}")
    return np.uint64((domain_id << 39) | (a << 26) | (b << 13) | c)


def decode_key(key: int) -> tuple[int, int, int, int]:
    return ((key >> 39) & 3, (key >> 26) & MASK13, (key >> 13) & MASK13, key & MASK13)


def _fingerprint(files: list[Path]) -> list[dict[str, int | str]]:
    return [{"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for path in files]


def build_index(source_root: Path, cache_root: Path) -> dict:
    source_dir = source_root / "OneReason_Pid2Sid"
    files = sorted(source_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {source_dir}")
    fingerprint = _fingerprint(files)
    meta_path = cache_root / "sid_universe_meta.json"
    index_path = cache_root / "sid_keys_sorted.uint64.mmap"
    if meta_path.exists() and index_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("parser_version") == PARSER_VERSION and old.get("source_files") == fingerprint:
            expected = int(old["unique_sid_count"]) * 8
            if index_path.stat().st_size == expected:
                return old
    cache_root.mkdir(parents=True, exist_ok=True)
    raw_path = cache_root / "sid_keys_unsorted.uint64.tmp"
    scanned = success = failed = 0
    parsed_by_domain = {name: 0 for name in DOMAIN_IDS}
    with raw_path.open("wb") as raw_handle:
      for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["domain", "sid_three"], batch_size=262144):
            domains = batch.column(0).to_pylist()
            sids = batch.column(1).to_pylist()
            keys = np.empty(len(domains), dtype=np.uint64)
            used = 0
            for raw_domain, raw_sid in zip(domains, sids):
                scanned += 1
                domain = SOURCE_DOMAINS.get(str(raw_domain))
                try:
                    if domain is None or raw_sid is None or len(raw_sid) != 3:
                        raise ValueError
                    abc = tuple(int(value) for value in raw_sid)
                    if any(float(value) != integer for value, integer in zip(raw_sid, abc)):
                        raise ValueError
                    keys[used] = encode_key(DOMAIN_IDS[domain], *abc)
                except (TypeError, ValueError, OverflowError):
                    failed += 1
                    continue
                used += 1
                success += 1
                parsed_by_domain[domain] += 1
            if used:
                keys[:used].tofile(raw_handle)
    if success == 0:
        raise RuntimeError("SID parser produced no legal keys")
    keys = np.memmap(raw_path, mode="r+", dtype=np.uint64, shape=(success,))
    keys.sort()
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        previous = None
        unique_count = 0
        for start in range(0, success, 1_000_000):
            block = np.asarray(keys[start:start + 1_000_000])
            keep = np.ones(block.size, dtype=bool)
            if block.size > 1:
                keep[1:] = block[1:] != block[:-1]
            if previous is not None and block.size and int(block[0]) == previous:
                keep[0] = False
            selected = block[keep]
            selected.tofile(handle)
            unique_count += int(selected.size)
            if block.size:
                previous = int(block[-1])
        handle.flush()
        os.fsync(handle.fileno())
    del keys
    raw_path.unlink()
    temporary.replace(index_path)
    keys = np.memmap(index_path, mode="r", dtype=np.uint64, shape=(unique_count,))
    counts = {}
    for name, domain_id in DOMAIN_IDS.items():
        low, high = domain_id << 39, (domain_id + 1) << 39
        counts[name] = int(np.searchsorted(keys, high) - np.searchsorted(keys, low))
    report = {
        "schema_version": 1, "parser_version": PARSER_VERSION, "build_seed": 19260817,
        "source_root": str(source_root), "source_files": fingerprint,
        "sid_rows_scanned": scanned, "sid_parse_success": success, "sid_parse_failed": failed,
        "sid_unique_total": unique_count, "unique_sid_count": unique_count,
        "domain_counts": counts, "parsed_rows_by_domain": parsed_by_domain,
        "packed_key_sha256": _sha256_file(index_path),
    }
    meta_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (cache_root / "sid_universe_build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class LegalSidUniverse:
    def __init__(self, cache_root: Path):
        meta = json.loads((cache_root / "sid_universe_meta.json").read_text(encoding="utf-8"))
        self.meta = meta
        self.keys = np.memmap(cache_root / "sid_keys_sorted.uint64.mmap", mode="r", dtype=np.uint64, shape=(int(meta["unique_sid_count"]),))
        self._value_cache: dict[tuple[str, int | None, int | None], np.ndarray] = {}

    def level_values(self, domain: str, level: str, a: int | None = None, b: int | None = None) -> np.ndarray:
        key = (DOMAIN_ALIASES[domain], a if level != "a" else None, b if level == "c" else None)
        cached = self._value_cache.get(key)
        if cached is not None:
            return cached
        lo, hi = self.prefix_range(domain, a if level != "a" else None, b if level == "c" else None)
        shift = {"a": 26, "b": 13, "c": 0}[level]
        values = np.unique((self.keys[lo:hi] >> np.uint64(shift)) & np.uint64(MASK13))
        if level != "c" or values.size <= 4096:
            self._value_cache[key] = values
        return values

    def prefix_range(self, domain: str, a: int | None = None, b: int | None = None) -> tuple[int, int]:
        domain_id = DOMAIN_IDS[DOMAIN_ALIASES[domain]]
        if a is None:
            low, high = domain_id << 39, (domain_id + 1) << 39
        elif b is None:
            low = (domain_id << 39) | (a << 26); high = low + (1 << 26)
        else:
            low = (domain_id << 39) | (a << 26) | (b << 13); high = low + (1 << 13)
        return int(np.searchsorted(self.keys, low)), int(np.searchsorted(self.keys, high))

    def available_counts(self, domain: str, anchor: tuple[int, int, int], positives: Iterable[tuple[int, int, int]]) -> dict[str, int]:
        positives = list(positives); pos_a = {x[0] for x in positives}; pos_b = {x[1] for x in positives if x[0] == anchor[0]}; pos_c = {x[2] for x in positives if x[:2] == anchor[:2]}
        a_values = self.level_values(domain, "a")
        b_values = self.level_values(domain, "b", anchor[0])
        c_values = self.level_values(domain, "c", anchor[0], anchor[1])
        def remaining(values: np.ndarray, excluded: set[int]) -> int:
            present = 0
            for value in excluded:
                index = int(np.searchsorted(values, value))
                present += int(index < values.size and int(values[index]) == value)
            return int(values.size) - present
        return {"a": remaining(a_values, pos_a), "b": remaining(b_values, pos_b), "c": remaining(c_values, pos_c)}

    def sample_pairs(self, domain: str, anchor: tuple[int, int, int], positives: list[tuple[int, int, int]], rng: random.Random, quota: dict[str, int]) -> dict[str, list[tuple[int, int]]]:
        pos_a = {x[0] for x in positives}; pos_b = {x[1] for x in positives if x[0] == anchor[0]}; pos_c = {x[2] for x in positives if x[:2] == anchor[:2]}
        excluded = {"a": pos_a, "b": pos_b, "c": pos_c}; shift = {"a": 26, "b": 13, "c": 0}; positive = {"a": anchor[0], "b": anchor[1], "c": anchor[2]}
        output = {}
        for level in ("a", "b", "c"):
            values = self.level_values(domain, level, anchor[0] if level != "a" else None, anchor[1] if level == "c" else None)
            target = min(int(quota[level]), self.available_counts(domain, anchor, positives)[level])
            selected_set: set[int] = set()
            while len(selected_set) < target:
                value = int(values[rng.randrange(int(values.size))])
                if value not in excluded[level]:
                    selected_set.add(value)
            selected = sorted(selected_set)
            output[level] = [(positive[level], value) for value in selected]
        return output


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("build", "self-test")); parser.add_argument("--source-root", type=Path, default=Path("/data/Explorer_LLM_Rec_Competition/data")); parser.add_argument("--cache-root", type=Path, default=Path("/data/Explorer_LLM_Rec_Competition/cache/fdr_sid_universe_v5")); args = parser.parse_args()
    if args.command == "build": print(json.dumps(build_index(args.source_root, args.cache_root), ensure_ascii=False, indent=2)); return
    for domain in range(4):
        for value in (0, 8191): assert decode_key(int(encode_key(domain, value, value, value))) == (domain, value, value, value)
    print(json.dumps({"event": "legal_sid_universe_self_test_passed"}))


if __name__ == "__main__": main()
