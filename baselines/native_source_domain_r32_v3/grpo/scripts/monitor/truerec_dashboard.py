"""Read-only API routes for TrueRec-GRPO live runs."""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask


CHECKPOINT_STEP_RE = re.compile(r"(?:checkpoint[-_]?step[-_]?|checkpoint[-_]?)(\d+)$", re.I)
CHECKPOINT_NAME_RE = re.compile(r"checkpoint-step-\d+$")
BETA_ADAPTER_SOURCE = Path(
    "/data/outputs/baselines/native_source_domain_r32_v3/"
    "BATA-BASELINE-R32-2E-GC04-4GPU-AUTO-RETRY3-20260812-063333/checkpoint-1106"
)
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def read_json(path: Path, default: Any) -> Any:
    """Read an atomically-replaced JSON file without surfacing transient errors."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Return complete JSON objects, ignoring a concurrently appended tail."""
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except (UnicodeError, json.JSONDecodeError):
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except OSError:
        pass
    return rows


def normalize_run_roots(root: str | Path | Iterable[str | Path] | None) -> tuple[Path, ...]:
    if root is None:
        return (Path("/nonexistent/truerec-runs"),)
    values = (root,) if isinstance(root, (str, Path)) else tuple(root)
    return tuple(dict.fromkeys(Path(value).expanduser().resolve() for value in values))


def run_paths(roots: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        try:
            children = list(root.iterdir())
        except OSError:
            continue
        result.extend(
            path for path in children
            if path.is_dir() and any((path / name).exists() for name in ("live_state.json", "train_groups.jsonl"))
        )
    return list(dict.fromkeys(path.resolve() for path in result))


def selected_run(roots: Iterable[Path], run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise HTTPException(status_code=400, detail="a valid run_id is required")
    matches = []
    for root in roots:
        candidate = (root / run_id).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            continue
        if candidate.is_dir():
            matches.append(candidate)
    matches = list(dict.fromkeys(matches))
    if not matches:
        raise HTTPException(status_code=404, detail="unknown TrueRec run")
    if len(matches) > 1:
        raise HTTPException(status_code=409, detail="TrueRec run_id exists in multiple roots")
    return matches[0]


def probe_steps(run: Path) -> list[int]:
    root = run / "probe"
    if not root.is_dir():
        return []
    result = []
    try:
        children = root.iterdir()
    except OSError:
        return []
    for path in children:
        if path.is_dir():
            try:
                result.append(int(path.name.replace("step", "").lstrip("-_")))
            except ValueError:
                continue
    return sorted(set(result))


def adapter_inventory(checkpoint: Path, adapter_source: Path) -> dict[str, int]:
    if not (checkpoint / "state.pt").is_file():
        return {}
    config = adapter_source / "adapter_config.json"
    reference_weights = adapter_source / "adapter_model.safetensors"
    if not config.is_file() or not reference_weights.is_file():
        return {}
    return {
        "adapter_config.json": config.stat().st_size,
        # The exported state must match the reference key/shape/dtype contract exactly.
        "adapter_model.safetensors": reference_weights.stat().st_size,
    }


def checkpoint_rows(
    run: Path, probes: Iterable[int], adapter_source: Path = BETA_ADAPTER_SOURCE,
) -> list[dict[str, Any]]:
    root = run / "checkpoints"
    probe_set = set(probes)
    if not root.is_dir():
        return []
    rows = []
    try:
        children = list(root.iterdir())
    except OSError:
        return []
    for path in children:
        if not path.is_dir():
            continue
        metadata = read_json(path / "metadata.json", {})
        match = CHECKPOINT_STEP_RE.search(path.name)
        step = metadata.get("global_step", metadata.get("driver_state", {}).get("global_step"))
        if step is None and match:
            step = int(match.group(1))
        try:
            step = int(step)
        except (TypeError, ValueError):
            continue
        cursor = metadata.get("next_group_index", metadata.get("cursor"))
        if cursor is None and isinstance(metadata.get("driver_state"), dict):
            cursor = metadata["driver_state"].get("next_group_index")
        if cursor is None and isinstance(metadata.get("training_cursor"), dict):
            cursor = metadata["training_cursor"].get("next_group_index")
        world_size = metadata.get("world_size", metadata.get("cuda_device_count_saved", 4))
        rows.append({
            "checkpoint": path.name, "step": step, "cursor": cursor, "path": str(path),
            "world_size": world_size, "probe_available": step in probe_set,
            "files": adapter_inventory(path, adapter_source),
        })
    return sorted(rows, key=lambda row: row["step"])


def checkpoint_directory(run: Path, checkpoint: str) -> Path:
    if Path(checkpoint).name != checkpoint or not CHECKPOINT_NAME_RE.fullmatch(checkpoint):
        raise HTTPException(status_code=404, detail="checkpoint not found")
    root = (run / "checkpoints").resolve()
    candidate = (root / checkpoint).resolve()
    if candidate.parent != root or not candidate.is_dir():
        raise HTTPException(status_code=404, detail="checkpoint not found")
    return candidate


def export_adapter_safetensors(
    checkpoint: Path, adapter_source: Path, temporary_root: Path | None = None,
) -> Path:
    """Export only the LoRA state from a formal training checkpoint."""
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("adapter export dependencies are unavailable") from exc

    state_path = checkpoint / "state.pt"
    reference_path = adapter_source / "adapter_model.safetensors"
    if not state_path.is_file() or not reference_path.is_file():
        raise RuntimeError("checkpoint adapter source is incomplete")
    payload = torch.load(state_path, map_location="cpu", weights_only=False, mmap=True)
    state = payload.get("model_state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not state:
        raise RuntimeError("checkpoint has no model_state_dict")
    if any(not isinstance(key, str) or "lora_" not in key for key in state):
        raise RuntimeError("checkpoint model state is not LoRA-only")
    with safe_open(reference_path, framework="pt", device="cpu") as reference:
        reference_keys = set(reference.keys())
        if set(state) != reference_keys:
            raise RuntimeError("checkpoint LoRA key contract mismatch")
        for key, tensor in state.items():
            if tuple(tensor.shape) != tuple(reference.get_slice(key).get_shape()):
                raise RuntimeError(f"checkpoint LoRA shape mismatch: {key}")
            if tensor.dtype != reference.get_tensor(key).dtype:
                raise RuntimeError(f"checkpoint LoRA dtype mismatch: {key}")

    temporary_dir = Path(temporary_root) if temporary_root else Path("/root/truerec_adapter_exports")
    temporary_dir.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f"{checkpoint.name}-", suffix=".safetensors", dir=temporary_dir, delete=False
    )
    output = Path(handle.name)
    handle.close()
    try:
        tensors = {key: tensor.detach().cpu().contiguous() for key, tensor in state.items()}
        save_file(tensors, output, metadata={"format": "pt"})
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return output


def find_record(rows: Iterable[dict[str, Any]], *, step: int | None = None, group_id: str | None = None) -> dict[str, Any] | None:
    for row in rows:
        row_step = row.get("global_step", row.get("probe_step"))
        if step is not None and row_step != step:
            continue
        if group_id is not None and row.get("recommendation_group_id") != group_id:
            continue
        return row
    return None


def install_truerec_routes(
    app: Any, root: str | Path | Iterable[str | Path] | None, static_dir: Path,
    adapter_source_dir: str | Path = BETA_ADAPTER_SOURCE,
) -> None:
    """Install an isolated read-only surface into the existing monitor service."""
    runs_roots = normalize_run_roots(root)
    adapter_source = Path(adapter_source_dir).expanduser().resolve()
    router = APIRouter(prefix="/api/truerec")
    gold_cache: dict[str, dict[str, Any]] | None = None

    def gold_index() -> dict[str, dict[str, Any]]:
        nonlocal gold_cache
        if gold_cache is not None:
            return gold_cache
        gold_cache = {}
        sources = tuple(
            source
            for runs_root in runs_roots
            for source in (
                runs_root.parent / "data" / "pilot4096" / "pilot4096_records.jsonl",
                runs_root.parent / "data" / "fixed_domain_abc" / "probe20_records.jsonl",
            )
        )
        for source in sources:
            for row in read_jsonl(source):
                group_id = row.get("recommendation_group_id")
                if group_id:
                    gold_cache[str(group_id)] = {
                        "all_gold_abc": row.get("all_gold_abc", []),
                        "all_gold_sids": row.get("all_gold_sids", []),
                        "target_domain": row.get("target_domain"),
                        "K": row.get("K"),
                    }
        return gold_cache

    def with_gold(row: dict[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result["gold_reference"] = gold_index().get(str(row.get("recommendation_group_id", "")), {})
        return result

    @app.get("/truerec")
    def truerec_dashboard() -> FileResponse:
        return FileResponse(static_dir / "truerec.html")

    @router.get("/runs")
    def runs() -> list[dict[str, Any]]:
        result = []
        for run in run_paths(runs_roots):
            live = read_json(run / "live_state.json", {})
            groups = read_jsonl(run / "train_groups.jsonl")
            latest = live.get("current_step", groups[-1].get("global_step", 0) if groups else 0)
            try:
                updated = max(path.stat().st_mtime for path in (run / "live_state.json", run / "train_groups.jsonl") if path.exists())
            except (OSError, ValueError):
                updated = 0.0
            result.append({"run_id": run.name, "latest_step": latest, "total_steps": live.get("total_steps", 4096), "updated_at": updated})
        return sorted(result, key=lambda row: (row["updated_at"], row["run_id"]), reverse=True)

    @router.get("/overview")
    def overview(run_id: str) -> dict[str, Any]:
        run = selected_run(runs_roots, run_id)
        live = read_json(run / "live_state.json", {})
        steps = probe_steps(run)
        checkpoints = checkpoint_rows(run, steps, adapter_source)
        current = int(live.get("current_step", 0) or 0)
        interval = 256
        return {
            "status": "live" if live else "waiting", "run_id": run.name, "live": live,
            "last_probe": max((step for step in steps if step <= current), default=None),
            "next_probe": min((step for step in range(0, int(live.get("total_steps", 4096)) + 1, interval) if step > current), default=None),
            "next_checkpoint": min((step for step in range(interval, int(live.get("total_steps", 4096)) + 1, interval) if step > current), default=None),
            "checkpoints": checkpoints, "probe_steps": steps,
        }

    @router.get("/curves")
    def curves(run_id: str) -> dict[str, Any]:
        run = selected_run(runs_roots, run_id)
        rows = read_jsonl(run / "train_groups.jsonl")
        return {"rows": rows, "count": len(rows)}

    @router.get("/train/steps")
    def train_steps(run_id: str) -> list[dict[str, Any]]:
        run = selected_run(runs_roots, run_id)
        return [
            {"global_step": row.get("global_step"), "group_id": row.get("recommendation_group_id"), "domain": row.get("target_domain")}
            for row in read_jsonl(run / "train_groups.jsonl")
        ]

    @router.get("/train/explain/{step}")
    def train_explain(step: int, run_id: str) -> dict[str, Any]:
        run = selected_run(runs_roots, run_id)
        row = find_record(read_jsonl(run / "train_explain.jsonl"), step=step)
        if row is None:
            raise HTTPException(status_code=404, detail="training explanation not available")
        return with_gold(row)

    @router.get("/probes")
    def probes(run_id: str) -> list[dict[str, Any]]:
        run = selected_run(runs_roots, run_id)
        return [{"step": step, "summary": read_json(run / "probe" / f"step{step}" / "summary.json", read_json(run / "probe" / str(step) / "summary.json", {}))} for step in probe_steps(run)]

    def probe_dir(run: Path, step: int) -> Path:
        candidates = (run / "probe" / f"step{step}", run / "probe" / str(step))
        return next((path for path in candidates if path.is_dir()), candidates[0])

    @router.get("/probe/{step}/groups")
    def probe_groups(step: int, run_id: str) -> list[dict[str, Any]]:
        run = selected_run(runs_roots, run_id)
        return read_jsonl(probe_dir(run, step) / "groups.jsonl")

    @router.get("/probe/{step}/explain")
    def probe_explain(step: int, group_id: str, run_id: str) -> dict[str, Any]:
        run = selected_run(runs_roots, run_id)
        row = find_record(read_jsonl(probe_dir(run, step) / "explain.jsonl"), group_id=group_id)
        if row is None:
            raise HTTPException(status_code=404, detail="probe explanation not available")
        return with_gold(row)

    @router.get("/probe/compare")
    def probe_compare(step_a: int, step_b: int, group_id: str, run_id: str) -> dict[str, Any]:
        run = selected_run(runs_roots, run_id)
        def record(step: int) -> dict[str, Any] | None:
            row = find_record(read_jsonl(probe_dir(run, step) / "explain.jsonl"), group_id=group_id)
            return with_gold(row) if row is not None else None
        return {"step_a": step_a, "step_b": step_b, "group_id": group_id, "a": record(step_a), "b": record(step_b)}

    @router.get("/checkpoints")
    def checkpoints(run_id: str) -> list[dict[str, Any]]:
        run = selected_run(runs_roots, run_id)
        return checkpoint_rows(run, probe_steps(run), adapter_source)

    @router.get("/checkpoints/{checkpoint}/download")
    def download_checkpoint(checkpoint: str, file: str, run_id: str):
        if file not in ADAPTER_FILES:
            raise HTTPException(status_code=404, detail="adapter file not found")
        run = selected_run(runs_roots, run_id)
        directory = checkpoint_directory(run, checkpoint)
        if not adapter_inventory(directory, adapter_source):
            raise HTTPException(status_code=404, detail="checkpoint adapter is not ready")
        if file == "adapter_config.json":
            return FileResponse(adapter_source / file, filename=file)
        try:
            exported = export_adapter_safetensors(directory, adapter_source)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=f"adapter export failed: {exc}") from exc
        return FileResponse(
            exported, filename=file,
            background=BackgroundTask(exported.unlink, missing_ok=True),
        )

    app.include_router(router)
