"""Read-only API routes for TrueRec-GRPO live runs."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
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
CURRICULUM_V2_DIR = Path("/data/GRPO/truerec_grpo/data/curriculum2048_v2")
CURRICULUM_V2_RECORDS_SHA256 = "8f1a4567aa0d6a953127e74f5667ab2907b9fbbe4aa3c0b92e27d5ecf1587542"
CURRICULUM_STAGE_FOCUS = {
    "stage1": "A-rich 起步：扩大一级兴趣命中面",
    "stage2": "A-rich + B-rich：推进 AB 层级学习",
    "stage3": "B-rich + C-rich：增加细粒度 credit 暴露",
    "stage4": "C-rich + Singleton：困难样本与富前缀复习",
}
BEAM32_RESULT_DIR = "beam32_probe"
BEAM32_WORKER = Path(os.environ.get(
    "TRUEREC_BEAM32_WORKER",
    "/data/GRPO/truerec_grpo/eval/run_probe20_beam32_checkpoints.py",
))


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


def beam32_checkpoint_rows(
    run: Path, probes: Iterable[int], adapter_source: Path = BETA_ADAPTER_SOURCE,
) -> list[dict[str, Any]]:
    """Include ancestor checkpoints when the selected run is a continuation branch."""
    rows: list[dict[str, Any]] = []
    manifest = read_json(run / "run_manifest.json", {})
    source = manifest.get("continuation_source_run")
    if source:
        source_path = Path(source).expanduser().resolve()
        if source_path.is_dir() and source_path.parent == run.resolve().parent:
            rows.extend(checkpoint_rows(source_path, probes, adapter_source))
    rows.extend(checkpoint_rows(run, probes, adapter_source))
    return sorted({int(row["step"]): row for row in rows}.values(), key=lambda row: row["step"])


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
    beam32_worker: str | Path = BEAM32_WORKER,
) -> None:
    """Install an isolated read-only surface into the existing monitor service."""
    runs_roots = normalize_run_roots(root)
    adapter_source = Path(adapter_source_dir).expanduser().resolve()
    beam_worker = Path(beam32_worker).expanduser().resolve()
    router = APIRouter(prefix="/api/truerec")
    gold_cache: dict[str, dict[str, Any]] | None = None
    curriculum_cache: dict[str, Any] | None = None

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
        ) + (CURRICULUM_V2_DIR / "records.jsonl",)
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

    def curriculum_inventory() -> dict[str, Any]:
        nonlocal curriculum_cache
        if curriculum_cache is not None:
            return curriculum_cache
        manifest = read_json(CURRICULUM_V2_DIR / "manifest.json", {})
        records = read_jsonl(CURRICULUM_V2_DIR / "records.jsonl")
        by_id = {str(row.get("recommendation_group_id")): row for row in records}
        valid = (
            manifest.get("records", {}).get("sha256") == CURRICULUM_V2_RECORDS_SHA256
            and len(records) == len(by_id) == 2048
        )
        hierarchy_totals: dict[str, int] = {}
        domain_features: dict[str, dict[str, float | int]] = {}
        for row in records:
            hierarchy = str(row.get("hierarchy_class", "OTHER"))
            hierarchy_totals[hierarchy] = hierarchy_totals.get(hierarchy, 0) + 1
            domain = str(row.get("target_domain", "unknown"))
            values = domain_features.setdefault(domain, {"N": 0, "K_A": 0, "K_AB": 0, "K_ABC": 0})
            values["N"] += 1
            for key in ("K_A", "K_AB", "K_ABC"):
                values[key] += int(row.get(key, 0) or 0)
        for values in domain_features.values():
            count = max(1, int(values["N"]))
            for key in ("K_A", "K_AB", "K_ABC"):
                values[f"{key}_mean"] = values[key] / count
        curriculum_cache = {
            "valid": valid,
            "manifest": manifest,
            "records": by_id,
            "hierarchy_totals": hierarchy_totals,
            "domain_features": domain_features,
        }
        return curriculum_cache

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

    @router.get("/curriculum")
    def curriculum(run_id: str) -> dict[str, Any]:
        run = selected_run(runs_roots, run_id)
        run_manifest = read_json(run / "run_manifest.json", {})
        inventory = curriculum_inventory()
        if (
            run_manifest.get("records_sha256") != CURRICULUM_V2_RECORDS_SHA256
            or not inventory["valid"]
        ):
            return {"available": False}
        live = read_json(run / "live_state.json", {})
        step = max(0, min(4096, int(live.get("current_step", 0) or 0)))
        epoch = 1 if step <= 2048 else 2
        epoch_step = step if epoch == 1 else step - 2048
        current_group_id = str(live.get("current_group_id", ""))
        current = inventory["records"].get(current_group_id, {})
        stage_key = str(current.get("stage", "stage1"))
        try:
            stage_number = int(stage_key.replace("stage", ""))
        except ValueError:
            stage_number = 1
        if epoch == 1:
            stage_start = (stage_number - 1) * 512
            stage_progress = max(0.0, min(1.0, (epoch_step - stage_start) / 512))
            phase = {
                "kind": "HIERARCHY_CURRICULUM",
                "label": f"Epoch 1 · Stage {stage_number}",
                "focus": CURRICULUM_STAGE_FOCUS.get(stage_key, "层级课程学习"),
                "stage": stage_number,
                "stage_progress": stage_progress,
            }
        else:
            phase = {
                "kind": "PROGRESS_AWARE_REPLAY",
                "label": "Epoch 2 · Progress-aware Replay",
                "focus": "按 HPR_C → HPR_B → NONE → HPR_A 优先回放，每 64 组保持四域均衡",
                "stage": None,
                "stage_progress": epoch_step / 2048,
            }
        return {
            "available": True,
            "epoch": epoch,
            "epoch_step": epoch_step,
            "epoch_total": 2048,
            "epoch_progress": epoch_step / 2048,
            "phase": phase,
            "current_sample": {
                key: current.get(key)
                for key in (
                    "recommendation_group_id", "target_domain", "stage", "hierarchy_class",
                    "K_A", "K_AB", "K_ABC", "K", "prefix_rich", "context_token_count",
                    "context_length_percentile", "selection_tier",
                )
            },
            "stage_focus": CURRICULUM_STAGE_FOCUS,
            "stage_domain_hierarchy": inventory["manifest"].get("stage_domain_hierarchy_census", {}),
            "hierarchy_totals": inventory["hierarchy_totals"],
            "domain_features": inventory["domain_features"],
            "records_sha256": CURRICULUM_V2_RECORDS_SHA256,
            "epoch1_order_sha256": run_manifest.get("epoch1_order_sha256"),
            "epoch2_order_manifest": read_json(run / "epoch2_order_manifest.json", None),
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

    def beam32_root(run: Path) -> Path:
        return run / BEAM32_RESULT_DIR

    @router.get("/beam32/status")
    def beam32_status(run_id: str) -> dict[str, Any]:
        run = selected_run(runs_roots, run_id)
        root = beam32_root(run)
        status = read_json(root / "status.json", {"state": "NOT_STARTED"})
        curve = read_json(root / "curve.json", [])
        return {**status, "curve": curve, "available_checkpoints": len(beam32_checkpoint_rows(run, probe_steps(run), adapter_source))}

    @router.post("/beam32/run-all")
    def beam32_run_all(run_id: str) -> dict[str, Any]:
        run = selected_run(runs_roots, run_id)
        checkpoints = beam32_checkpoint_rows(run, probe_steps(run), adapter_source)
        if not checkpoints:
            raise HTTPException(status_code=409, detail="当前实验尚无可评测 checkpoint")
        if not beam_worker.is_file():
            raise HTTPException(status_code=503, detail="Beam32 worker 尚未同步到 runtime")
        root = beam32_root(run)
        root.mkdir(parents=True, exist_ok=True)
        status_path = root / "status.json"
        current = read_json(status_path, {})
        if current.get("state") in {"QUEUED", "WAITING_FOR_GPU", "RUNNING"}:
            return current
        completed = len(read_json(root / "curve.json", []))
        queued = {
            "state": "QUEUED", "run_id": run.name, "total": len(checkpoints),
            "completed": completed, "pending": max(0, len(checkpoints) - completed),
            "min_free_gib": 70.0, "updated_at": time.time(),
        }
        temporary = root / f".status.json.tmp-{os.getpid()}"
        temporary.write_text(json.dumps(queued, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, status_path)
        log_handle = (root / "worker.log").open("ab")
        try:
            process = subprocess.Popen(
                [sys.executable, str(beam_worker), "--run-dir", str(run), "--min-free-gib", "70"],
                stdin=subprocess.DEVNULL, stdout=log_handle, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True,
            )
        except OSError as exc:
            failed = {**queued, "state": "FAILED", "error": str(exc), "updated_at": time.time()}
            status_path.write_text(json.dumps(failed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise HTTPException(status_code=500, detail=f"Beam32 worker 启动失败: {exc}") from exc
        finally:
            log_handle.close()
        latest = read_json(status_path, queued)
        if latest.get("state") == "QUEUED":
            latest["pid"] = process.pid
            temporary = root / f".status.json.tmp-{os.getpid()}"
            temporary.write_text(json.dumps(latest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, status_path)
        return latest

    @router.get("/beam32/checkpoint/{checkpoint}/groups")
    def beam32_groups(checkpoint: str, run_id: str) -> list[dict[str, Any]]:
        run = selected_run(runs_roots, run_id)
        result_dir = beam32_root(run) / checkpoint
        if not CHECKPOINT_NAME_RE.fullmatch(checkpoint) or not result_dir.is_dir():
            raise HTTPException(status_code=404, detail="Beam32 checkpoint result not found")
        return read_jsonl(result_dir / "groups.jsonl")

    @router.get("/beam32/checkpoint/{checkpoint}/explain")
    def beam32_explain(checkpoint: str, group_id: str, run_id: str) -> dict[str, Any]:
        run = selected_run(runs_roots, run_id)
        result_dir = beam32_root(run) / checkpoint
        if not CHECKPOINT_NAME_RE.fullmatch(checkpoint) or not result_dir.is_dir():
            raise HTTPException(status_code=404, detail="Beam32 checkpoint result not found")
        row = find_record(read_jsonl(result_dir / "explain.jsonl"), group_id=group_id)
        if row is None:
            raise HTTPException(status_code=404, detail="该 checkpoint 尚无 Beam32 group 明细")
        result = dict(row)
        result["gold_reference"] = {
            "all_gold_abc": row.get("all_gold_abc", []),
            "all_gold_sids": row.get("all_gold_sids", []),
            "target_domain": row.get("target_domain"),
            "K": row.get("K"),
        }
        return result

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
