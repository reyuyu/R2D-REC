"""Read-only API routes for TrueRec-GRPO live runs."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse


CHECKPOINT_STEP_RE = re.compile(r"(?:checkpoint[-_]?step[-_]?|checkpoint[-_]?)(\d+)$", re.I)


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


def run_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    try:
        children = list(root.iterdir())
    except OSError:
        return []
    return [
        path for path in children
        if path.is_dir() and any((path / name).exists() for name in ("live_state.json", "train_groups.jsonl"))
    ]


def selected_run(root: Path, run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise HTTPException(status_code=400, detail="a valid run_id is required")
    candidate = (root / run_id).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="unknown TrueRec run") from exc
    if not candidate.is_dir():
        raise HTTPException(status_code=404, detail="unknown TrueRec run")
    return candidate


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


def checkpoint_rows(run: Path, probes: Iterable[int]) -> list[dict[str, Any]]:
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
        world_size = metadata.get("world_size", metadata.get("cuda_device_count_saved", 4))
        rows.append({
            "step": step, "cursor": cursor, "path": str(path),
            "world_size": world_size, "probe_available": step in probe_set,
        })
    return sorted(rows, key=lambda row: row["step"])


def find_record(rows: Iterable[dict[str, Any]], *, step: int | None = None, group_id: str | None = None) -> dict[str, Any] | None:
    for row in rows:
        row_step = row.get("global_step", row.get("probe_step"))
        if step is not None and row_step != step:
            continue
        if group_id is not None and row.get("recommendation_group_id") != group_id:
            continue
        return row
    return None


def install_truerec_routes(app: Any, root: str | Path | None, static_dir: Path) -> None:
    """Install an isolated read-only surface into the existing monitor service."""
    runs_root = Path(root).expanduser().resolve() if root else Path("/nonexistent/truerec-runs")
    router = APIRouter(prefix="/api/truerec")
    gold_cache: dict[str, dict[str, Any]] | None = None

    def gold_index() -> dict[str, dict[str, Any]]:
        nonlocal gold_cache
        if gold_cache is not None:
            return gold_cache
        gold_cache = {}
        project_root = runs_root.parent
        sources = (
            project_root / "data" / "pilot4096" / "pilot4096_records.jsonl",
            project_root / "data" / "fixed_domain_abc" / "probe20_records.jsonl",
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
        for run in run_paths(runs_root):
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
        run = selected_run(runs_root, run_id)
        live = read_json(run / "live_state.json", {})
        steps = probe_steps(run)
        checkpoints = checkpoint_rows(run, steps)
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
        run = selected_run(runs_root, run_id)
        rows = read_jsonl(run / "train_groups.jsonl")
        return {"rows": rows, "count": len(rows)}

    @router.get("/train/steps")
    def train_steps(run_id: str) -> list[dict[str, Any]]:
        run = selected_run(runs_root, run_id)
        return [
            {"global_step": row.get("global_step"), "group_id": row.get("recommendation_group_id"), "domain": row.get("target_domain")}
            for row in read_jsonl(run / "train_groups.jsonl")
        ]

    @router.get("/train/explain/{step}")
    def train_explain(step: int, run_id: str) -> dict[str, Any]:
        run = selected_run(runs_root, run_id)
        row = find_record(read_jsonl(run / "train_explain.jsonl"), step=step)
        if row is None:
            raise HTTPException(status_code=404, detail="training explanation not available")
        return with_gold(row)

    @router.get("/probes")
    def probes(run_id: str) -> list[dict[str, Any]]:
        run = selected_run(runs_root, run_id)
        return [{"step": step, "summary": read_json(run / "probe" / f"step{step}" / "summary.json", read_json(run / "probe" / str(step) / "summary.json", {}))} for step in probe_steps(run)]

    def probe_dir(run: Path, step: int) -> Path:
        candidates = (run / "probe" / f"step{step}", run / "probe" / str(step))
        return next((path for path in candidates if path.is_dir()), candidates[0])

    @router.get("/probe/{step}/groups")
    def probe_groups(step: int, run_id: str) -> list[dict[str, Any]]:
        run = selected_run(runs_root, run_id)
        return read_jsonl(probe_dir(run, step) / "groups.jsonl")

    @router.get("/probe/{step}/explain")
    def probe_explain(step: int, group_id: str, run_id: str) -> dict[str, Any]:
        run = selected_run(runs_root, run_id)
        row = find_record(read_jsonl(probe_dir(run, step) / "explain.jsonl"), group_id=group_id)
        if row is None:
            raise HTTPException(status_code=404, detail="probe explanation not available")
        return with_gold(row)

    @router.get("/probe/compare")
    def probe_compare(step_a: int, step_b: int, group_id: str, run_id: str) -> dict[str, Any]:
        run = selected_run(runs_root, run_id)
        def record(step: int) -> dict[str, Any] | None:
            row = find_record(read_jsonl(probe_dir(run, step) / "explain.jsonl"), group_id=group_id)
            return with_gold(row) if row is not None else None
        return {"step_a": step_a, "step_b": step_b, "group_id": group_id, "a": record(step_a), "b": record(step_b)}

    @router.get("/checkpoints")
    def checkpoints(run_id: str) -> list[dict[str, Any]]:
        run = selected_run(runs_root, run_id)
        return checkpoint_rows(run, probe_steps(run))

    app.include_router(router)
