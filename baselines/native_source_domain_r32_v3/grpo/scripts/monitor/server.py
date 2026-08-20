"""Local FastAPI server for an append-only GRPO monitor run."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from .advantage_adapter import reconstruct_groups
except ImportError:  # Direct execution: python monitor/server.py
    from advantage_adapter import reconstruct_groups


STATIC_DIR = Path(__file__).resolve().parent / "static"
RECOMMENDATION_RUN_KIND = "recommendation_grpo"
USER_RUN_KIND = "user_grpo"
CHECKPOINT_NAME_RE = re.compile(r"checkpoint-(?:step)?(\d+)")
FINAL_CHECKPOINT_NAME = "full-epoch-final"


def checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_NAME_RE.fullmatch(path.name)
    if match is not None:
        return int(match.group(1))
    if path.name != FINAL_CHECKPOINT_NAME:
        return None
    try:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        step = int(metadata["global_optimizer_step"])
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return step if step >= 0 else None


def checkpoint_name_allowed(name: str) -> bool:
    return CHECKPOINT_NAME_RE.fullmatch(name) is not None or name == FINAL_CHECKPOINT_NAME


def normalized_run_kind(manifest: dict[str, Any]) -> str:
    """Return the supported run kind, preserving legacy Recommendation runs."""
    return USER_RUN_KIND if manifest.get("run_kind") == USER_RUN_KIND else RECOMMENDATION_RUN_KIND


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read complete valid lines and ignore a concurrently-written tail."""
    if not path.exists():
        return []
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except OSError:
        return []
    return rows


def filter_rows(
    rows: Iterable[dict[str, Any]],
    from_step: int | None = None,
    to_step: int | None = None,
    route: str | None = None,
    rollout_id: int | None = None,
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        step = row.get("step")
        if from_step is not None and (step is None or step < from_step):
            continue
        if to_step is not None and (step is None or step > to_step):
            continue
        if route is not None and row.get("route") != route:
            continue
        if rollout_id is not None and row.get("rollout_id") != rollout_id:
            continue
        result.append(row)
    return result


def create_app(
    run_dir: str | Path | None = None,
    *,
    runs_dir: str | Path | None = None,
    outputs_dir: str | Path | None = None,
    user_runs_dir: str | Path | None = None,
) -> FastAPI:
    if (run_dir is None) == (runs_dir is None):
        raise ValueError("exactly one of run_dir or runs_dir is required")
    single_run = Path(run_dir).expanduser().resolve() if run_dir is not None else None
    root = single_run.parent if single_run is not None else Path(runs_dir).expanduser().resolve()
    outputs_root = Path(outputs_dir).expanduser().resolve() if outputs_dir is not None else None
    user_runs_root = Path(user_runs_dir).expanduser().resolve() if user_runs_dir is not None else None
    app = FastAPI(title="GRPO Monitor", docs_url="/api/docs", redoc_url=None)
    app.state.run_dir = single_run
    app.state.runs_dir = root
    app.state.outputs_dir = outputs_root
    app.state.user_runs_dir = user_runs_root
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    source_cache: dict[tuple[str, int, int, str], dict[str, dict[str, Any]]] = {}

    def available_runs() -> list[dict[str, Any]]:
        paths = [single_run] if single_run is not None else (
            [path for path in root.iterdir() if path.is_dir()] if root.exists() else []
        )
        runs = []
        for path in paths:
            if path is None:
                continue
            try:
                manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
            metrics = read_jsonl(path / "metrics.jsonl")
            latest = metrics[-1] if metrics else {}
            try:
                updated_at = max(
                    candidate.stat().st_mtime
                    for candidate in (path / "manifest.json", path / "metrics.jsonl", path / "rollouts.jsonl")
                    if candidate.exists()
                )
            except (OSError, ValueError):
                updated_at = 0.0
            runs.append({
                "run_id": path.name,
                "run_kind": normalized_run_kind(manifest),
                "demo": bool(manifest.get("demo", False)),
                "start_time": manifest.get("start_time"),
                "max_steps": manifest.get("effective_max_steps", manifest.get("max_steps")),
                "latest_step": latest.get("step", 0),
                "latest_route": latest.get("route"),
                "updated_at": updated_at,
            })
        return sorted(runs, key=lambda item: (item["updated_at"], item["run_id"]), reverse=True)

    def selected_run(run_id: str | None) -> Path:
        if single_run is not None:
            if run_id is not None and run_id != single_run.name:
                raise HTTPException(status_code=404, detail="unknown run_id")
            return single_run
        if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
            raise HTTPException(status_code=400, detail="a valid run_id is required")
        candidate = (root / run_id).resolve()
        if candidate.parent != root or not candidate.is_dir():
            raise HTTPException(status_code=404, detail="unknown run_id")
        return candidate

    def source_rows(selected: Path, source_kind: str) -> dict[str, dict[str, Any]]:
        """Load a manifest-declared frozen dataset and expose only display fields."""
        try:
            manifest_data = json.loads((selected / "manifest.json").read_text(encoding="utf-8"))
            if source_kind == "probe":
                declaration = manifest_data["fixed_probe"]
                dataset = Path(declaration["dataset"]).expanduser().resolve()
                expected_sha = str(declaration["sha256"])
            elif source_kind == "train":
                dataset = Path(manifest_data["dataset"]).expanduser().resolve()
                expected_sha = str(manifest_data["dataset_sha"][dataset.name])
            else:
                return {}
            stat = dataset.stat()
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return {}
        if dataset.suffix != ".jsonl" or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            return {}
        cache_key = (str(dataset), stat.st_mtime_ns, stat.st_size, expected_sha)
        cached = source_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            if hashlib.sha256(dataset.read_bytes()).hexdigest() != expected_sha:
                return {}
        except OSError:
            return {}
        allowed = ("prompt", "history_sids", "history_events", "gold_sids", "gold_events")
        indexed = {
            str(row["sample_id"]): {key: row[key] for key in allowed if key in row}
            for row in read_jsonl(dataset)
            if row.get("sample_id")
        }
        source_cache[cache_key] = indexed
        return indexed

    def enrich_source_fields(
        rows: list[dict[str, Any]],
        indexed: dict[str, dict[str, Any]],
        *,
        one_per_source: bool = False,
    ) -> None:
        enriched_ids: set[str] = set()
        for row in rows:
            source_id = str(row.get("sample_id") or row.get("group_id") or "")
            if one_per_source and source_id in enriched_ids:
                continue
            for key, value in indexed.get(source_id, {}).items():
                row.setdefault(key, value)
            if source_id in indexed:
                enriched_ids.add(source_id)

    def checkpoint_run_dirs(selected: Path) -> list[Path]:
        """Find approved adapter-output layouts without exposing arbitrary manifest paths."""
        candidates = []
        if outputs_root is not None and outputs_root.is_dir():
            candidates.append(outputs_root / selected.name)
            try:
                candidates.extend(path / selected.name for path in outputs_root.iterdir() if path.is_dir())
            except OSError:
                pass
        if user_runs_root is not None and user_runs_root.is_dir():
            try:
                manifest = json.loads((selected / "manifest.json").read_text(encoding="utf-8"))
                declared = Path(str(manifest["output_dir"])).expanduser().resolve()
                if declared.parent == user_runs_root and declared.name == selected.name:
                    candidates.append(declared)
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        resolved = []
        for candidate in candidates:
            try:
                path = candidate.resolve()
                if outputs_root is not None:
                    try:
                        path.relative_to(outputs_root)
                    except ValueError:
                        if user_runs_root is None:
                            continue
                        path.relative_to(user_runs_root)
            except (OSError, ValueError):
                continue
            if path.is_dir() and path not in resolved:
                resolved.append(path)
        return resolved

    @app.get("/")
    def dashboard():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/runs")
    def runs(run_kind: str | None = None):
        items = available_runs()
        if run_kind is None:
            return items
        if run_kind not in {RECOMMENDATION_RUN_KIND, USER_RUN_KIND}:
            raise HTTPException(status_code=400, detail="unknown run_kind")
        return [item for item in items if item["run_kind"] == run_kind]

    @app.get("/api/checkpoints")
    def checkpoints(run_id: str | None = None):
        selected = selected_run(run_id)
        result = {}
        for run_output in checkpoint_run_dirs(selected):
            try:
                children = run_output.iterdir()
            except OSError:
                continue
            for path in children:
                if not path.is_dir():
                    continue
                step = checkpoint_step(path)
                if step is None:
                    continue
                files = {}
                for name in ("adapter_config.json", "adapter_model.safetensors"):
                    candidate = path / name
                    if candidate.is_file():
                        files[name] = candidate.stat().st_size
                result.setdefault(path.name, {
                    "checkpoint": path.name,
                    "step": step,
                    "files": files,
                })
        return sorted(result.values(), key=lambda item: item["step"])

    @app.get("/api/checkpoints/{checkpoint}/download")
    def download_checkpoint_file(checkpoint: str, file: str, run_id: str | None = None):
        selected = selected_run(run_id)
        allowed = {"adapter_config.json", "adapter_model.safetensors"}
        if file not in allowed or not checkpoint_name_allowed(checkpoint):
            raise HTTPException(status_code=404, detail="checkpoint file not found")
        for run_output in checkpoint_run_dirs(selected):
            checkpoint_dir = (run_output / checkpoint).resolve()
            candidate = (checkpoint_dir / file).resolve()
            if checkpoint_dir.parent == run_output and candidate.parent == checkpoint_dir and candidate.is_file():
                return FileResponse(candidate, filename=f"{selected.name}-{checkpoint}-{file}")
        raise HTTPException(status_code=404, detail="checkpoint file not found")

    @app.delete("/api/checkpoints/{checkpoint}")
    def delete_checkpoint(checkpoint: str, run_id: str | None = None, confirm: str | None = None):
        selected = selected_run(run_id)
        if not checkpoint_name_allowed(checkpoint):
            raise HTTPException(status_code=404, detail="checkpoint not found")
        if confirm != checkpoint:
            raise HTTPException(status_code=400, detail="checkpoint confirmation does not match")
        targets = []
        for run_output in checkpoint_run_dirs(selected):
            candidate = (run_output / checkpoint).resolve()
            if candidate.parent == run_output and candidate.is_dir():
                targets.append(candidate)
        if not targets:
            raise HTTPException(status_code=404, detail="checkpoint not found")
        released_bytes = 0
        for target in targets:
            try:
                released_bytes += sum(path.stat().st_size for path in target.rglob("*") if path.is_file())
                shutil.rmtree(target)
            except OSError as exc:
                raise HTTPException(status_code=500, detail=f"checkpoint deletion failed: {exc}") from exc
        return {
            "run_id": selected.name,
            "checkpoint": checkpoint,
            "deleted_directories": len(targets),
            "released_bytes": released_bytes,
        }

    @app.get("/api/manifest")
    def manifest(run_id: str | None = None):
        selected = selected_run(run_id)
        try:
            data = json.loads((selected / "manifest.json").read_text(encoding="utf-8"))
            data["run_kind"] = normalized_run_kind(data)
            if "effective_max_steps" in data:
                data["max_steps"] = data["effective_max_steps"]
            return data
        except (OSError, json.JSONDecodeError):
            return {}

    @app.get("/api/capabilities")
    def capabilities(run_id: str | None = None):
        selected = selected_run(run_id)
        try:
            manifest_data = json.loads((selected / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest_data = {}
        fixed_probe = manifest_data.get("fixed_probe")
        dsr_files = ("dsr_metrics.jsonl", "dsr_steps.jsonl", "dsr_traces.jsonl")
        probe_rows = read_jsonl(selected / "probes.jsonl")
        dsr = bool(
            manifest_data.get("dsr")
            or any((selected / name).is_file() for name in dsr_files)
            or any(isinstance(row.get("dsr"), dict) for row in probe_rows)
        )
        checkpoint_available = any(
            path.is_dir() and checkpoint_step(path) is not None
            for output in checkpoint_run_dirs(selected)
            for path in output.iterdir()
        )
        return {
            "run_kind": normalized_run_kind(manifest_data),
            "user_grpo": normalized_run_kind(manifest_data) == USER_RUN_KIND,
            "dsr": dsr,
            "probes": bool(
                (isinstance(fixed_probe, dict) and fixed_probe.get("enabled"))
                or (selected / "probes.jsonl").is_file()
            ),
            "checkpoints": checkpoint_available,
        }

    def queried(path: Path, from_step, to_step, route, rollout_id):
        return filter_rows(read_jsonl(path), from_step, to_step, route, rollout_id)

    @app.get("/api/metrics")
    def metrics(
        run_id: str | None = None,
        from_step: int | None = None,
        to_step: int | None = None,
        route: str | None = None,
        rollout_id: int | None = None,
    ):
        return queried(selected_run(run_id) / "metrics.jsonl", from_step, to_step, route, rollout_id)

    @app.get("/api/rollouts")
    def rollouts(
        run_id: str | None = None,
        from_step: int | None = None,
        to_step: int | None = None,
        route: str | None = None,
        rollout_id: int | None = None,
    ):
        return queried(selected_run(run_id) / "rollouts.jsonl", from_step, to_step, route, rollout_id)

    @app.get("/api/sample-context")
    def sample_context(sample_id: str, run_id: str | None = None):
        if re.fullmatch(r"[0-9a-f]{64}", sample_id) is None:
            raise HTTPException(status_code=404, detail="sample context not found")
        context = source_rows(selected_run(run_id), "train").get(sample_id)
        if context is None:
            raise HTTPException(status_code=404, detail="sample context not found")
        return context

    @app.get("/api/ranks")
    def ranks(
        run_id: str | None = None,
        from_step: int | None = None,
        to_step: int | None = None,
        route: str | None = None,
        rollout_id: int | None = None,
        rank: int | None = Query(default=None, ge=0),
    ):
        selected = selected_run(run_id)
        paths = [selected / "ranks" / f"rank{rank}.jsonl"] if rank is not None else sorted((selected / "ranks").glob("rank*.jsonl"))
        rows = [row for path in paths for row in read_jsonl(path)]
        return queried_rows(rows, from_step, to_step, route, rollout_id)

    def queried_rows(rows, from_step, to_step, route, rollout_id):
        return filter_rows(rows, from_step, to_step, route, rollout_id)

    @app.get("/api/traces")
    def traces(
        run_id: str | None = None,
        from_step: int | None = None,
        to_step: int | None = None,
        route: str | None = None,
        rollout_id: int | None = None,
    ):
        rows = []
        trace_dir = selected_run(run_id) / "traces"
        for path in sorted(trace_dir.glob("*.jsonl")):
            rows.extend(read_jsonl(path))
        for path in sorted(trace_dir.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                rows.extend(value if isinstance(value, list) else [value])
            except (OSError, json.JSONDecodeError):
                continue
        rows = queried_rows(rows, from_step, to_step, route, rollout_id)
        enrich_source_fields(rows, source_rows(selected_run(run_id), "train"))
        return rows
    @app.get("/api/advantages")
    def advantages(
        run_id: str | None = None,
        from_step: int | None = None,
        to_step: int | None = None,
        route: str | None = None,
        rollout_id: int | None = None,
        group_id: str | None = None,
        limit: int = Query(default=40, ge=1, le=200),
    ):
        """Reconstruct display-only credit from immutable trace rows."""
        rows = []
        trace_dir = selected_run(run_id) / "traces"
        for path in sorted(trace_dir.glob("*.jsonl")):
            rows.extend(read_jsonl(path))
        for path in sorted(trace_dir.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                rows.extend(value if isinstance(value, list) else [value])
            except (OSError, json.JSONDecodeError):
                continue
        rows = queried_rows(rows, from_step, to_step, route, rollout_id)
        if group_id is not None:
            rows = [row for row in rows if row.get("group_id") == group_id]
        rows = rows[-limit:]
        enrich_source_fields(rows, source_rows(selected_run(run_id), "train"))
        return {
            "read_only": True,
            "provenance": {
                "captured": "训练时直接落盘",
                "reconstructed": "由已落盘数据只读复算，非训练时直接采集",
            },
            "groups": reconstruct_groups(rows),
        }

    @app.get("/api/probes")
    def probes(
        run_id: str | None = None,
        from_step: int | None = None,
        to_step: int | None = None,
        group_id: str | None = None,
    ):
        selected = selected_run(run_id)
        rows = filter_rows(
            read_jsonl(selected / "probes.jsonl"),
            from_step=from_step,
            to_step=to_step,
        )
        enrich_source_fields(rows, source_rows(selected, "probe"), one_per_source=True)
        if group_id is not None:
            rows = [row for row in rows if row.get("group_id") == group_id]
        return rows

    def dsr_rows(filename, run_id, from_step, to_step, route, rollout_id):
        return queried(
            selected_run(run_id) / filename,
            from_step,
            to_step,
            route,
            rollout_id,
        )

    @app.get("/api/dsr/metrics")
    def dsr_metrics(
        run_id: str | None = None,
        from_step: int | None = None,
        to_step: int | None = None,
        route: str | None = None,
        rollout_id: int | None = None,
    ):
        return dsr_rows("dsr_metrics.jsonl", run_id, from_step, to_step, route, rollout_id)

    @app.get("/api/dsr/steps")
    def dsr_steps(
        run_id: str | None = None,
        from_step: int | None = None,
        to_step: int | None = None,
        route: str | None = None,
        rollout_id: int | None = None,
    ):
        return dsr_rows("dsr_steps.jsonl", run_id, from_step, to_step, route, rollout_id)

    @app.get("/api/dsr/traces")
    def dsr_traces(
        run_id: str | None = None,
        from_step: int | None = None,
        to_step: int | None = None,
        route: str | None = None,
        rollout_id: int | None = None,
    ):
        return dsr_rows("dsr_traces.jsonl", run_id, from_step, to_step, route, rollout_id)

    @app.get("/api/dsr/gate")
    def dsr_gate(run_id: str | None = None):
        selected = selected_run(run_id)
        metrics = read_jsonl(selected / "metrics.jsonl")
        current_step = int(metrics[-1].get("step", 0)) if metrics else 0
        try:
            report = json.loads((selected / "gate200_report.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = {
                "gate_step": 200,
                "decision": "PENDING",
                "reasons": [],
                "warnings": [],
                "metrics": {},
            }
        return {**report, "current_step": current_step}

    @app.get("/api/health")
    def health():
        return {"ok": True, "mode": "single" if single_run is not None else "multi", "runs_dir": str(root)}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a GRPO monitor run")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", help="Serve one run (backward-compatible mode)")
    source.add_argument("--runs-dir", help="Serve an experiment list rooted at this directory")
    parser.add_argument("--outputs-dir", help="Formal output root used for checkpoint adapter downloads")
    parser.add_argument("--user-runs-dir", help="Approved User-GRPO run root declared by monitor manifests")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    import uvicorn

    print(f"GRPO Monitor: http://{args.host}:{args.port}", flush=True)
    uvicorn.run(
        create_app(args.run_dir, runs_dir=args.runs_dir, outputs_dir=args.outputs_dir, user_runs_dir=args.user_runs_dir),
        host=args.host, port=args.port, log_level="warning",
    )


if __name__ == "__main__":
    main()
