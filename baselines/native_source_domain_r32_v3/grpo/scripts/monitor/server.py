"""Local FastAPI server for an append-only GRPO monitor run."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse


STATIC_DIR = Path(__file__).resolve().parent / "static"


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
) -> FastAPI:
    if (run_dir is None) == (runs_dir is None):
        raise ValueError("exactly one of run_dir or runs_dir is required")
    single_run = Path(run_dir).expanduser().resolve() if run_dir is not None else None
    root = single_run.parent if single_run is not None else Path(runs_dir).expanduser().resolve()
    outputs_root = Path(outputs_dir).expanduser().resolve() if outputs_dir is not None else None
    app = FastAPI(title="GRPO Monitor", docs_url="/api/docs", redoc_url=None)
    app.state.run_dir = single_run
    app.state.runs_dir = root
    app.state.outputs_dir = outputs_root

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
                "start_time": manifest.get("start_time"),
                "max_steps": manifest.get("max_steps"),
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

    @app.get("/")
    def dashboard():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/runs")
    def runs():
        return available_runs()

    @app.get("/api/checkpoints")
    def checkpoints(run_id: str | None = None):
        selected = selected_run(run_id)
        if outputs_root is None:
            return []
        run_output = (outputs_root / selected.name).resolve()
        if run_output.parent != outputs_root or not run_output.is_dir():
            return []
        result = []
        for path in run_output.iterdir():
            match = re.fullmatch(r"checkpoint-(\d+)", path.name)
            if not path.is_dir() or match is None:
                continue
            files = {}
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                candidate = path / name
                if candidate.is_file():
                    files[name] = candidate.stat().st_size
            result.append({"checkpoint": path.name, "step": int(match.group(1)), "files": files})
        return sorted(result, key=lambda item: item["step"])

    @app.get("/api/checkpoints/{checkpoint}/download")
    def download_checkpoint_file(checkpoint: str, file: str, run_id: str | None = None):
        selected = selected_run(run_id)
        allowed = {"adapter_config.json", "adapter_model.safetensors"}
        if outputs_root is None or file not in allowed or re.fullmatch(r"checkpoint-\d+", checkpoint) is None:
            raise HTTPException(status_code=404, detail="checkpoint file not found")
        checkpoint_dir = (outputs_root / selected.name / checkpoint).resolve()
        expected_parent = (outputs_root / selected.name).resolve()
        candidate = (checkpoint_dir / file).resolve()
        if checkpoint_dir.parent != expected_parent or candidate.parent != checkpoint_dir or not candidate.is_file():
            raise HTTPException(status_code=404, detail="checkpoint file not found")
        return FileResponse(candidate, filename=f"{selected.name}-{checkpoint}-{file}")

    @app.get("/api/manifest")
    def manifest(run_id: str | None = None):
        selected = selected_run(run_id)
        try:
            return json.loads((selected / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

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
        return queried_rows(rows, from_step, to_step, route, rollout_id)

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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    import uvicorn

    print(f"GRPO Monitor: http://{args.host}:{args.port}", flush=True)
    uvicorn.run(create_app(args.run_dir, runs_dir=args.runs_dir, outputs_dir=args.outputs_dir), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
