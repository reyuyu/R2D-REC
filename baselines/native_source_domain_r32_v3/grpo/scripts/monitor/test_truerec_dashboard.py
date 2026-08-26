import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from truerec_dashboard import install_truerec_routes, read_json, read_jsonl


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def jsonl(path: Path, rows, partial=False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(text + ('{"partial":' if partial else ""), encoding="utf-8")


def build_run(root: Path) -> Path:
    run = root / "TRUEREC-MOCK"
    dump(run / "live_state.json", {
        "current_step": 1, "total_steps": 4096, "current_group_id": "group-1",
        "domain": "video", "rank_health": [{"rank": i, "healthy": True} for i in range(4)],
    })
    group = {"global_step": 1, "recommendation_group_id": "group-1", "target_domain": "video", "total_value": .2}
    jsonl(run / "train_groups.jsonl", [group], partial=True)
    explain = {
        "global_step": 1, "recommendation_group_id": "group-1", "global_candidate_indices": list(range(8)),
        "candidates": [{"candidate_index": i, "source_rank": i // 2, "completion_ids": [1, 2, 3], "action_tokens": []} for i in range(8)],
        "hpr_trigger": "HPR_A", "hpr_sites": [],
    }
    jsonl(run / "train_explain.jsonl", [explain], partial=True)
    probe = run / "probe" / "step0"
    dump(probe / "summary.json", {"probe_step": 0, "group_count": 20})
    jsonl(probe / "groups.jsonl", [{"probe_step": 0, "recommendation_group_id": "group-1"}])
    jsonl(probe / "explain.jsonl", [{**explain, "mode": "PROBE", "optimizer_update": False}])
    checkpoint = run / "checkpoints" / "checkpoint-step-256"
    dump(checkpoint / "metadata.json", {"global_step": 256, "next_group_index": 256, "world_size": 4})
    return run


class TrueRecDashboardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_concurrent_readers_ignore_partial_tail(self):
        path = self.tmp_path / "rows.jsonl"
        jsonl(path, [{"step": 1}, {"step": 2}], partial=True)
        self.assertEqual(read_jsonl(path), [{"step": 1}, {"step": 2}])
        (self.tmp_path / "live.json").write_text('{"incomplete":', encoding="utf-8")
        self.assertEqual(read_json(self.tmp_path / "live.json", {"waiting": True}), {"waiting": True})

    def test_truerec_read_only_api_and_schemas(self):
        root = self.tmp_path / "runs"
        run = build_run(root)
        app = FastAPI()
        install_truerec_routes(app, root, Path(__file__).parent / "static")
        client = TestClient(app)
        watched = [run / "live_state.json", run / "train_groups.jsonl", run / "train_explain.jsonl"]
        before = {str(path): (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()) for path in watched}
        self.assertEqual(client.get("/api/truerec/runs").json()[0]["run_id"], run.name)
        overview = client.get("/api/truerec/overview", params={"run_id": run.name}).json()
        self.assertEqual(overview["live"]["current_step"], 1)
        self.assertEqual(overview["probe_steps"], [0])
        self.assertEqual(client.get("/api/truerec/curves", params={"run_id": run.name}).json()["count"], 1)
        explain = client.get("/api/truerec/train/explain/1", params={"run_id": run.name}).json()
        self.assertEqual(explain["global_candidate_indices"], list(range(8)))
        self.assertEqual([row["candidate_index"] for row in explain["candidates"]], list(range(8)))
        probes = client.get("/api/truerec/probes", params={"run_id": run.name}).json()
        self.assertEqual(probes[0]["summary"]["group_count"], 20)
        comparison = client.get("/api/truerec/probe/compare", params={"run_id": run.name, "step_a": 0, "step_b": 0, "group_id": "group-1"}).json()
        self.assertIs(comparison["a"]["optimizer_update"], False)
        checkpoints = client.get("/api/truerec/checkpoints", params={"run_id": run.name}).json()
        self.assertEqual(checkpoints[0]["step"], 256)
        self.assertEqual(checkpoints[0]["world_size"], 4)
        after = {str(path): (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()) for path in watched}
        self.assertEqual(after, before)

    def test_missing_run_and_missing_probe_are_safe(self):
        app = FastAPI()
        install_truerec_routes(app, self.tmp_path / "missing", Path(__file__).parent / "static")
        client = TestClient(app)
        self.assertEqual(client.get("/api/truerec/runs").json(), [])
        self.assertEqual(client.get("/api/truerec/overview", params={"run_id": "missing"}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
