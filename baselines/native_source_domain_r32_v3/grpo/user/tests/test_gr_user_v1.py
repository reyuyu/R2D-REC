import argparse
import collections
import json
import random
import re
import sys
import unittest
from pathlib import Path

from transformers import AutoTokenizer


SID_RE = re.compile(r"<\|(?:video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")
ACTION_EXPECTED = {"1-5": 225, "6-10": 300, "11-20": 450, "21-30": 300, "31-40": 150, "41+": 75}
CHAIN_EXPECTED = {"2": 225, "3": 825, "4": 375, "5": 75}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def final_json_text(output: str) -> str:
    return output.split("</think>", 1)[-1].strip()


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = Path(DATA_DIR)
        cls.train = read_jsonl(cls.data_dir / "train_3000.jsonl")
        cls.pilot = read_jsonl(cls.data_dir / "pilot_600.jsonl")
        cls.probe = read_jsonl(cls.data_dir / "probe_v1.jsonl")
        cls.manifest = json.loads((cls.data_dir / "manifest.json").read_text(encoding="utf-8"))
        cls.tokenizer = AutoTokenizer.from_pretrained(PARENT_CHECKPOINT, local_files_only=True)
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        sys.path.insert(0, str(scripts))
        from user_prompt_adapter import split_source_prompt
        cls.split_source_prompt = staticmethod(split_source_prompt)

    def test_core_counts_and_disjointness(self):
        self.assertEqual(len(self.train), 3000)
        self.assertEqual(sum(row["route"] == "action" for row in self.train), 1500)
        self.assertEqual(sum(row["route"] == "chain" for row in self.train), 1500)
        self.assertEqual(sum(row["converted_from_cot"] for row in self.train), 300)
        self.assertEqual(len(self.pilot), 600)
        self.assertEqual(len(self.probe), 20)
        train_ids = {row["sample_id"] for row in self.train}
        pilot_ids = {row["sample_id"] for row in self.pilot}
        probe_ids = {row["sample_id"] for row in self.probe}
        self.assertTrue(pilot_ids < train_ids)
        self.assertTrue(train_ids.isdisjoint(probe_ids))

    def test_buckets(self):
        action = collections.Counter(str(row["bucket"]) for row in self.train if row["route"] == "action")
        chain = collections.Counter(str(row["bucket"]) for row in self.train if row["route"] == "chain")
        self.assertEqual(dict(action), ACTION_EXPECTED)
        self.assertEqual(dict(chain), CHAIN_EXPECTED)

    def test_action_grounding_and_contract(self):
        for row in (item for item in self.train + self.probe if item["route"] == "action"):
            self.assertTrue(row["gold_sids"])
            self.assertEqual(len(row["gold_sids"]), len(set(row["gold_sids"])))
            self.assertTrue(set(row["gold_sids"]).issubset(set(row["history_sids"])))
            self.assertTrue(row["prompt"].endswith("/no_think"))
            self.assertIn("以下案例来自其他用户", row["prompt"])
            self.assertNotIn("<|im_start|>", row["prompt"])

    def test_chain_grounding_chronology_and_contract(self):
        for row in (item for item in self.train + self.probe if item["route"] == "chain"):
            events = row["gold_events"]
            self.assertIn(len(events), (2, 3, 4, 5))
            self.assertEqual([event["date"] for event in events], sorted(event["date"] for event in events))
            keys = [(event["date"], event["action"]) for event in events]
            self.assertEqual(len(keys), len(set(keys)))
            history = collections.defaultdict(set)
            for event in row["history_events"]:
                history[event["date"]].add(event["raw"])
            for event in events:
                for part in event["action"].split("；"):
                    self.assertIn(part.strip(), history[event["date"]])
            self.assertIn("有效逻辑链案例", row["prompt"])
            self.assertTrue(row["prompt"].endswith("/no_think"))

    def test_converted_rows_preserve_source_final_and_context(self):
        converted = [row for row in self.train if row["converted_from_cot"]]
        chosen = random.Random(20260819).sample(converted, 30)
        by_file = collections.defaultdict(list)
        for row in chosen:
            by_file[row["source_file"]].append(row)
        for filename, rows in by_file.items():
            wanted = {row["source_line"]: row for row in rows}
            with Path(filename).open(encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    if index not in wanted:
                        continue
                    source = json.loads(line)
                    row = wanted[index]
                    self.assertEqual(row["raw_gold_output"], final_json_text(source["output"]))
                    source_history, source_topic = self.split_source_prompt(source["input"])
                    adapted_history, adapted_topic = self.split_source_prompt(row["prompt"])
                    self.assertEqual(source_history, adapted_history)
                    self.assertEqual(source_topic, adapted_topic)
                    self.assertNotIn("</think>", row["raw_gold_output"])
                    self.assertTrue(row["prompt"].endswith("/no_think"))

    def test_parent_renderer_contract(self):
        categories = {
            "action": [row for row in self.train if row["route"] == "action"][:20],
            "chain_native": [row for row in self.train if row["route"] == "chain" and not row["converted_from_cot"]][:20],
            "chain_converted": [row for row in self.train if row["converted_from_cot"]][:20],
        }
        for rows in categories.values():
            self.assertEqual(len(rows), 20)
            for row in rows:
                rendered = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                self.assertEqual(rendered.count("<|im_start|>user"), 1)
                self.assertEqual(rendered.count("<|im_start|>assistant"), 1)
                self.assertEqual(rendered.count("<think>"), 1)
                self.assertEqual(rendered.count("</think>"), 1)
                self.assertIn("/no_think<|im_end|>", rendered)
                self.assertLessEqual(len(self.tokenizer.encode(rendered, add_special_tokens=False)), 8192)
                for sid in SID_RE.findall(row["prompt"])[:10]:
                    self.assertIn(sid, self.tokenizer.decode(self.tokenizer.encode(sid, add_special_tokens=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    args, remaining = parser.parse_known_args()
    DATA_DIR = args.data_dir
    PARENT_CHECKPOINT = args.parent_checkpoint
    unittest.main(argv=[sys.argv[0], *remaining])
