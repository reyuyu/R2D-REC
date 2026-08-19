import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from backfill_user_fixed_probe import append_events


class UserFixedProbeBackfillTests(unittest.TestCase):
    def test_output_is_create_then_explicit_append(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "probes.jsonl"
            append_events(output, [{"step": 0}], append=False)
            with self.assertRaises(FileExistsError):
                append_events(output, [{"step": 10}], append=False)
            append_events(output, [{"step": 20}], append=True)
            self.assertEqual(output.read_text(encoding="utf-8").count("\n"), 2)


if __name__ == "__main__":
    unittest.main()
