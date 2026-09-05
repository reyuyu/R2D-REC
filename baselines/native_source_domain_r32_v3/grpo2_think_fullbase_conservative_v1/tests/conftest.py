import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
SCRIPTS = PACKAGE / "scripts"
FULLBASE = PACKAGE.parent / "grpo_fullbase_conservative_v1" / "scripts"
GRPO = PACKAGE.parent / "grpo" / "scripts"
for path in reversed((SCRIPTS, FULLBASE, GRPO)):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
