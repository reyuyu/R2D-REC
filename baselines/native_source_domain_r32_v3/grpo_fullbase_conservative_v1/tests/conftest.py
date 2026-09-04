import sys
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
SCRIPTS = PACKAGE / "scripts"
LEGACY = PACKAGE.parent / "grpo" / "scripts"
for path in (SCRIPTS, LEGACY):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
