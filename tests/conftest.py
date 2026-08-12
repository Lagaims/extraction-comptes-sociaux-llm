"""Configuration pytest partagée.

Rend les modules de `scripts/` et `libs/src/` importables depuis les tests,
sans avoir à les packager.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for path in (ROOT / "scripts", ROOT / "libs" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
