"""Configuration pytest partagée.

Rend les modules de `scripts/`, `libs/src/` et `website/` importables depuis les tests,
sans avoir à les packager. `website/build_data.py` y est inclus parce qu'il porte la
classification des cellules dont sortent les chiffres publiés sur le site.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for path in (ROOT / "scripts", ROOT / "libs" / "src", ROOT / "website"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
