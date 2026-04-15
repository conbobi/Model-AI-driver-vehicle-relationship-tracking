from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PACKAGE_PARENT = PACKAGE_DIR.parent

for candidate in (PACKAGE_PARENT, PACKAGE_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.append(candidate_str)

# Compatibility alias for files that still use `from TPHYolov5...`.
sys.modules.setdefault("TPHYolov5", sys.modules[__name__])
