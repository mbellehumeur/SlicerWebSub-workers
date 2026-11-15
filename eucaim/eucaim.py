#!/usr/bin/env python3
"""Back-compat entry for EUCAIM — prefer ``eucaim_docker.py``.

Inference is Docker-only; this module forwards to ``eucaim_docker``.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().with_name("eucaim_docker.py")
    print(
        "eucaim/eucaim.py is a compatibility alias; "
        "prefer: python eucaim/eucaim_docker.py",
        file=sys.stderr,
    )
    runpy.run_path(str(target), run_name="__main__")
