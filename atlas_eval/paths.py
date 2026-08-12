"""Locate the server-side ``vq_2608`` source tree.

The atlas model, data bundles and loaders live in a separate repository on the
training server (``/stor/znx/vq_2608``).  This module puts that tree on
``sys.path`` so the scorecard can reuse the exact loaders training used instead
of re-implementing preprocessing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_REPO_ROOT = "/stor/znx/vq_2608"


def vq2608_root(explicit: str | None = None) -> Path:
    root = Path(explicit or os.environ.get("VQ2608_REPO_ROOT", DEFAULT_REPO_ROOT)).expanduser()
    if not (root / "src").is_dir():
        raise FileNotFoundError(
            f"{root} does not look like the vq_2608 repository (no src/). "
            "Set VQ2608_REPO_ROOT or pass --vq2608-root."
        )
    return root


def ensure_on_path(explicit: str | None = None) -> Path:
    root = vq2608_root(explicit)
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return root


__all__ = ["DEFAULT_REPO_ROOT", "ensure_on_path", "vq2608_root"]
