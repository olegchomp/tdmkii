"""
Flux2KleinPipeline for Flux Klein 4B (vendored from huggingface/diffusers main).
Uses _diffusers_tmp/src when Flux2KleinPipeline is first accessed (lazy).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DIFFUSERS_SRC = _REPO_ROOT / "_diffusers_main" / "src"


def __getattr__(name: str):
    if name != "Flux2KleinPipeline":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if _DIFFUSERS_SRC.exists():
        path_str = str(_DIFFUSERS_SRC)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        from diffusers import Flux2KleinPipeline as _Kl
        globals()["Flux2KleinPipeline"] = _Kl  # cache for next access
        return _Kl
    raise ImportError(
        "Flux2KleinPipeline requires diffusers main. Run in repo root:\n"
        "  git clone --depth 1 https://github.com/huggingface/diffusers.git _diffusers_main\n"
        f"Expected path: {_DIFFUSERS_SRC}"
    )


__all__ = ["Flux2KleinPipeline"]
