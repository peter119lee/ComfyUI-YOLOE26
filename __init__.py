"""
ComfyUI-YOLOE26
===============
Open-vocabulary prompt segmentation for ComfyUI powered by YOLOE-26.

See README.md for provenance and attribution notes.
"""

from __future__ import annotations


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


def _should_tolerate_direct_module_import(exc: ImportError) -> bool:
    return __package__ in (None, "") and "attempted relative import with no known parent package" in str(exc)


try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError as exc:
    if _should_tolerate_direct_module_import(exc):
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}
    else:
        raise RuntimeError(
            "Failed to import ComfyUI-YOLOE26 node registrations from nodes.py. "
            "This is a production import/registration failure, not a safe fallback. "
            "Inspect the chained exception for the missing dependency or broken registration."
        ) from exc
