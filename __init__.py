"""
ComfyUI-YOLOE26
===============
Open-vocabulary prompt segmentation for ComfyUI powered by YOLOE-26.

See README.md for provenance and attribution notes.
"""

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
except ImportError:
    # Gracefully skip when loaded outside of a package context (e.g. pytest).
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
