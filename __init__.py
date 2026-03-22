"""
ComfyUI-YOLOE26
===============
Open-vocabulary prompt segmentation for ComfyUI powered by YOLOE-26.

Core segmentation logic adapted from spawner1145's prompt_segment.py
https://github.com/spawner1145
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
