"""Visual token compression (VTC) methods: one module per method.

To add a method: drop a new module in this folder with a ``BaseVTCCompressor`` subclass
decorated with ``@register_compressor("<vtc_method>")``. It is discovered automatically.
"""
import importlib
import pkgutil

from .base import BaseVTCCompressor
from .llm.base import BaseLLMVTCCompressor
from .registry import VTC_REGISTRY, build_compressor, register_compressor

for _module in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_module.name}")


from .llm.pyramid_drop import PyramidDrop

__all__ = ["BaseVTCCompressor", "BaseLLMVTCCompressor", "VTC_REGISTRY", "build_compressor", "register_compressor", "PyramidDrop"]
