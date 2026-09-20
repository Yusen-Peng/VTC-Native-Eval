from typing import Callable, Dict, Type

from .base import BaseVTCCompressor

VTC_REGISTRY: Dict[str, Type[BaseVTCCompressor]] = {}


def register_compressor(name: str) -> Callable[[Type[BaseVTCCompressor]], Type[BaseVTCCompressor]]:
    """Class decorator: make a compressor selectable via ``vtc_method=name``."""

    def decorator(cls: Type[BaseVTCCompressor]) -> Type[BaseVTCCompressor]:
        if not issubclass(cls, BaseVTCCompressor):
            raise TypeError(f"{cls.__name__} must inherit BaseVTCCompressor")
        if name in VTC_REGISTRY:
            raise ValueError(f"VTC method '{name}' is already registered by {VTC_REGISTRY[name].__name__}")
        VTC_REGISTRY[name] = cls
        return cls

    return decorator


def build_compressor(vtc_method: str = "none", compression_ratio: float = 1.0, **kwargs) -> BaseVTCCompressor:
    if vtc_method not in VTC_REGISTRY:
        raise ValueError(f"Unknown VTC method: {vtc_method} (available: {sorted(VTC_REGISTRY)})")
    return VTC_REGISTRY[vtc_method](compression_ratio=compression_ratio, **kwargs)
