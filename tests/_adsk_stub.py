"""A stand-in ``adsk`` package so Fusion-facing modules can be imported and exercised without Fusion.

Any attribute of ``adsk.core`` / ``adsk.fusion`` is an auto-created class; any attribute of such a
class is a distinct string, so enum members compare and hash consistently.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Dict, Iterator

FUSION_MODULE_PREFIXES = ("fusion2usd.fusion.", "fusion2usd.ui")
PURE = {"fusion2usd.fusion.grouping", "fusion2usd.fusion.probe_math", "fusion2usd.fusion.units"}


class _AutoMeta(type):
    def __getattr__(cls, name: str) -> str:
        if name.startswith("__"):
            raise AttributeError(name)
        value = f"{cls.__name__}.{name}"
        setattr(cls, name, value)
        return value


class _StubModule(ModuleType):
    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        cls = _AutoMeta(name, (object,), {"__init__": lambda self, *a, **k: None})
        setattr(self, name, cls)
        return cls


def install() -> Dict[str, ModuleType]:
    adsk = ModuleType("adsk")
    adsk.__path__ = []  # type: ignore[attr-defined]
    core, fusion = _StubModule("adsk.core"), _StubModule("adsk.fusion")
    adsk.core, adsk.fusion = core, fusion  # type: ignore[attr-defined]
    mods = {"adsk": adsk, "adsk.core": core, "adsk.fusion": fusion}
    sys.modules.update(mods)
    return mods


def uninstall() -> None:
    for name in ("adsk", "adsk.core", "adsk.fusion"):
        sys.modules.pop(name, None)
    for name in list(_fusion_modules()):
        del sys.modules[name]


def _fusion_modules() -> Iterator[str]:
    for name in sys.modules:
        if name.startswith(FUSION_MODULE_PREFIXES) and name not in PURE:
            yield name
