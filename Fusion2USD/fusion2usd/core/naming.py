"""USD-safe, unique identifiers."""

from __future__ import annotations

import re
from typing import Dict, Set

_INVALID = re.compile(r"[^A-Za-z0-9_]")


def sanitize(name: str, fallback: str = "item") -> str:
    """Map an arbitrary CAD name to a valid USD prim name (``[A-Za-z_][A-Za-z0-9_]*``)."""
    s = _INVALID.sub("_", name.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = fallback
    if s[0].isdigit():
        s = "_" + s
    return s


class NameRegistry:
    """Hands out unique sanitized names within one namespace."""

    def __init__(self, reserved: Set[str] | None = None) -> None:
        self._used: Set[str] = set(reserved or ())
        self._counters: Dict[str, int] = {}

    def claim(self, name: str, fallback: str = "item") -> str:
        base = sanitize(name, fallback)
        candidate = base
        while candidate in self._used:
            self._counters[base] = self._counters.get(base, 1) + 1
            candidate = f"{base}_{self._counters[base]}"
        self._used.add(candidate)
        return candidate

    def __contains__(self, name: str) -> bool:
        return name in self._used
