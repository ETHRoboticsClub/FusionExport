"""Collects user-facing diagnostics during an export."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger("fusion2usd")


@dataclass
class ExportReport:
    infos: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)  # the export finished, but something it was asked for did not
    outputs: List[str] = field(default_factory=list)

    def info(self, msg: str) -> None:
        logger.info(msg)
        self.infos.append(msg)

    def warn(self, msg: str) -> None:
        logger.warning(msg)
        self.warnings.append(msg)

    def error(self, msg: str) -> None:
        logger.error(msg)
        self.errors.append(msg)

    def summary(self) -> str:
        lines = [f"Wrote: {p}" for p in self.outputs]
        lines += self.infos
        if self.warnings:
            lines.append(f"{len(self.warnings)} warning(s):")
            lines += [f"  - {w}" for w in self.warnings]
        if self.errors:
            lines.append(f"{len(self.errors)} error(s):")
            lines += [f"  - {e}" for e in self.errors]
        return "\n".join(lines)
