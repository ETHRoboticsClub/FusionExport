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

    def brief(self, shown: int = 5) -> str:
        """A summary short enough for a message box, which cannot scroll; the full one is in the report file."""
        lines = [f"Wrote: {self.outputs[0]}"] if self.outputs else []
        if len(self.outputs) > 2:
            lines.append(f"  and {len(self.outputs) - 2} other file(s)")
        if self.infos:
            lines.append(self.infos[-1])
        for label, items in (("error", self.errors), ("warning", self.warnings)):
            if items:
                lines.append(f"{len(items)} {label}(s):")
                lines += [f"  - {m}" for m in items[:shown]]
                if len(items) > shown:
                    lines.append(f"  ... {len(items) - shown} more")
        if len(self.outputs) > 1:
            lines.append(f"Full report: {self.outputs[-1]}")
        return "\n".join(lines)
