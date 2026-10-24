"""Writes a robot stage to disk as text (.usda) or, when ``pxr`` is available, binary crate."""

from __future__ import annotations

import os
import tempfile
from typing import List

from ..core.report import ExportReport
from ..model.robot import RobotModel
from ..options import ExportOptions
from .builder import UsdRobotBuilder

_BINARY_EXTS = (".usd", ".usdc")


class UsdWriter:
    def __init__(self, options: ExportOptions, report: ExportReport) -> None:
        self.options = options
        self.report = report

    def write(self, model: RobotModel, path: str) -> List[str]:
        """Write ``model`` to ``path`` and return the written file paths."""
        stage = UsdRobotBuilder(self.options, self.report).build(model)
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in _BINARY_EXTS:
            if ext != ".usda":
                self.report.warn(f"unknown USD extension {ext!r}; writing USDA text")
            stage.save(path)
            return [path]

        try:
            from pxr import Sdf  # type: ignore[import-not-found]
        except ImportError:
            fallback = root + ".usda"
            self.report.warn(f"pxr is not available for binary {ext} output; wrote {fallback} instead")
            stage.save(fallback)
            return [fallback]

        fd, tmp = tempfile.mkstemp(suffix=".usda", dir=directory)
        os.close(fd)
        try:
            stage.save(tmp)
            layer = Sdf.Layer.FindOrOpen(tmp)
            if layer is None or not layer.Export(path, args={"format": "usdc"}):
                raise RuntimeError(f"failed to export {path}")
        finally:
            os.remove(tmp)
        return [path]
