"""Writes a robot stage to disk as text (.usda) or, when ``pxr`` is available, binary crate."""

from __future__ import annotations

import os
import tempfile
from typing import List

from ..core.report import ExportReport
from ..model.robot import RobotModel
from ..options import ExportOptions
from ..usd_support import has_pxr
from .builder import UsdRobotBuilder
from .usda import UsdaStage

_BINARY_EXTS = (".usd", ".usdc")
_NO_PXR = "pxr is not available (run Install USD support)"


class UsdWriter:
    def __init__(self, options: ExportOptions, report: ExportReport) -> None:
        self.options = options
        self.report = report

    def write(self, model: RobotModel, path: str) -> List[str]:
        """Write ``model`` to ``path`` and return the written file paths, root layer first."""
        stage = UsdRobotBuilder(self.options, self.report).build(model)
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in _BINARY_EXTS + (".usda",):
            self.report.warn(f"unknown USD extension {ext!r}; writing USDA text")
        elif ext in _BINARY_EXTS and not has_pxr():
            self.report.warn(f"{_NO_PXR} for binary {ext} output; wrote {root}.usda instead")
            path = root + ".usda"
        meshes: List[str] = []
        if self.options.mesh_layer:
            binary = has_pxr()
            if not binary:
                self.report.warn(f"{_NO_PXR}; the mesh layer is USDA text")
            mesh_path = root + ("_meshes.usdc" if binary else "_meshes.usda")
            layer = stage.split_off("Mesh")
            stage.sublayers.append("./" + os.path.basename(mesh_path))
            self._save(layer, mesh_path)
            meshes.append(mesh_path)
        self._save(stage, path)
        return [path] + meshes

    @staticmethod
    def _save(stage: UsdaStage, path: str) -> None:
        if os.path.splitext(path)[1].lower() not in _BINARY_EXTS:
            stage.save(path)
            return
        from pxr import Sdf  # type: ignore[import-not-found]

        fd, tmp = tempfile.mkstemp(suffix=".usda", dir=os.path.dirname(os.path.abspath(path)))
        os.close(fd)
        try:
            stage.save(tmp)
            layer = Sdf.Layer.FindOrOpen(tmp)
            if layer is None or not layer.Export(path, args={"format": "usdc"}):
                raise RuntimeError(f"failed to export {path}")
        finally:
            os.remove(tmp)
