"""Built-in post-export actions."""

from __future__ import annotations

from typing import List

from .base import ExportArtifacts, PostExportAction, register_action


@register_action("isaaclab_cfg")
class IsaacLabCfgAction(PostExportAction):
    """Writes ``<name>_cfg.py`` with a ready-to-use Isaac Lab ``ArticulationCfg``."""

    def run(self, artifacts: ExportArtifacts) -> List[str]:
        from ..isaaclab.cfg_writer import IsaacLabCfgWriter

        path = artifacts.sibling("_cfg.py")
        IsaacLabCfgWriter(artifacts.options, artifacts.report).write(artifacts.model, artifacts.usd_path, path)
        return [path]


@register_action("urdf")
class UrdfAction(PostExportAction):
    """Writes ``<name>.urdf`` plus STL meshes from the same model the USD was built from."""

    def run(self, artifacts: ExportArtifacts) -> List[str]:
        from ..urdf.writer import UrdfWriter

        return UrdfWriter(artifacts.options, artifacts.report).write(artifacts.model, artifacts.sibling(".urdf"))
