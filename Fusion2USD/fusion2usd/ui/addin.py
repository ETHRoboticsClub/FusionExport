"""Registers the Fusion2USD commands in their own panel on the Design workspace's UTILITIES tab."""

from __future__ import annotations

from typing import Any, List, Optional

import adsk.core

from .. import usd_support
from .base import CommandBase
from .export_command import ExportCommand
from .install_command import InstallUsdCommand
from .properties_command import JointPropertiesCommand, LinkPropertiesCommand, SensorPropertiesCommand

WORKSPACE_ID = "FusionSolidEnvironment"
PANEL_ID = "Fusion2USDPanel"
PANEL_NAME = "FUSION2USD"


class AddIn:
    def __init__(self) -> None:
        self.commands: List[CommandBase] = [ExportCommand(), JointPropertiesCommand(), LinkPropertiesCommand(),
                                           SensorPropertiesCommand(), InstallUsdCommand()]
        self._panel: Optional[Any] = None

    def start(self) -> None:
        usd_support.activate()
        ui = adsk.core.Application.get().userInterface
        workspace = ui.workspaces.itemById(WORKSPACE_ID)
        if workspace is None:
            ui.messageBox(f"Fusion2USD: workspace {WORKSPACE_ID!r} not found; commands are not on the toolbar.")
        else:
            panels = workspace.toolbarPanels
            # Without a tab, Fusion places new panels on the Tools (UTILITIES) tab.
            self._panel = panels.itemById(PANEL_ID) or panels.add(PANEL_ID, PANEL_NAME)
        for cmd in self.commands:
            cmd.register(self._panel)
            cmd.promote()

    def stop(self) -> None:
        for cmd in self.commands:
            cmd.unregister()
        if self._panel is not None and self._panel.isValid:
            self._panel.deleteMe()
        self._panel = None
