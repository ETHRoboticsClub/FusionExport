"""Registers the Fusion2USD commands in the Design workspace."""

from __future__ import annotations

from typing import List

import adsk.core

from .base import CommandBase
from .export_command import ExportCommand
from .properties_command import JointPropertiesCommand, LinkPropertiesCommand

WORKSPACE_ID = "FusionSolidEnvironment"
PANEL_ID = "SolidScriptsAddinsPanel"  # UTILITIES > ADD-INS


class AddIn:
    def __init__(self) -> None:
        self.commands: List[CommandBase] = [ExportCommand(), JointPropertiesCommand(), LinkPropertiesCommand()]

    def start(self) -> None:
        ui = adsk.core.Application.get().userInterface
        workspace = ui.workspaces.itemById(WORKSPACE_ID)
        panel = workspace.toolbarPanels.itemById(PANEL_ID) if workspace is not None else None
        for cmd in self.commands:
            cmd.register(panel)

    def stop(self) -> None:
        for cmd in self.commands:
            cmd.unregister()
