"""The "Install USD support" command: fetches usd-core so the add-in can write binary .usdc."""

from __future__ import annotations

import traceback
from typing import Any

import adsk.core

from .. import usd_support
from .base import CommandBase


class InstallUsdCommand(CommandBase):
    id = "fusion2usd_install_usd"
    name = "Install USD support"
    tooltip = "Download usd-core from PyPI so exports can write binary .usdc."

    def promote(self) -> None:
        pass  # rarely needed: keep it in the panel's menu

    def on_create(self, command: Any, inputs: Any) -> None:
        pass  # no inputs: Fusion runs execute right away

    def on_execute(self, command: Any, inputs: Any) -> None:
        installed = usd_support.has_pxr()
        question = ("usd-core is already installed. Download the latest version?" if installed else
                    "Download usd-core from PyPI (up to 40 MB, about 250 MB unpacked) into\n"
                    f"{usd_support.lib_dir()}?\n\nFusion stays busy until it is done.")
        answer = self.ui.messageBox(question, self.name, adsk.core.MessageBoxButtonTypes.YesNoButtonType)
        if answer != adsk.core.DialogResults.DialogYes:
            return
        try:
            version = usd_support.install()
        except Exception:
            self.message(f"Installing usd-core failed:\n\n{traceback.format_exc()}")
            return
        if installed:
            self.message(f"Installed usd-core {version}. Restart Fusion to use it.")
        elif usd_support.has_pxr():
            self.message(f"Installed usd-core {version}. Exports can now write .usdc.")
        else:
            self.message(f"Installed usd-core {version}, but it does not load in Fusion:\n\n"
                         f"{usd_support.import_error()}")
