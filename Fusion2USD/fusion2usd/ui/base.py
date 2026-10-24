"""Reusable wrapper around Fusion command definitions, toolbar buttons and event handlers."""

from __future__ import annotations

import traceback
from typing import Any, List

import adsk.core


def show_error(title: str) -> None:
    ui = adsk.core.Application.get().userInterface
    ui.messageBox(f"{title}\n\n{traceback.format_exc()}", "Fusion2USD",
                  adsk.core.MessageBoxButtonTypes.OKButtonType, adsk.core.MessageBoxIconTypes.CriticalIconType)


class _Created(adsk.core.CommandCreatedEventHandler):
    def __init__(self, owner: "CommandBase") -> None:
        super().__init__()
        self._owner = owner

    def notify(self, args: Any) -> None:
        try:
            self._owner._created(args.command)
        except Exception:
            show_error(f"{self._owner.name}: failed to open the dialog")


class _Execute(adsk.core.CommandEventHandler):
    def __init__(self, owner: "CommandBase") -> None:
        super().__init__()
        self._owner = owner

    def notify(self, args: Any) -> None:
        try:
            self._owner.on_execute(args.command, args.command.commandInputs)
        except Exception:
            show_error(f"{self._owner.name} failed")


class _InputChanged(adsk.core.InputChangedEventHandler):
    def __init__(self, owner: "CommandBase") -> None:
        super().__init__()
        self._owner = owner

    def notify(self, args: Any) -> None:
        try:
            self._owner.on_input_changed(args.input, args.inputs)
        except Exception:
            show_error(f"{self._owner.name}: input handling failed")


class _Validate(adsk.core.ValidateInputsEventHandler):
    def __init__(self, owner: "CommandBase") -> None:
        super().__init__()
        self._owner = owner

    def notify(self, args: Any) -> None:
        try:
            args.areInputsValid = self._owner.validate(args.inputs)
        except Exception:
            args.areInputsValid = False


class _Destroy(adsk.core.CommandEventHandler):
    def __init__(self, owner: "CommandBase") -> None:
        super().__init__()
        self._owner = owner

    def notify(self, args: Any) -> None:
        self._owner._release_command_handlers()


class CommandBase:
    """Subclasses set ``id``/``name``/``tooltip`` and implement ``on_create``/``on_execute``.

    Fusion only keeps weak references to handlers, so they are stored on the instance.
    """

    id = ""
    name = ""
    tooltip = ""
    resource_folder = ""

    def __init__(self) -> None:
        self.app = adsk.core.Application.get()
        self.ui = self.app.userInterface
        self._definition = None
        self._control = None
        self._handlers: List[Any] = []
        self._command_handlers: List[Any] = []

    # -- registration
    def register(self, panel: Any) -> None:
        defs = self.ui.commandDefinitions
        existing = defs.itemById(self.id)
        if existing is not None:
            existing.deleteMe()
        self._definition = defs.addButtonDefinition(self.id, self.name, self.tooltip, self.resource_folder)
        handler = _Created(self)
        self._definition.commandCreated.add(handler)
        self._handlers.append(handler)
        if panel is not None:
            old = panel.controls.itemById(self.id)
            if old is not None:
                old.deleteMe()
            self._control = panel.controls.addCommand(self._definition)

    def unregister(self) -> None:
        if self._control is not None and self._control.isValid:
            self._control.deleteMe()
        if self._definition is not None and self._definition.isValid:
            self._definition.deleteMe()
        self._control = self._definition = None
        self._handlers.clear()
        self._command_handlers.clear()

    # -- events
    def _created(self, command: Any) -> None:
        self._command_handlers = [_Execute(self), _InputChanged(self), _Validate(self), _Destroy(self)]
        execute, changed, validate, destroy = self._command_handlers
        command.execute.add(execute)
        command.inputChanged.add(changed)
        command.validateInputs.add(validate)
        command.destroy.add(destroy)
        self.on_create(command, command.commandInputs)

    def _release_command_handlers(self) -> None:
        self._command_handlers = []

    # -- overridables
    def on_create(self, command: Any, inputs: Any) -> None:
        raise NotImplementedError

    def on_execute(self, command: Any, inputs: Any) -> None:
        raise NotImplementedError

    def on_input_changed(self, changed: Any, inputs: Any) -> None:
        pass

    def validate(self, inputs: Any) -> bool:
        return True

    # -- helpers
    def message(self, text: str) -> None:
        self.ui.messageBox(text, self.name)
