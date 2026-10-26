"""The add-in must put its buttons on a visible panel (live Fusion showed no export button)."""

from types import SimpleNamespace

import pytest

import _adsk_stub


class Items:
    def __init__(self):
        self.items = {}

    def itemById(self, id):
        return self.items.get(id)


class Control:
    def __init__(self, definition):
        self.definition, self.isValid = definition, True
        self.isPromoted = self.isPromotedByDefault = False

    def deleteMe(self):
        self.isValid = False


class Controls(Items):
    def addCommand(self, definition):
        self.items[definition.id] = Control(definition)
        return self.items[definition.id]


class Panel:
    def __init__(self, id, name):
        self.id, self.name, self.isValid, self.controls = id, name, True, Controls()

    def deleteMe(self):
        self.isValid = False


class Panels(Items):
    def add(self, id, name):
        self.items[id] = Panel(id, name)
        return self.items[id]


class Event:
    def add(self, handler):
        return True


class Definition:
    def __init__(self, id):
        self.id, self.isValid, self.commandCreated = id, True, Event()

    def deleteMe(self):
        self.isValid = False


class Definitions(Items):
    def addButtonDefinition(self, id, name, tooltip, resources=""):
        self.items[id] = Definition(id)
        return self.items[id]


@pytest.fixture
def ui():
    mods = _adsk_stub.install()
    workspace = SimpleNamespace(toolbarPanels=Panels())
    workspaces = Items()
    workspaces.items["FusionSolidEnvironment"] = workspace
    fake_ui = SimpleNamespace(workspaces=workspaces, commandDefinitions=Definitions(), messageBox=print)
    mods["adsk.core"].Application = SimpleNamespace(get=lambda: SimpleNamespace(userInterface=fake_ui))
    yield fake_ui, workspace
    _adsk_stub.uninstall()


def test_addin_creates_promoted_buttons_on_own_panel(ui):
    fake_ui, workspace = ui
    from fusion2usd.ui.addin import PANEL_ID, AddIn

    addin = AddIn()
    addin.start()
    panel = workspace.toolbarPanels.itemById(PANEL_ID)
    assert panel is not None and panel.name == "FUSION2USD"
    controls = list(panel.controls.items.values())
    assert len(controls) == 4 and all(c.isPromoted and c.isPromotedByDefault for c in controls)

    addin.stop()
    assert not panel.isValid and not any(c.isValid for c in controls)


def test_mesh_quality_is_not_remembered(tmp_path):
    from fusion2usd.options import ExportOptions, MeshQuality
    from fusion2usd.ui.settings import OptionsStore

    store = OptionsStore(str(tmp_path / "opts.json"))
    store.save(ExportOptions(robot_name="V1", visual_quality=MeshQuality.HIGH, collision_quality=MeshQuality.HIGH))
    loaded = store.load()
    assert loaded.robot_name == "V1"
    assert (loaded.visual_quality, loaded.collision_quality) == (MeshQuality.NORMAL, MeshQuality.LOW)
