"""Dialogs that store per-joint / per-link export properties as ``fusion2usd`` attributes."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence, Tuple

from ..fusion.metadata import JSON_KEY, AttributeReader, AttributeWriter
from ..options import CollisionApproximation
from .base import CommandBase
from .inputs import Choice, add_dropdown, dropdown_value, optional_float, string_value

TRI_STATE = [("Default", ""), ("Yes", "true"), ("No", "false")]


class PropertiesCommand(CommandBase):
    """One selection, a set of string fields, optional dropdowns and a free-form JSON box."""

    selection_filters: Sequence[str] = ()
    prompt = "Select an entity"
    float_fields: Sequence[Tuple[str, str]] = ()    # (attribute name, label)
    int_fields: Sequence[Tuple[str, str]] = ()
    text_fields: Sequence[Tuple[str, str]] = ()
    dropdowns: Sequence[Tuple[str, str, Sequence[Choice]]] = ()  # (attribute name, label, choices)
    checkboxes: Sequence[Tuple[str, str]] = ()      # stored as "true" or removed

    def __init__(self) -> None:
        super().__init__()
        self._reader = AttributeReader()
        self._writer = AttributeWriter()

    def on_create(self, command: Any, inputs: Any) -> None:
        sel = inputs.addSelectionInput("entity", "Entity", self.prompt)
        # Filter names vary across Fusion builds; skip unknown ones instead of failing the dialog.
        added = 0
        for f in self.selection_filters:
            try:
                sel.addSelectionFilter(f)
                added += 1
            except RuntimeError:
                pass
        if self.selection_filters and not added:
            raise RuntimeError(f"no supported selection filter among {self.selection_filters}")
        sel.setSelectionLimits(1, 1)
        for name, label in self._string_fields():
            inputs.addStringValueInput(name, label, "")
        for name, label, choices in self.dropdowns:
            add_dropdown(inputs, name, label, choices, choices[0][1])
        for name, label in self.checkboxes:
            inputs.addBoolValueInput(name, label, True, "", False)
        box = inputs.addTextBoxCommandInput(JSON_KEY, "Extra JSON", "", 4, False)
        box.text = ""

    def on_input_changed(self, changed: Any, inputs: Any) -> None:
        if changed.id == "entity":
            entity = self._selected(inputs)
            if entity is not None:
                self._prefill(inputs, self._reader.read_raw(entity))

    def validate(self, inputs: Any) -> bool:
        if self._selected(inputs) is None:
            return False
        try:
            self._values(inputs)
        except ValueError:
            return False
        return True

    def on_execute(self, command: Any, inputs: Any) -> None:
        entity = self._selected(inputs)
        if entity is None:
            return
        self._writer.write(entity, self._values(inputs))

    # -- helpers
    def _string_fields(self) -> Tuple[Tuple[str, str], ...]:
        return (*self.text_fields, *self.float_fields, *self.int_fields)

    @staticmethod
    def _selected(inputs: Any) -> Optional[Any]:
        sel = inputs.itemById("entity")
        return sel.selection(0).entity if sel is not None and sel.selectionCount else None

    def _prefill(self, inputs: Any, raw: Dict[str, str]) -> None:
        for name, _ in self._string_fields():
            inputs.itemById(name).value = raw.get(name, "")
        for name, _, choices in self.dropdowns:
            current = raw.get(name, "")
            label = next((lbl for lbl, v in choices if v == current), choices[0][0])
            for item in _list_items(inputs.itemById(name)):
                if item.name == label:
                    item.isSelected = True  # single-select list: deselects the others
        for name, _ in self.checkboxes:
            inputs.itemById(name).value = raw.get(name, "").lower() == "true"
        inputs.itemById(JSON_KEY).text = raw.get(JSON_KEY, "")

    def _values(self, inputs: Any) -> Dict[str, Optional[str]]:
        """Attribute values to write; ``None`` deletes. Raises ValueError on invalid input."""
        values: Dict[str, Optional[str]] = {}
        for name, _ in self.text_fields:
            values[name] = string_value(inputs, name) or None
        for name, _ in self.float_fields:
            number = optional_float(string_value(inputs, name))
            values[name] = None if number is None else repr(number)
        for name, _ in self.int_fields:
            text = string_value(inputs, name)
            values[name] = str(int(text)) if text else None
        for name, _, choices in self.dropdowns:
            values[name] = dropdown_value(inputs, name, choices) or None
        for name, _ in self.checkboxes:
            values[name] = "true" if inputs.itemById(name).value else None
        text = inputs.itemById(JSON_KEY).text.strip()
        if text:
            extra = json.loads(text)  # json.JSONDecodeError is a ValueError
            if not isinstance(extra, dict):
                raise ValueError("extra JSON must be an object")
            values[JSON_KEY] = AttributeWriter.encode_json(extra) or None
        else:
            values[JSON_KEY] = None
        return values


def _list_items(dropdown: Any):
    items = dropdown.listItems
    return (items.item(i) for i in range(items.count))


class JointPropertiesCommand(PropertiesCommand):
    id = "fusion2usd_joint_properties"
    name = "USD Joint Properties"
    tooltip = "Set drive/dynamics properties of a joint for the USD export (SI units, empty = default)."
    selection_filters = ("Joints", "AsBuiltJoints")
    prompt = "Select a joint"
    float_fields = (
        ("effort", "Max effort (N·m / N)"),
        ("velocity", "Max velocity (rad/s / m/s)"),
        ("stiffness", "Drive stiffness"),
        ("damping", "Drive damping"),
        ("armature", "Armature"),
        ("friction", "Friction"),
    )
    dropdowns = (("actuated", "Actuated", TRI_STATE),)
    checkboxes = (("loop_closure", "Prefer cutting the loop here"),)


class LinkPropertiesCommand(PropertiesCommand):
    id = "fusion2usd_link_properties"
    name = "USD Link Properties"
    tooltip = "Set the USD link name, a mass override and the collision approximation of an occurrence."
    selection_filters = ("Occurrences",)
    prompt = "Select an occurrence"
    text_fields = (("name", "Link name"),)
    float_fields = (("mass", "Mass override (kg)"),)
    dropdowns = (("collision_approximation", "Collision approximation",
                  [("Default", "")] + [(a.value, a.value) for a in CollisionApproximation]),)


SENSOR_KINDS = [("Camera", "camera"), ("IMU", "imu"), ("Contact", "contact"), ("Frame", "frame"), ("None", "")]
# Fields shown per sensor kind; hidden fields are removed on OK.
SENSOR_FIELDS: Dict[str, Tuple[str, ...]] = {
    "camera": ("width", "height", "fov", "focal_length", "horizontal_aperture", "near", "far", "rate",
               "data_types", JSON_KEY),
    "imu": ("rate", JSON_KEY),
    "contact": ("rate", "history_length", "track_air_time", "force_threshold", JSON_KEY),
    "frame": (JSON_KEY,),
    "": (),
}


class SensorPropertiesCommand(PropertiesCommand):
    id = "fusion2usd_sensor_properties"
    name = "USD Sensor Properties"
    tooltip = ("Tag a joint origin as a camera, IMU, contact sensor or named frame for the USD export "
               "(cameras look along +Z, X right, Y down; SI units, empty = default).")
    selection_filters = ("JointOrigins",)
    prompt = "Select a joint origin"
    text_fields = (("data_types", "Data types (e.g. rgb,depth)"),)
    int_fields = (
        ("width", "Width (px)"),
        ("height", "Height (px)"),
        ("history_length", "History length"),
    )
    float_fields = (
        ("fov", "Horizontal FOV (deg)"),
        ("focal_length", "Focal length"),
        ("horizontal_aperture", "Horizontal aperture"),
        ("near", "Near clip (m)"),
        ("far", "Far clip (m)"),
        ("rate", "Update rate (Hz)"),
        ("force_threshold", "Force threshold (N)"),
    )
    dropdowns = (
        ("sensor", "Type", SENSOR_KINDS),
        ("track_air_time", "Track air time", TRI_STATE),
    )

    def on_create(self, command: Any, inputs: Any) -> None:
        super().on_create(command, inputs)
        self._update_visibility(inputs)

    def on_input_changed(self, changed: Any, inputs: Any) -> None:
        super().on_input_changed(changed, inputs)
        if changed.id in ("entity", "sensor"):
            self._update_visibility(inputs)

    def _values(self, inputs: Any) -> Dict[str, Optional[str]]:
        values = super()._values(inputs)
        if not self._kind(inputs):  # None: drop every fusion2usd attribute, including unknown ones
            entity = self._selected(inputs)
            values.update(dict.fromkeys(self._reader.read_raw(entity) if entity is not None else ()))
        shown = SENSOR_FIELDS[self._kind(inputs)]
        return {k: (v if k == "sensor" or k in shown else None) for k, v in values.items()}

    def _update_visibility(self, inputs: Any) -> None:
        shown = SENSOR_FIELDS[self._kind(inputs)]
        for name in self._field_names():
            inputs.itemById(name).isVisible = name in shown

    @staticmethod
    def _kind(inputs: Any) -> str:
        return dropdown_value(inputs, "sensor", SENSOR_KINDS)

    def _field_names(self) -> Tuple[str, ...]:
        names = [n for n, _ in self._string_fields()] + [n for n, _, _ in self.dropdowns if n != "sensor"]
        return (*names, JSON_KEY)
