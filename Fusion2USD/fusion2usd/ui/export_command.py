"""The "Export to USD" dialog."""

from __future__ import annotations

import os
import traceback
from typing import Any

import adsk.core
import adsk.fusion

from ..options import CollisionApproximation, ExportOptions, MeshQuality, PhysicsEngine, ZeroPose
from .base import CommandBase
from .inputs import add_dropdown, bool_value, dropdown_value, select_dropdown, string_value
from .settings import OptionsStore

ENGINE = [("Newton", PhysicsEngine.NEWTON.value), ("PhysX (Isaac Sim / Isaac Lab)", PhysicsEngine.PHYSX.value)]
UP_AXIS = [("Auto (modeling preference)", "auto"), ("Y up", "Y"), ("Z up", "Z")]
ZERO_POSE = [("Fusion joint zero", ZeroPose.FUSION.value), ("Current pose", ZeroPose.CURRENT.value)]
FIXED_BASE = [("Auto (grounded in Fusion)", "auto"), ("Fixed", "fixed"), ("Floating", "floating")]
QUALITY = [("Low", MeshQuality.LOW.value), ("Normal", MeshQuality.NORMAL.value),
           ("High", MeshQuality.HIGH.value), ("Very high", MeshQuality.VERY_HIGH.value)]
COLLISION_QUALITY = [("Same as visual (no extra copy)", MeshQuality.VISUAL.value)] + QUALITY
APPROX = [(a.value, a.value) for a in CollisionApproximation]
FORMAT = [("Text (.usda)", "usda"), ("Binary (.usdc)", "usdc"), ("Text, meshes in a binary layer", "layered")]

_FLAGS = (  # (input id, ExportOptions field, label)
    ("probe", "probe_joints", "Probe joint axes"),
    ("mimic", "export_mimic_joints", "Export motion links as mimic joints"),
    ("isaaclab", "write_isaaclab_cfg", "Write Isaac Lab config"),
    ("urdf", "write_urdf", "Write URDF"),
    ("model_json", "write_model_json", "Write model JSON"),
    ("hidden", "include_hidden", "Include hidden bodies"),
    ("self_collisions", "self_collisions", "Enable self-collisions"),
)

USD_FILTER = "USD files (*.usd *.usda *.usdc);;All files (*.*)"
JSON_FILTER = "JSON files (*.json);;All files (*.*)"


class ExportCommand(CommandBase):
    id = "fusion2usd_export"
    name = "Export to USD"
    tooltip = "Export the active design as a USD articulation for Isaac Sim / Isaac Lab."

    def __init__(self) -> None:
        super().__init__()
        self.store = OptionsStore()

    def on_create(self, command: Any, inputs: Any) -> None:
        opts = self.store.load()
        design = adsk.fusion.Design.cast(self.app.activeProduct)
        if not opts.output_path and design is not None:
            name = design.parentDocument.name if design.parentDocument else "robot"
            opts.output_path = os.path.join(os.path.expanduser("~"), f"{name}.usda")
        command.setDialogInitialSize(480, 560)

        inputs.addStringValueInput("output", "Output file", opts.output_path)
        inputs.addBoolValueInput("browse_output", "Browse…", False, "", False)
        add_dropdown(inputs, "format", "USD format", FORMAT, usd_format(opts.output_path, opts.mesh_layer))
        inputs.addStringValueInput("robot_name", "Robot name", opts.robot_name or "")
        add_dropdown(inputs, "engine", "Physics engine", ENGINE, opts.physics_engine.value)
        add_dropdown(inputs, "zero_pose", "Joint zero", ZERO_POSE, opts.zero_pose.value)
        up = opts.source_up_axis.upper() if opts.source_up_axis.upper() in ("Y", "Z") else "auto"
        add_dropdown(inputs, "up_axis", "Source up axis", UP_AXIS, up)
        fixed = "auto" if opts.fixed_base is None else ("fixed" if opts.fixed_base else "floating")
        add_dropdown(inputs, "fixed_base", "Base", FIXED_BASE, fixed)
        add_dropdown(inputs, "visual_quality", "Visual mesh quality", QUALITY, opts.visual_quality.value)
        add_dropdown(inputs, "collision_quality", "Collision mesh quality", COLLISION_QUALITY,
                     opts.collision_quality.value)
        inputs.addIntegerSpinnerCommandInput("max_triangles", "Max triangles per body (0: no limit)", 0, 10_000_000,
                                             1000, opts.max_triangles)
        add_dropdown(inputs, "collision_approx", "Collision approximation", APPROX,
                     opts.collision_approximation.value)
        for input_id, field, label in _FLAGS:
            inputs.addBoolValueInput(input_id, label, True, "", bool(getattr(opts, field)))
        inputs.addStringValueInput("post_actions", "Extra post-export actions (comma-separated)",
                                   ", ".join(opts.post_actions))
        inputs.addStringValueInput("overrides", "Overrides JSON", opts.overrides_path or "")
        inputs.addBoolValueInput("browse_overrides", "Browse overrides…", False, "", False)
        inputs.addStringValueInput("urdf_mesh_uri", "URDF mesh URI (package://pkg/urdf; empty: relative)",
                                   opts.urdf_mesh_uri)

    def on_input_changed(self, changed: Any, inputs: Any) -> None:
        if changed.id == "browse_output":
            path = self._pick_file("Export USD", USD_FILTER, string_value(inputs, "output"), save=True)
            if path:
                inputs.itemById("output").value = path
                layered = dropdown_value(inputs, "format", FORMAT) == "layered"
                select_dropdown(inputs, "format", FORMAT, usd_format(path, layered and path.lower().endswith(".usda")))
        elif changed.id == "format":
            output = inputs.itemById("output")
            output.value = with_format(output.value.strip(), dropdown_value(inputs, "format", FORMAT))
        elif changed.id == "browse_overrides":
            path = self._pick_file("Overrides JSON", JSON_FILTER, string_value(inputs, "overrides"), save=False)
            if path:
                inputs.itemById("overrides").value = path

    def validate(self, inputs: Any) -> bool:
        return string_value(inputs, "output").lower().endswith((".usd", ".usda", ".usdc"))

    def on_execute(self, command: Any, inputs: Any) -> None:
        design = adsk.fusion.Design.cast(self.app.activeProduct)
        if design is None:
            self.message("The active document is not a Fusion design.")
            return
        opts = self._options(inputs)
        self.store.save(opts)
        try:
            from ..exporter import Exporter  # deferred: heavy and only needed on execute
            report = Exporter(opts).export_design(design)
        except Exception:
            self.message(f"Export failed:\n\n{traceback.format_exc()}")
            return
        self.message(report.brief() or "Export finished.")

    def _options(self, inputs: Any) -> ExportOptions:
        opts = self.store.load()
        fmt = dropdown_value(inputs, "format", FORMAT)
        opts.output_path = with_format(string_value(inputs, "output"), fmt)
        opts.mesh_layer = fmt == "layered"
        opts.robot_name = string_value(inputs, "robot_name") or None
        opts.physics_engine = PhysicsEngine(dropdown_value(inputs, "engine", ENGINE))
        opts.zero_pose = ZeroPose(dropdown_value(inputs, "zero_pose", ZERO_POSE))
        opts.source_up_axis = dropdown_value(inputs, "up_axis", UP_AXIS)
        opts.fixed_base = {"auto": None, "fixed": True, "floating": False}[
            dropdown_value(inputs, "fixed_base", FIXED_BASE)]
        opts.visual_quality = MeshQuality(dropdown_value(inputs, "visual_quality", QUALITY))
        opts.collision_quality = MeshQuality(dropdown_value(inputs, "collision_quality", COLLISION_QUALITY))
        opts.max_triangles = int(inputs.itemById("max_triangles").value)
        opts.collision_approximation = CollisionApproximation(dropdown_value(inputs, "collision_approx", APPROX))
        for input_id, field, _ in _FLAGS:
            setattr(opts, field, bool_value(inputs, input_id))
        opts.post_actions = [a.strip() for a in string_value(inputs, "post_actions").split(",") if a.strip()]
        opts.overrides_path = string_value(inputs, "overrides") or None
        opts.urdf_mesh_uri = string_value(inputs, "urdf_mesh_uri")
        return opts

    def _pick_file(self, title: str, filter_: str, current: str, save: bool) -> str:
        dlg = self.ui.createFileDialog()
        dlg.title = title
        dlg.filter = filter_
        if current:
            dlg.initialDirectory = os.path.dirname(current)
            dlg.initialFilename = os.path.basename(current)
        result = dlg.showSave() if save else dlg.showOpen()
        return dlg.filename if result == adsk.core.DialogResults.DialogOK else ""


def usd_format(path: str, mesh_layer: bool) -> str:
    if mesh_layer:
        return "layered"
    return "usdc" if path.lower().endswith((".usd", ".usdc")) else "usda"


def with_format(path: str, fmt: str) -> str:
    """``path`` with the extension the format writes (the root layer is text for "layered")."""
    root, ext = os.path.splitext(path)
    if not root or ext.lower() not in (".usd", ".usda", ".usdc", ""):
        return path
    if fmt == "usdc":
        return path if ext.lower() in (".usd", ".usdc") else root + ".usdc"
    return root + ".usda"
