"""Turns user metadata (Fusion attributes + JSON overrides) into joint/link properties.

Values may carry a unit suffix (``"2.5 Nm"``, ``"180 deg/s"``, ``"15 kgf*cm"``); bare numbers are SI.
Keys that are not recognized are kept and exported as USD ``customData``.
"""

from __future__ import annotations

import fnmatch
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..core.report import ExportReport
from ..model.robot import Joint, JointProperties, Limit, Link, RobotModel
from ..core.inertia import MassProperties
from ..core.math3d import Vec3
from ..options import CollisionApproximation
from .pipeline import ModelPass, PassContext

_UNITS: Dict[str, float] = {
    # angle / angular velocity
    "rad": 1.0, "deg": math.pi / 180.0, "°": math.pi / 180.0,
    "rad/s": 1.0, "deg/s": math.pi / 180.0, "rpm": 2.0 * math.pi / 60.0,
    # length / linear velocity
    "m": 1.0, "cm": 0.01, "mm": 0.001, "m/s": 1.0, "cm/s": 0.01, "mm/s": 0.001,
    # force / torque
    "n": 1.0, "kn": 1000.0, "nm": 1.0, "n*m": 1.0, "n.m": 1.0, "n·m": 1.0, "ncm": 0.01, "n*cm": 0.01,
    "nmm": 0.001, "n*mm": 0.001, "kgf*cm": 0.0980665, "kgcm": 0.0980665, "kg*cm": 0.0980665, "kgfcm": 0.0980665,
    # mass
    "kg": 1.0, "g": 0.001,
}

_NUMBER_WITH_UNIT = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf)\s*(.*?)\s*$", re.I)

_JOINT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "effort": ("effort", "torque", "force", "max_torque", "max_force", "max_effort"),
    "velocity": ("velocity", "max_velocity", "speed", "max_speed"),
    "stiffness": ("stiffness", "kp", "p_gain"),
    "damping": ("damping", "kd", "d_gain"),
    "armature": ("armature",),
    "friction": ("friction", "joint_friction"),
    "actuated": ("actuated", "active", "driven"),
    "drive_type": ("drive_type",),
    "target": ("target", "target_position"),
}
_JOINT_DIRECTIVES = ("name", "loop_closure", "enabled", "lower", "upper")
_LINK_DIRECTIVES = ("name", "mass", "collision", "collision_approximation", "base")

_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n"}


class ValueParser:
    """Parses attribute strings (or JSON values) into typed values."""

    def __init__(self, report: ExportReport) -> None:
        self.report = report

    def number(self, value: Any, where: str) -> Optional[float]:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if value is None or str(value).strip() == "":
            return None
        m = _NUMBER_WITH_UNIT.match(str(value))
        if not m:
            self.report.warn(f"{where}: cannot parse number {value!r}")
            return None
        number, unit = float(m.group(1)), m.group(2).strip().lower().replace(" ", "")
        if not unit:
            return number
        if unit not in _UNITS:
            self.report.warn(f"{where}: unknown unit {unit!r} in {value!r}")
            return None
        return number * _UNITS[unit]

    def boolean(self, value: Any, where: str) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
        if s in ("", "default", "none"):
            return None
        self.report.warn(f"{where}: cannot parse boolean {value!r}")
        return None

    @staticmethod
    def loose(value: Any) -> Any:
        """Best-effort typing for free-form metadata."""
        if not isinstance(value, str):
            return value
        s = value.strip()
        for conv in (int, float):
            try:
                return conv(s)
            except ValueError:
                pass
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        if s[:1] in "[{":
            try:
                return json.loads(s)
            except ValueError:
                pass
        return value


class OverrideSet:
    """Per-entity overrides from a JSON document::

        {"defaults": {"joint": {...}, "link": {...}},
         "joints": {"Revolute 1": {...}, "finger_*": {...}},
         "links": {"base:1": {...}}}

    Keys are matched against the Fusion name and the export name; glob patterns are allowed.
    Exact matches beat patterns, which beat defaults.
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        data = data or {}
        self.defaults: Dict[str, Dict[str, Any]] = data.get("defaults", {})
        self.joints: Dict[str, Dict[str, Any]] = data.get("joints", {})
        self.links: Dict[str, Dict[str, Any]] = data.get("links", {})

    @staticmethod
    def load(path: Optional[str]) -> "OverrideSet":
        if not path:
            return OverrideSet()
        with open(path, "r", encoding="utf-8") as f:
            return OverrideSet(json.load(f))

    @staticmethod
    def _resolve(table: Dict[str, Dict[str, Any]], names: Iterable[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        names = [n for n in names if n]
        globbed: Dict[str, Any] = {}
        exact: Dict[str, Any] = {}
        for key, values in table.items():
            if key in names:
                exact.update(values)
            elif any(ch in key for ch in "*?[") and any(fnmatch.fnmatchcase(n, key) for n in names):
                globbed.update(values)
        return globbed, exact

    def for_joint(self, names: Iterable[str]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        g, e = self._resolve(self.joints, names)
        return self.defaults.get("joint", {}), g, e

    def for_link(self, names: Iterable[str]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        g, e = self._resolve(self.links, names)
        return self.defaults.get("link", {}), g, e


def _layered(*layers: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for layer in layers:
        out.update({k.strip().lower(): v for k, v in layer.items()})
    return out


class PropertiesPass(ModelPass):
    """Applies metadata to ``Joint.properties`` and link settings; leftovers go to ``metadata['user']``."""

    def __init__(self, overrides: Optional[OverrideSet] = None) -> None:
        self.overrides = overrides or OverrideSet()

    def run(self, model: RobotModel, ctx: PassContext) -> RobotModel:
        parser = ValueParser(ctx.report)
        for joint in model.joints:
            self._apply_joint(joint, parser)
        for link in list(model.links.values()):
            self._apply_link(model, link, parser, ctx)
        return model

    # -- joints
    def _apply_joint(self, joint: Joint, parser: ValueParser) -> None:
        names = [joint.source, joint.name, joint.metadata.get("fusion_name", "")]
        defaults, globbed, exact = self.overrides.for_joint(names)
        raw = _layered(defaults, joint.metadata.pop("attributes", {}) or {}, globbed, exact)
        where = f"joint {joint.name!r}"

        props = JointProperties()
        consumed = set()
        for field_name, aliases in _JOINT_ALIASES.items():
            key = next((a for a in aliases if a in raw), None)
            if key is None:
                continue
            consumed.update(a for a in aliases if a in raw)
            value = raw[key]
            if field_name == "actuated":
                props.actuated = parser.boolean(value, where)
            elif field_name == "drive_type":
                props.drive_type = str(value).strip().lower() or None
            else:
                setattr(props, field_name, parser.number(value, where))
        joint.properties = joint.properties.merged(props)

        if "name" in raw and str(raw["name"]).strip():
            joint.metadata["export_name"] = str(raw["name"]).strip()
        if "loop_closure" in raw:
            joint.loop_closure_preferred = bool(parser.boolean(raw["loop_closure"], where))
        if "enabled" in raw:
            enabled = parser.boolean(raw["enabled"], where)
            joint.enabled = True if enabled is None else enabled
        if ("lower" in raw or "upper" in raw) and len(joint.dofs) == 1:
            dof = joint.dofs[0]
            lim = dof.limit or Limit()
            if "lower" in raw:
                lim.lower = parser.number(raw["lower"], where)
            if "upper" in raw:
                lim.upper = parser.number(raw["upper"], where)
            dof.limit = lim if lim.is_bounded else None
        consumed.update(_JOINT_DIRECTIVES)

        user = {k: ValueParser.loose(v) for k, v in raw.items() if k not in consumed}
        if user:
            joint.metadata.setdefault("user", {}).update(user)

    # -- links
    def _apply_link(self, model: RobotModel, link: Link, parser: ValueParser, ctx: PassContext) -> None:
        names = [link.source, link.name]
        defaults, globbed, exact = self.overrides.for_link(names)
        raw = _layered(defaults, link.metadata.pop("attributes", {}) or {}, globbed, exact)
        where = f"link {link.name!r}"

        if "name" in raw and str(raw["name"]).strip():
            link.metadata["export_name"] = str(raw["name"]).strip()
        if "mass" in raw:
            mass = parser.number(raw["mass"], where)
            if mass is not None and mass > 0:
                link.mass = link.mass.scaled_mass(mass) if link.mass else _box_mass(link, mass, ctx)
        collision = raw.get("collision", raw.get("collision_approximation"))
        if collision is not None and str(collision).strip():
            link.collision_approximation = _collision_approximation(str(collision), where, ctx) \
                or link.collision_approximation
        if "base" in raw and parser.boolean(raw["base"], where) and not ctx.options.base_link:
            model.base_link = link.name

        user = {k: ValueParser.loose(v) for k, v in raw.items() if k not in _LINK_DIRECTIVES}
        if user:
            link.metadata.setdefault("user", {}).update(user)


def _box_mass(link: Link, mass: float, ctx: PassContext) -> MassProperties:
    """A body Fusion gave no mass: treat it as a solid box over its mesh bounds."""
    bounds = [m.bounds() for m in (link.collisions or link.visuals) if m.vertex_count]
    if not bounds:
        ctx.report.warn(f"Link {link.name!r}: mass override on a body without geometry; inertia set to the minimum")
        return MassProperties(mass, Vec3(), _iso(ctx.options.min_inertia))
    lo = Vec3(*map(min, zip(*(tuple(b[0]) for b in bounds))))
    hi = Vec3(*map(max, zip(*(tuple(b[1]) for b in bounds))))
    x, y, z = hi - lo
    ctx.report.info(f"Link {link.name!r}: inertia of the mass override estimated as a solid box over its meshes")
    k = mass / 12.0
    return MassProperties(mass, (lo + hi) * 0.5, ((k * (y * y + z * z), 0.0, 0.0), (0.0, k * (x * x + z * z), 0.0),
                                                 (0.0, 0.0, k * (x * x + y * y))))


def _collision_approximation(value: str, where: str, ctx: PassContext) -> Optional[str]:
    """Accepts the USD tokens in any case and with separators (``convex_hull``, ``Convex Hull``)."""
    key = re.sub(r"[\s_-]", "", value).lower()
    for approx in CollisionApproximation:
        if approx.value.lower() == key:
            return approx.value
    ctx.report.warn(f"{where}: unknown collision approximation {value.strip()!r}; using the export default")
    return None


def _iso(value: float):
    return ((value, 0.0, 0.0), (0.0, value, 0.0), (0.0, 0.0, value))


__all__: List[str] = ["ValueParser", "OverrideSet", "PropertiesPass"]
