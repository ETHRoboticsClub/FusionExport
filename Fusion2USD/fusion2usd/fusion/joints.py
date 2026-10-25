"""Fusion joints and as-built joints -> raw :class:`Joint` records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import adsk.fusion

from ..core.math3d import Transform, Vec3
from ..core.report import ExportReport
from ..model.robot import Dof, DofType, Joint, Limit
from .metadata import AttributeReader
from .units import CM, items, optional_transform_m, optional_vec, point_m, vec

JT = adsk.fusion.JointTypes
MT = adsk.fusion.JointMotionTypes
JD = adsk.fusion.JointDirections


@dataclass(frozen=True)
class DofSpec:
    """How one DOF of a Fusion joint motion is read and driven."""

    label: str
    type: DofType
    value_attr: str          # e.g. "rotationValue"; rad or cm, settable (Drive Joints)
    limits_attr: str         # JointLimits property
    axis_attr: Optional[str]  # direction Vector3D property (None: derived)
    direction_attr: Optional[str]  # JointDirections property used when the vector is null
    default_column: int      # geometry frame column used as the last resort
    motion_type: Any         # JointMotionTypes value for motion links


def _rot(label: str, value: str, limits: str, axis: Optional[str], direction: Optional[str],
         column: int, motion: Any) -> DofSpec:
    return DofSpec(label, DofType.ROTATION, value, limits, axis, direction, column, motion)


def _slide(label: str, value: str, limits: str, axis: Optional[str], direction: Optional[str],
           column: int, motion: Any) -> DofSpec:
    return DofSpec(label, DofType.TRANSLATION, value, limits, axis, direction, column, motion)


# DOF order per joint type is part of the model contract.
DOF_SPECS: Dict[Any, Tuple[DofSpec, ...]] = {
    JT.RigidJointType: (),
    JT.RevoluteJointType: (
        _rot("rot", "rotationValue", "rotationLimits", "rotationAxisVector", "rotationAxis", 2,
             MT.RevoluteJointRotateMotionType),
    ),
    JT.SliderJointType: (
        _slide("slide", "slideValue", "slideLimits", "slideDirectionVector", "slideDirection", 2,
               MT.SliderJointSlideMotionType),
    ),
    JT.CylindricalJointType: (
        _rot("rot", "rotationValue", "rotationLimits", "rotationAxisVector", "rotationAxis", 2,
             MT.CylindricalJointRotateMotionType),
        # Cylindrical slide shares the rotation axis.
        _slide("slide", "slideValue", "slideLimits", "rotationAxisVector", "rotationAxis", 2,
               MT.CylindricalJointSlideMotionType),
    ),
    JT.PinSlotJointType: (
        _rot("rot", "rotationValue", "rotationLimits", "rotationAxisVector", "rotationAxis", 2,
             MT.PinSlotJointRotateMotionType),
        _slide("slide", "slideValue", "slideLimits", "slideDirectionVector", "slideDirection", 0,
               MT.PinSlotJointSlideMotionType),
    ),
    JT.PlanarJointType: (
        _rot("rot", "rotationValue", "rotationLimits", "normalDirectionVector", "normalDirection", 2,
             MT.PlanarJointRotateMotionType),
        _slide("slide1", "primarySlideValue", "primarySlideLimits", "primarySlideDirectionVector",
               "primarySlideDirection", 0, MT.PlanarJointSlideOneMotionType),
        _slide("slide2", "secondarySlideValue", "secondarySlideLimits", "secondarySlideDirectionVector",
               None, 1, MT.PlanarJointSlideTwoMotionType),
    ),
    JT.BallJointType: (
        _rot("pitch", "pitchValue", "pitchLimits", "pitchDirectionVector", "pitchDirection", 2,
             MT.BallJointPitchMotionType),
        _rot("yaw", "yawValue", "yawLimits", "yawDirectionVector", "yawDirection", 0,
             MT.BallJointYawMotionType),
        _rot("roll", "rollValue", "rollLimits", "rollDirectionVector", None, 1,
             MT.BallJointRollMotionType),
    ),
}

TYPE_NAMES = {
    JT.RigidJointType: "rigid", JT.RevoluteJointType: "revolute", JT.SliderJointType: "slider",
    JT.CylindricalJointType: "cylindrical", JT.PinSlotJointType: "pin_slot",
    JT.PlanarJointType: "planar", JT.BallJointType: "ball",
}
if hasattr(JT, "InferredJointType"):  # missing in some Fusion builds
    TYPE_NAMES[JT.InferredJointType] = "inferred"

# JointMotionTypes value -> DOF index within its joint (for motion links).
MOTION_DOF_INDEX: Dict[Any, int] = {
    spec.motion_type: i for specs in DOF_SPECS.values() for i, spec in enumerate(specs)
}

_DIRECTION_COLUMN = {JD.XAxisJointDirection: 0, JD.YAxisJointDirection: 1, JD.ZAxisJointDirection: 2}


def to_si(spec: DofSpec, value: float) -> float:
    return value * CM if spec.type is DofType.TRANSLATION else value


def from_si(spec: DofSpec, value: float) -> float:
    return value / CM if spec.type is DofType.TRANSLATION else value


def occurrence_key(occurrence: Any) -> Optional[str]:
    return None if occurrence is None else occurrence.fullPathName


def context_path(entity: Any) -> str:
    ctx = getattr(entity, "assemblyContext", None)
    return ctx.fullPathName if ctx is not None else ""


@dataclass
class FusionJoint:
    """A non-suppressed Fusion joint with everything the reader needs."""

    entity: Any
    name: str              # unique, human readable
    fusion_name: str
    context: str           # assembly context path ("" in the root component)
    as_built: bool
    joint_type: Any        # JointTypes value
    occurrence_one: Any    # moving side (child); None -> root
    occurrence_two: Any    # parent side
    frame: Optional[Transform]  # world joint frame (geometry two)

    @property
    def motion(self) -> Any:
        return self.entity.jointMotion

    @property
    def specs(self) -> Tuple[DofSpec, ...]:
        return DOF_SPECS.get(self.joint_type, ())

    @property
    def is_rigid(self) -> bool:
        return self.joint_type == JT.RigidJointType

    @property
    def key_one(self) -> Optional[str]:
        return occurrence_key(self.occurrence_one)

    @property
    def key_two(self) -> Optional[str]:
        return occurrence_key(self.occurrence_two)

    @property
    def identity(self) -> Tuple[str, str]:
        return (self.context, self.fusion_name)


class JointCollector:
    """Collects the joints and as-built joints of the whole design (root-context proxies)."""

    def __init__(self, root_component: Any, report: ExportReport) -> None:
        self._root = root_component
        self._report = report

    def collect(self) -> List[FusionJoint]:
        out: List[FusionJoint] = []
        for entity, as_built in self._entities():
            if entity.isSuppressed:
                continue
            motion = entity.jointMotion
            if motion is None:
                self._report.warn(f"joint {entity.name!r} has no motion definition; skipped")
                continue
            joint_type = motion.jointType
            if joint_type not in DOF_SPECS:
                self._report.warn(f"joint {entity.name!r} has unsupported type "
                                  f"{TYPE_NAMES.get(joint_type, joint_type)}; skipped")
                continue
            out.append(FusionJoint(
                entity=entity, name=entity.name, fusion_name=entity.name, context=context_path(entity),
                as_built=as_built, joint_type=joint_type, occurrence_one=entity.occurrenceOne,
                occurrence_two=entity.occurrenceTwo, frame=self._frame(entity, as_built),
            ))
        self._make_names_unique(out)
        return out

    def _entities(self):
        for j in items(self._root.allJoints):
            yield j, False
        for j in items(self._root.allAsBuiltJoints):
            yield j, True

    def _frame(self, entity: Any, as_built: bool) -> Optional[Transform]:
        try:
            if not as_built:
                return optional_transform_m(entity.geometryTwoTransform)
            frame = optional_transform_m(entity.transform)
            if frame is None and entity.geometry is not None:
                g = entity.geometry
                frame = Transform.from_axes(point_m(g.origin), vec(g.secondaryAxisVector), vec(g.thirdAxisVector))
            return frame
        except Exception as exc:  # broken references make these getters throw
            self._report.warn(f"joint {entity.name!r}: cannot read joint frame ({exc})")
            return None

    @staticmethod
    def _make_names_unique(joints: List[FusionJoint]) -> None:
        counts: Dict[str, int] = {}
        for j in joints:
            counts[j.fusion_name] = counts.get(j.fusion_name, 0) + 1
        used = set()
        for j in joints:
            name = j.fusion_name
            if counts[name] > 1 and j.context:
                ctx = j.entity.assemblyContext
                name = f"{ctx.name}/{j.fusion_name}"
                if name in used:
                    name = f"{j.context}/{j.fusion_name}"
            base, n = name, 1
            while name in used:
                n += 1
                name = f"{base}_{n}"
            used.add(name)
            j.name = name


class JointBuilder:
    """Turns a :class:`FusionJoint` into a raw model :class:`Joint` (world frame, Fusion orientation)."""

    def __init__(self, report: ExportReport, attributes: Optional[AttributeReader] = None) -> None:
        self._report = report
        self._attributes = attributes or AttributeReader()

    def build(self, fj: FusionJoint, parent: str, child: str) -> Joint:
        frame = fj.frame or self._fallback_frame(fj)
        motion = fj.motion
        dofs = [self._dof(fj, spec, motion, frame) for spec in fj.specs]
        try:
            attrs = self._attributes.read(fj.entity)
        except ValueError as exc:
            self._report.warn(f"joint {fj.name!r}: invalid fusion2usd attributes ({exc})")
            attrs = self._attributes.read_raw(fj.entity)
        type_name = TYPE_NAMES.get(fj.joint_type, str(fj.joint_type))
        return Joint(
            name=fj.name, parent=parent, child=child, origin=frame.translation, dofs=dofs,
            metadata={"fusion_type": type_name, "fusion_name": fj.fusion_name,
                      "as_built": fj.as_built, "attributes": attrs},
            source=fj.name, source_type=type_name, frame_hint=frame,
        )

    def _fallback_frame(self, fj: FusionJoint) -> Transform:
        self._report.warn(f"joint {fj.name!r}: no joint frame available; using the child occurrence origin")
        occ = fj.occurrence_one or fj.occurrence_two
        return optional_transform_m(occ.transform2) if occ is not None else Transform()

    def _dof(self, fj: FusionJoint, spec: DofSpec, motion: Any, frame: Transform) -> Dof:
        axis = optional_vec(getattr(motion, spec.axis_attr)) if spec.axis_attr else None
        if axis is None:
            axis = self._fallback_axis(spec, motion, frame)
        return Dof(spec.type, axis.normalized(), to_si(spec, getattr(motion, spec.value_attr)),
                   self._limit(spec, getattr(motion, spec.limits_attr)), spec.label)

    @staticmethod
    def _fallback_axis(spec: DofSpec, motion: Any, frame: Transform) -> Vec3:
        column = spec.default_column
        if spec.direction_attr:
            column = _DIRECTION_COLUMN.get(getattr(motion, spec.direction_attr), column)
        return frame.axis(column)

    @staticmethod
    def _limit(spec: DofSpec, limits: Any) -> Optional[Limit]:
        if limits is None:
            return None
        lower = to_si(spec, limits.minimumValue) if limits.isMinimumValueEnabled else None
        upper = to_si(spec, limits.maximumValue) if limits.isMaximumValueEnabled else None
        limit = Limit(lower, upper)
        return limit if limit.is_bounded else None
