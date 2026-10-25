"""Measures true joint axes and signs by nudging each joint DOF in Fusion and restoring it."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, List, Tuple

from ..core.math3d import Transform
from ..core.report import ExportReport
from ..model.robot import DofType, Joint
from .joints import JT, DofSpec, FusionJoint, to_si
from .probe_math import axis_deviation, measure_dof, project_onto_line, relative_motion
from .units import transform_m

_ROT_DELTA = 0.02    # rad
_SLIDE_DELTA = 0.05  # cm (Fusion internal units)
_AXIS_TYPES = (JT.RevoluteJointType, JT.CylindricalJointType, JT.PinSlotJointType)


def _world(occurrence: Any) -> Transform:
    return Transform() if occurrence is None else transform_m(occurrence.transform2)


class JointProbe:
    def __init__(self, report: ExportReport, max_deviation_deg: float = 5.0,
                 max_origin_shift: float = 1e-3) -> None:
        self._report = report
        self._max_dev = math.radians(max_deviation_deg)
        self._max_shift = max_origin_shift

    def run(self, entries: List[Tuple[FusionJoint, Joint]]) -> None:
        """Update ``Joint.dofs[*].axis`` (and revolute origins) in place from measured motion."""
        # Motion links move other joints, so every value is snapshotted and restored.
        snapshot = [(fj.entity, spec.value_attr, getattr(fj.motion, spec.value_attr))
                    for fj, _ in entries for spec in fj.specs]
        try:
            for fj, joint in entries:
                for i, spec in enumerate(fj.specs):
                    try:
                        self._probe(fj, joint, i, spec)
                    except Exception as exc:
                        self._report.warn(f"joint {joint.name!r} DOF {spec.label!r}: probing failed ({exc}); "
                                          "using API axis")
        finally:
            self._restore(snapshot)

    def _restore(self, snapshot: List[Tuple[Any, str, float]]) -> None:
        for entity, attr, value in snapshot:
            try:
                setattr(entity.jointMotion, attr, value)
            except Exception as exc:
                self._report.warn(f"joint {entity.name!r}: could not restore {attr} ({exc}); "
                                  "check the design pose")

    @staticmethod
    def _delta(spec: DofSpec, motion: Any, value: float) -> float:
        delta = _ROT_DELTA if spec.type is DofType.ROTATION else _SLIDE_DELTA
        limits = getattr(motion, spec.limits_attr)
        if limits is None:
            return delta
        hi = limits.maximumValue if limits.isMaximumValueEnabled else math.inf
        lo = limits.minimumValue if limits.isMinimumValueEnabled else -math.inf
        if value + delta <= hi:
            return delta
        if value - delta >= lo:
            return -delta
        room_up, room_down = hi - value, value - lo
        return room_up / 2.0 if room_up >= room_down else -room_down / 2.0

    def _probe(self, fj: FusionJoint, joint: Joint, index: int, spec: DofSpec) -> None:
        motion = fj.motion
        v0 = getattr(motion, spec.value_attr)
        delta = self._delta(spec, motion, v0)
        if delta == 0.0:
            raise ValueError("no room within the joint limits")
        a0, b0 = _world(fj.occurrence_one), _world(fj.occurrence_two)
        setattr(motion, spec.value_attr, v0 + delta)
        try:
            a1, b1 = _world(fj.occurrence_one), _world(fj.occurrence_two)
        finally:
            setattr(motion, spec.value_attr, v0)
        d = relative_motion(a0, b0, a1, b1)
        delta_si = to_si(spec, delta)
        m = measure_dof(spec.type, delta_si, d, near=joint.origin)
        # Wrong magnitude: other DOFs moved too (e.g. Fusion re-solving a closed loop), so the axis is unreliable.
        if abs(m.magnitude - abs(delta_si)) > 0.2 * abs(delta_si):
            self._report.warn(f"joint {joint.name!r} DOF {spec.label!r}: moved {m.magnitude:.4g} for a "
                              f"commanded {abs(delta_si):.4g}; using API axis")
            return
        dof = joint.dofs[index]
        if axis_deviation(m.axis, dof.axis) > self._max_dev:
            self._report.warn(f"joint {joint.name!r} DOF {spec.label!r}: measured axis deviates "
                              f"{math.degrees(axis_deviation(m.axis, dof.axis)):.1f}° from the API axis")
        joint.dofs[index] = replace(dof, axis=m.axis)
        joint.metadata.setdefault("probed", []).append(spec.label)
        if index == 0 and m.point is not None and fj.joint_type in _AXIS_TYPES:
            origin = project_onto_line(joint.origin, m.point, m.axis)
            if (origin - joint.origin).norm() > self._max_shift:
                self._report.warn(f"joint {joint.name!r}: joint origin is {(origin - joint.origin).norm():.4g} m "
                                  "off the measured rotation axis; moved onto it")
            joint.origin = origin
