"""Recover joint axes from nudged occurrence poses (pure logic, no ``adsk``)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..core.math3d import Transform, Vec3, angle_between
from ..model.robot import DofType


def relative_motion(a0: Transform, b0: Transform, a1: Transform, b1: Transform) -> Transform:
    """World-frame motion of the child (A) relative to the parent (B), holding B at its initial pose.

    ``D = B0 · (B1⁻¹ A1) · (B0⁻¹ A0)⁻¹ · B0⁻¹``, i.e. ``D · A0 == B0 · B1⁻¹ · A1``.
    """
    return b0 * (b1.inverse() * a1) * (b0.inverse() * a0).inverse() * b0.inverse()


@dataclass
class MeasuredDof:
    axis: Vec3                    # unit; +delta of the CAD value moves the child along/about it
    magnitude: float              # measured |rotation angle| (rad) or |translation| (m)
    point: Optional[Vec3] = None  # rotation only: point on the axis closest to the reference
    residual: float = 0.0         # translation (rotation DOF: along the axis) or rotation (translation DOF)


def rotation_axis_point(d: Transform, axis: Vec3, angle: float, near: Vec3) -> Vec3:
    """Point on the rotation axis of ``d`` closest to ``near``.

    Solves ``(I − R) p = t`` in the plane perpendicular to ``axis`` (where ``I − R`` is
    invertible), then slides along the axis towards ``near``.
    """
    t = d.translation
    t_perp = t - axis * axis.dot(t)
    c, s = math.cos(angle), math.sin(angle)
    denom = 2.0 * (1.0 - c)
    if denom < 1e-18:
        raise ValueError("rotation too small to locate its axis")
    p0 = (t_perp * (1.0 - c) + axis.cross(t_perp) * s) / denom
    return p0 + axis * (near - p0).dot(axis)


def measure_dof(dof_type: DofType, delta: float, d: Transform, near: Vec3,
                min_magnitude: Optional[float] = None) -> MeasuredDof:
    """Axis of motion ``d`` that was produced by changing the CAD value by ``delta`` (signed)."""
    if delta == 0.0:
        raise ValueError("delta must be non-zero")
    sign = 1.0 if delta > 0 else -1.0
    limit = abs(delta) * 0.1 if min_magnitude is None else min_magnitude
    if dof_type is DofType.ROTATION:
        u, angle = d.rotation.to_axis_angle()  # angle in [0, pi]
        if angle < limit:
            raise ValueError(f"joint did not rotate (measured {angle:.3g} rad)")
        u = u.normalized()
        point = rotation_axis_point(d, u, angle, near)
        # Rotating by -|delta| about u equals rotating by +|delta| about -u.
        return MeasuredDof(u * sign, angle, point, residual=abs(d.translation.dot(u)))
    dist = d.translation.norm()
    if dist < limit:
        raise ValueError(f"joint did not translate (measured {dist:.3g} m)")
    _, angle = d.rotation.to_axis_angle()
    return MeasuredDof(d.translation.normalized() * sign, dist, None, residual=angle)


def axis_deviation(measured: Vec3, api: Vec3) -> float:
    """Angle (rad) between the measured axis and the API axis, ignoring sign."""
    a = angle_between(measured, api)
    return min(a, math.pi - a)


def project_onto_line(p: Vec3, point: Vec3, axis: Vec3) -> Vec3:
    u = axis.normalized()
    return point + u * (p - point).dot(u)
