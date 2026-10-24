import math

import pytest

from fusion2usd.core.math3d import Quat, Transform, Vec3
from fusion2usd.fusion.probe_math import (
    axis_deviation,
    measure_dof,
    project_onto_line,
    relative_motion,
)
from fusion2usd.model.robot import DofType


def rotation_about(axis: Vec3, point: Vec3, angle: float) -> Transform:
    """World rotation about the line ``point + s*axis``."""
    r = Transform(Quat.from_axis_angle(axis, angle))
    return Transform(translation=point) * r * Transform(translation=-point)


PARENT_POSE = Transform(Quat.from_axis_angle(Vec3(1, 2, 3), 0.7), Vec3(0.3, -0.2, 0.5))
CHILD_POSE = Transform(Quat.from_axis_angle(Vec3(0, 1, 1), -0.4), Vec3(0.1, 0.4, 0.2))


def nudged(world_motion: Transform, parent_moves: Transform):
    """Poses after the joint moved the child by ``world_motion`` and the whole assembly by ``parent_moves``."""
    a0, b0 = CHILD_POSE, PARENT_POSE
    a1 = parent_moves * world_motion * a0
    b1 = parent_moves * b0
    return a0, b0, a1, b1


@pytest.mark.parametrize("delta", [0.02, -0.02])
def test_revolute_off_origin_axis_with_moving_parent(delta):
    axis = Vec3(0.2, -0.5, 1.0).normalized()
    point = Vec3(0.4, 0.1, -0.3)
    motion = rotation_about(axis, point, delta)
    # Something else (e.g. a motion link) moved the parent too; the measurement must ignore that.
    drift = Transform(Quat.from_axis_angle(Vec3(1, 0, 0), 0.3), Vec3(0.05, 0.0, 0.01))
    a0, b0, a1, b1 = nudged(motion, drift)
    d = relative_motion(a0, b0, a1, b1)
    near = point + axis * 0.25 + Vec3(0.001, 0, 0)
    m = measure_dof(DofType.ROTATION, delta, d, near=near)
    assert m.axis.is_close(axis, 1e-9)  # positive Fusion value change -> +axis
    assert m.magnitude == pytest.approx(abs(delta), rel=1e-9)
    # Point on axis closest to `near`.
    expected = project_onto_line(near, point, axis)
    assert m.point.is_close(expected, 1e-7)
    assert m.residual < 1e-9


def test_revolute_sign_flip_detected():
    axis = Vec3(0, 0, 1)
    motion = rotation_about(axis, Vec3(), -0.02)  # Fusion value +0.02 rotates about -Z
    a0, b0, a1, b1 = nudged(motion, Transform())
    m = measure_dof(DofType.ROTATION, 0.02, relative_motion(a0, b0, a1, b1), near=Vec3())
    assert m.axis.is_close(Vec3(0, 0, -1), 1e-9)
    assert axis_deviation(m.axis, axis) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("delta", [0.0005, -0.0005])
def test_prismatic(delta):
    direction = Vec3(1.0, 1.0, 0.0).normalized()
    motion = Transform(translation=direction * delta)
    drift = Transform(Quat.from_axis_angle(Vec3(0, 0, 1), 1.0), Vec3(1, 2, 3))
    a0, b0, a1, b1 = nudged(motion, drift)
    m = measure_dof(DofType.TRANSLATION, delta, relative_motion(a0, b0, a1, b1), near=Vec3())
    assert m.axis.is_close(direction, 1e-9)
    assert m.magnitude == pytest.approx(abs(delta), rel=1e-6)
    assert m.point is None and m.residual < 1e-9


def test_no_motion_raises():
    a0, b0, a1, b1 = nudged(Transform(), Transform())
    with pytest.raises(ValueError):
        measure_dof(DofType.ROTATION, 0.02, relative_motion(a0, b0, a1, b1), near=Vec3())
    with pytest.raises(ValueError):
        measure_dof(DofType.TRANSLATION, 0.0005, relative_motion(a0, b0, a1, b1), near=Vec3())


def test_axis_deviation_ignores_sign():
    assert axis_deviation(Vec3(1, 0, 0), Vec3(-1, 0, 0)) == pytest.approx(0.0)
    assert axis_deviation(Vec3(1, 0, 0), Vec3(0, 1, 0)) == pytest.approx(math.pi / 2)
