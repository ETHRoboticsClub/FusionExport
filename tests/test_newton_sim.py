"""Newton-target exports simulate correctly in Newton (tools/newton_smoke_test.py). Skipped without newton."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytest.importorskip("newton")

from _mechanisms import four_bar, gripper  # noqa: E402
from _usd_models import link, mimic_model  # noqa: E402
from fusion2usd.core.report import ExportReport  # noqa: E402
from fusion2usd.kinematics import PassContext, default_pipeline  # noqa: E402
from fusion2usd.options import ExportOptions, PhysicsEngine, ZeroPose  # noqa: E402
from fusion2usd.usd.builder import UsdRobotBuilder  # noqa: E402

SMOKE_TEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "newton_smoke_test.py")


def rack_pinion():
    m = mimic_model()
    m.links["loose"].mass = link("x").mass  # Kamino rejects massless bodies
    return m


@pytest.mark.parametrize("make", [four_bar, gripper, rack_pinion])
def test_newton_export_simulates(make, tmp_path):
    opts = ExportOptions(physics_engine=PhysicsEngine.NEWTON, zero_pose=ZeroPose.CURRENT)
    report = ExportReport()
    model = default_pipeline().run(make(), PassContext(opts, report))
    path = tmp_path / "robot.usda"
    path.write_text(UsdRobotBuilder(opts, report).build(model).to_string())
    run = subprocess.run([sys.executable, SMOKE_TEST, str(path), "--solver", "mujoco", "kamino", "featherstone"],
                         capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stdout + run.stderr


def _native_multi_dof(make, tmp_path):
    from fusion2usd.kinematics import JointDecompositionPass
    opts = ExportOptions(physics_engine=PhysicsEngine.NEWTON)
    model = JointDecompositionPass().run(make(), PassContext(opts, ExportReport()))
    path = tmp_path / "robot.usda"
    path.write_text(UsdRobotBuilder(opts, ExportReport()).build(model).to_string())
    import newton
    builder = newton.ModelBuilder()
    builder.add_usd(str(path))
    labels = [str(label).rsplit("/", 1)[-1] for label in builder.joint_label]
    return path, newton, dict(zip(labels, builder.joint_type)), dict(zip(labels, builder.joint_dof_dim))


def _in_tree_ball_model():
    from _usd_models import ball_model
    m = ball_model()
    for name in ("ball_joint", "cyl_joint"):
        m.joint(name).exclude_from_articulation = False
    return m


def test_native_ball_and_cylindrical_keep_all_dofs(tmp_path):
    path, newton, types, dims = _native_multi_dof(_in_tree_ball_model, tmp_path)
    assert types["ball_joint"] == newton.JointType.BALL
    assert types["cyl_joint"] == newton.JointType.D6 and tuple(dims["cyl_joint"]) == (1, 1)  # slide + free rotation
    run = subprocess.run([sys.executable, SMOKE_TEST, str(path), "--solver", "mujoco", "kamino"],
                         capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stdout + run.stderr


def test_unlimited_pin_slot_is_not_welded(tmp_path):
    from test_usd_builder import _pin_slot
    _, newton, types, dims = _native_multi_dof(_pin_slot, tmp_path)
    assert types["slot"] == newton.JointType.D6 and tuple(dims["slot"]) == (1, 1)
