"""Regression tests for issues found in code review."""

import math
import re

import pytest

from _mechanisms import four_bar, gripper
from fusion2usd.core.math3d import Quat, Transform, Vec3
from fusion2usd.core.report import ExportReport
from fusion2usd.isaaclab.cfg_writer import IsaacLabCfgWriter
from fusion2usd.kinematics import PassContext, default_pipeline
from fusion2usd.options import ExportOptions, ZeroPose
from fusion2usd.usd.builder import UsdRobotBuilder


def process(model, **opts):
    options = ExportOptions(**opts)
    report = ExportReport()
    return default_pipeline().run(model, PassContext(options, report)), options, report


def test_asset_origin_is_upright_base_and_cfg_keeps_placement(tmp_path):
    """Isaac Lab spawns the asset at init_state and resets the base link to it: base must sit at the origin."""
    m = four_bar()
    m.metadata["source_up_axis"] = "Y"           # base frame becomes Rx(90°) after the up-axis pass
    m.links["ground"].transform = Transform(translation=Vec3(0.0, 0.3, 0.0))  # 0.3 m "up" in Y-up
    model, opts, report = process(m)
    base = model.links[model.base_link]
    assert base.transform.is_close(Transform())
    assert model.root_pose.translation.is_close(Vec3(0, 0, 0.3))
    assert model.root_pose.rotation.angle_to(Quat()) < 1e-9
    # geometry was re-expressed: the base box's local z extent now spans the former y extent
    lo, hi = base.visuals[0].bounds()
    assert (hi.z - lo.z) == pytest.approx(0.1) and (hi.y - lo.y) == pytest.approx(0.1)
    src = IsaacLabCfgWriter(opts, report).render(model, str(tmp_path / "r.usda"), str(tmp_path / "r_cfg.py"))
    pos = [float(v) for v in re.search(r"pos=\(([^)]*)\)", src).group(1).split(",")]
    assert pos == pytest.approx([0, 0, 0.3])
    assert "rot=" not in src  # identity by construction; avoids the 2.x/3.x quaternion-order change


def test_reversed_pin_slot_keeps_slot_in_fusion_parent_and_effort_on_rotation():
    m = gripper()
    # Fusion parent of the pin-slot is the moving "slider"; the tree from "palm" reverses it.
    pin_slot = m.joint("Cylindrical 1")
    pin_slot.parent, pin_slot.child = "slider", "palm"
    pin_slot.dofs[1].axis = Vec3(1, 0, 0)  # slot direction, perpendicular to the pin
    pin_slot.metadata["attributes"] = {"effort": "3 Nm"}
    model, _, report = process(m)
    rot, slide = model.joint("Cylindrical_1_rot"), model.joint("Cylindrical_1_slide")
    assert rot.properties.effort == 3.0 and slide.properties.effort is None
    # reversed chain: palm -> rot -> helper -> slide -> slider (the slot stays fixed in the slider)
    assert rot.parent == "palm" and slide.child == "slider" and rot.child == slide.parent


@pytest.mark.parametrize("zero, expected", [(ZeroPose.FUSION, -0.3), (ZeroPose.CURRENT, -0.5)])
def test_reversed_joint_negates_drive_target(zero, expected):
    m = gripper()
    m.joint("Revolute 2").metadata["attributes"] = {"target": "0.3"}  # CAD convention, value -0.2
    model, _, _ = process(m, zero_pose=zero)
    j = model.joint("Revolute_2")
    assert j.metadata["reversed"]
    assert j.properties.target == pytest.approx(expected)  # current mode: -0.3 shifted by -(+0.2)


def test_split_joint_properties_are_per_dof():
    m = gripper()
    m.joint("Cylindrical 1").metadata["attributes"] = {"effort": "2 Nm", "velocity": "180 deg/s", "target": 0.4}
    model, _, report = process(m, zero_pose=ZeroPose.CURRENT)
    rot, slide = model.joint("Cylindrical_1_rot"), model.joint("Cylindrical_1_slide")
    assert rot.properties is not slide.properties
    assert rot.properties.effort == 2.0 and rot.properties.velocity == pytest.approx(math.pi)
    assert slide.properties.effort is None and slide.properties.velocity is None
    assert rot.properties.target is None and slide.properties.target is None
    assert any("ambiguous" in w for w in report.warnings)


def test_decomposed_chain_bodies_are_filtered():
    model, opts, report = process(gripper())
    stage = UsdRobotBuilder(opts, report).build(model)
    slider = stage.get("/gripper/slider")
    targets = [str(t) for t in slider.get("physics:filteredPairs").targets]
    assert "/gripper/palm" in targets  # parent of the split cylindrical joint, two hops away


def test_mimic_offset_in_degrees_for_revolute_follower():
    m = gripper()
    m.couplings[0].offset = 0.1  # rad
    model, opts, report = process(m)
    stage = UsdRobotBuilder(opts, report).build(model)
    prim = stage.get("/gripper/joints/Revolute_2")
    # reversed follower: offset -> -0.1 rad; PhysX offset = -(that) in degrees
    assert prim.get("physxMimicJoint:rotX:offset").value == pytest.approx(math.degrees(0.1))
