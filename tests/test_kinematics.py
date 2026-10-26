import math

import pytest

from _mechanisms import four_bar, gripper, link
from fusion2usd.core.math3d import Vec3
from fusion2usd.core.report import ExportReport
from fusion2usd.kinematics import (
    JointDecompositionPass, KinematicTreePass, NamingPass, OverrideSet, PassContext, PropertiesPass,
    UpAxisPass, ValidationPass, ZeroPosePass, default_pipeline,
)
from fusion2usd.model.robot import DofRef, JointKind, Joint, Limit
from fusion2usd.options import DisconnectedPolicy, ExportOptions, ZeroPose


def ctx(**kw):
    return PassContext(ExportOptions(**kw), ExportReport())


def run(model, *passes, c=None):
    c = c or ctx()
    for p in passes:
        model = p.run(model, c)
    return model, c


def test_four_bar_cuts_exactly_one_loop_and_orients_tree():
    m, c = run(four_bar(), PropertiesPass(), NamingPass(), KinematicTreePass(), c=ctx(fixed_base=None))
    excluded = [j for j in m.joints if j.exclude_from_articulation]
    assert len(excluded) == 1
    assert m.base_link == "ground" and m.fixed_base  # auto: grounded in Fusion
    floating, _ = run(four_bar(), PropertiesPass(), NamingPass(), KinematicTreePass())
    assert not floating.fixed_base  # default
    # Every articulation joint's parent must be reachable before its child (tree order).
    seen = {"ground"}
    for j in sorted((j for j in m.joints if not j.exclude_from_articulation), key=lambda j: list(m.links).index(j.child)):
        assert j.parent in seen
        seen.add(j.child)
    assert seen == set(m.links)


def test_reversal_negates_values_and_limits():
    m, _ = run(four_bar(), PropertiesPass(), NamingPass(), KinematicTreePass())
    j2 = m.joint("Revolute_2")
    if j2.parent == "crank":  # it was reversed into the tree
        assert j2.dofs[0].value == pytest.approx(-0.2)
        assert j2.metadata["reversed"]


def test_user_loop_closure_hint_is_respected():
    m = four_bar()
    m.joint("Revolute 3").metadata["attributes"] = {"loop_closure": "true"}
    m, _ = run(m, PropertiesPass(), NamingPass(), KinematicTreePass())
    assert [j.name for j in m.joints if j.exclude_from_articulation] == ["Revolute_3"]


def test_coupling_follows_reversal_and_decomposition():
    m, c = run(gripper(), PropertiesPass(), NamingPass(), KinematicTreePass(), JointDecompositionPass())
    assert not m.fixed_base and m.base_link == "palm"
    mirror = next(cp for cp in m.couplings if cp.name == "Motion Link 1")
    assert m.joint("Revolute_2").parent == "palm"
    assert mirror.ratio == pytest.approx(1.0)  # q_r' = -q_r = q_l
    screw = next(cp for cp in m.couplings if cp.name == "Motion Link 2")
    assert screw.leader == DofRef("Cylindrical_1_rot", 0)
    assert screw.follower == DofRef("Cylindrical_1_slide", 0)
    # translation first (slot fixed in the parent), rotation on top
    helper = m.links["Cylindrical_1_slide_link"]
    assert helper.is_virtual
    assert m.joint("Cylindrical_1_slide").parent == "palm"
    assert m.joint("Cylindrical_1_slide").child == helper.name
    assert m.joint("Cylindrical_1_rot").parent == helper.name
    assert all(j.kind is not JointKind.MULTI for j in m.joints)


def test_excluded_multi_dof_joint_is_not_decomposed():
    m = four_bar()
    j = m.joint("Revolute 3")
    j.dofs.append(type(j.dofs[0])(j.dofs[0].type, Vec3(1, 0, 0), 0.0, None, "pitch"))
    m, _ = run(m, PropertiesPass(), NamingPass(), KinematicTreePass(), JointDecompositionPass())
    assert m.joint("Revolute_3").exclude_from_articulation  # multi-DOF joints are cheapest to cut
    assert m.joint("Revolute_3").kind is JointKind.MULTI
    assert not any(l.is_virtual for l in m.links.values())


def test_zero_pose_current_shifts_limits_and_offsets():
    m, c = run(gripper(), PropertiesPass(), NamingPass(), KinematicTreePass(), JointDecompositionPass())
    before = {cp.name: (cp.ratio, cp.offset) for cp in m.couplings}
    m, _ = run(m, ZeroPosePass(), c=PassContext(ExportOptions(zero_pose=ZeroPose.CURRENT), ExportReport()))
    finger = m.joint("Revolute_1")
    assert finger.dofs[0].value == 0
    assert finger.dofs[0].limit.lower == pytest.approx(-0.2) and finger.dofs[0].limit.upper == pytest.approx(0.8)
    # Screw: offset was chosen so the coupling holds at the export pose -> becomes 0 at the new zero.
    screw = next(cp for cp in m.couplings if cp.name == "Motion Link 2")
    assert screw.offset == pytest.approx(0.0)
    assert screw.ratio == before["Motion Link 2"][0]


def test_properties_units_aliases_and_overrides(tmp_path):
    m = four_bar()
    m.joint("Revolute 1").metadata["attributes"] = {
        "torque": "15 kgf*cm", "max_velocity": "180 deg/s", "kp": "20", "color": "red", "name": "crank_joint",
    }
    overrides = OverrideSet({
        "defaults": {"joint": {"damping": 0.5}},
        "joints": {"Revolute *": {"armature": 0.01}, "Revolute 4": {"actuated": False}},
        "links": {"ground": {"mass": "7 kg", "collision": "convexHull"}},
    })
    m, c = run(m, PropertiesPass(overrides), NamingPass())
    p = m.joint("crank_joint").properties
    assert p.effort == pytest.approx(1.470998, rel=1e-5)
    assert p.velocity == pytest.approx(math.pi)
    assert p.stiffness == 20 and p.damping == 0.5 and p.armature == 0.01
    assert m.joint("crank_joint").metadata["user"] == {"color": "red"}
    assert m.joint("Revolute_4").properties.actuated is False
    ground = m.links["ground"]
    assert ground.mass.mass == pytest.approx(7.0) and ground.collision_approximation == "convexHull"
    assert not c.report.warnings


def test_bad_values_warn():
    m = four_bar()
    m.joint("Revolute 1").metadata["attributes"] = {"effort": "lots", "velocity": "3 furlongs"}
    _, c = run(m, PropertiesPass())
    assert len(c.report.warnings) == 2


def test_continuous_and_default_value_warnings():
    def warnings(env):
        m, c = run(four_bar(), PropertiesPass(), NamingPass(), KinematicTreePass())
        ValidationPass(env).run(m, c)
        return c.report.warnings

    w = warnings({})
    continuous = [x for x in w if "continuous" in x]
    defaults = [x for x in w if "default values" in x]
    assert len(continuous) == 3  # Revolute 2..4 have no limits
    assert len(defaults) == 3    # driven tree joints (the loop-closing one is passive)
    assert len([x for x in warnings({"ALLOW_CONTINUOUS": "Revolute 2, Revolute_3"}) if "continuous" in x]) == 1
    assert not [x for x in warnings({"DISABLE_CONTINUOUS_WARN": "1"}) if "continuous" in x]


def test_disconnected_policies():
    for policy, expect_links, expect_weld in (
        (DisconnectedPolicy.FIX_TO_BASE, 5, True), (DisconnectedPolicy.DROP, 4, False), (DisconnectedPolicy.FREE, 5, False),
    ):
        m = four_bar()
        m.add_link(link("loose", 5, 5))
        m, c = run(m, PropertiesPass(), NamingPass(), KinematicTreePass(), c=ctx(disconnected=policy))
        assert len(m.links) == expect_links
        assert any(j.source_type == "weld" for j in m.joints) == expect_weld
        assert c.report.warnings


def test_up_axis_conversion():
    m = four_bar()
    m.metadata["source_up_axis"] = "Y"
    m, _ = run(m, UpAxisPass())
    assert m.joint("Revolute 1").dofs[0].axis.is_close(Vec3(0, -1, 0))  # old +Z -> new -Y
    assert m.links["crank"].transform.translation.is_close(Vec3(0, 0, 0.5))  # old +Y -> new +Z


def test_naming_avoids_collisions():
    m = four_bar()
    m.rename_link("coupler", "rocker 1")
    m.links["rocker"].metadata["export_name"] = "rocker_1"
    m, _ = run(m, NamingPass())
    assert len(set(m.links)) == 4
    assert all(j.parent in m.links and j.child in m.links for j in m.joints)


def test_full_pipeline_runs():
    m, c = run(gripper(), *default_pipeline().passes)
    assert all(l.mass is not None for l in m.links.values())
    assert m.links and m.joints
