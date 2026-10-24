"""Hand-built, already kinematically processed robot models for the USD/Isaac Lab writer tests."""

from __future__ import annotations

import math

from fusion2usd.core.inertia import MassProperties
from fusion2usd.core.math3d import Quat, Transform, Vec3
from fusion2usd.model.robot import (
    Coupling, Dof, DofRef, DofType, Joint, JointProperties, Limit, Link, Material, Mesh, RobotModel,
)

ROT, TRANS = DofType.ROTATION, DofType.TRANSLATION


def box_mesh(name: str, size: float = 0.05, material: str | None = None, normals: bool = True) -> Mesh:
    s = size / 2
    pts = []
    for x in (-s, s):
        for y in (-s, s):
            for z in (-s, s):
                pts += [x, y, z]
    tris = [0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1,
            2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3]
    nrm = None
    if normals:
        nrm = []
        for i in range(0, len(pts), 3):
            v = Vec3(*pts[i:i + 3]).normalized()
            nrm += list(v)
    return Mesh(name, pts, tris, nrm, material)


def link(name: str, pos=(0.0, 0.0, 0.0), rot: Quat | None = None, mass: float | None = 1.0, **kw) -> Link:
    t = Transform(rot or Quat(), Vec3(*pos))
    mp = None
    if mass is not None:
        mp = MassProperties.from_principal(mass, Vec3(0.01, 0.0, 0.02), (0.01, 0.02, 0.03),
                                           (Vec3(1, 0, 0), Vec3(0, 0, 1), Vec3(0, -1, 0)))
    return Link(name, t, mp, **kw)


def arm_model() -> RobotModel:
    """Fixed-base 2-link revolute arm with limits, effort and a material."""
    m = RobotModel("Test Arm", fixed_base=True, metadata={"source_document": "arm.f3d", "version": 3})
    m.add_material(Material("steel", (0.6, 0.6, 0.65), roughness=0.3, metallic=1.0))
    m.add_link(link("base", (0.1, 0.2, 0.0), Quat.from_axis_angle(Vec3(0, 0, 1), 0.4),
                    visuals=[box_mesh("body", material="steel")], metadata={"occurrence": "Base:1"}))
    m.add_link(link("link1", (0.1, 0.2, 0.1), Quat.from_axis_angle(Vec3(1, 1, 0), 0.7),
                    visuals=[box_mesh("v")], collisions=[box_mesh("c", normals=False)]))
    m.add_link(link("link2", (0.3, 0.2, 0.3), Quat.from_axis_angle(Vec3(0, 1, 1), -1.1),
                    visuals=[box_mesh("v", material="steel")], collision_approximation="convexHull"))
    m.base_link = "base"
    m.add_joint(Joint(
        "shoulder", "base", "link1", Vec3(0.1, 0.2, 0.1),
        [Dof(ROT, Vec3(0, 0, 1), 0.3, Limit(-1.0, 2.0))],
        JointProperties(effort=10.0, velocity=3.0, stiffness=100.0, damping=5.0, armature=0.01),
        source="Rev1", source_type="RevoluteJointType",
    ))
    m.add_joint(Joint(
        "elbow", "link1", "link2", Vec3(0.3, 0.2, 0.3),
        [Dof(ROT, Vec3(0, 1, 0), -0.5, Limit(None, 1.5))],
        frame_hint=Transform(Quat.from_axis_angle(Vec3(0, 1, 0), 0.2)),
        metadata={"nested": {"a": 1, "b": [1.5, 2]}},
    ))
    return m


def four_bar_model() -> RobotModel:
    """Floating-base planar four-bar; the rocker->ground joint closes the loop."""
    m = RobotModel("four_bar")
    m.add_link(link("ground", (0, 0, 0)))
    m.add_link(link("crank", (0, 0, 0.05), Quat.from_axis_angle(Vec3(0, 0, 1), 0.5)))
    m.add_link(link("coupler", (0.1, 0.05, 0.05), Quat.from_axis_angle(Vec3(0, 0, 1), -0.2)))
    m.add_link(link("rocker", (0.3, 0, 0.05), Quat.from_axis_angle(Vec3(0, 0, 1), 1.2)))
    m.base_link = "ground"
    z = Vec3(0, 0, 1)
    m.add_joint(Joint("crank_joint", "ground", "crank", Vec3(0, 0, 0), [Dof(ROT, z, 0.5)]))
    m.add_joint(Joint("coupler_joint", "crank", "coupler", Vec3(0.08, 0.04, 0), [Dof(ROT, z, -0.7)],
                      JointProperties(actuated=False)))
    m.add_joint(Joint("rocker_joint", "coupler", "rocker", Vec3(0.25, 0.08, 0), [Dof(ROT, z, 1.4)],
                      JointProperties(actuated=False)))
    m.add_joint(Joint("loop_joint", "rocker", "ground", Vec3(0.3, 0, 0), [Dof(ROT, -z, 1.2)],
                      exclude_from_articulation=True))
    return m


def mimic_model() -> RobotModel:
    """Wheel (revolute leader) driving a rack (prismatic follower), plus a free body."""
    m = RobotModel("rack_pinion", fixed_base=True)
    m.add_link(link("base", visuals=[box_mesh("v")]))
    m.add_link(link("wheel", (0, 0, 0.1)))
    m.add_link(link("rack", (0.05, 0, 0.1), Quat.from_axis_angle(Vec3(0, 1, 0), 0.3)))
    m.add_link(link("loose", (1, 1, 1), metadata={"articulated": False}, mass=None))
    m.base_link = "base"
    m.add_joint(Joint("wheel_joint", "base", "wheel", Vec3(0, 0, 0.1), [Dof(ROT, Vec3(0, 1, 0), 0.4)],
                      JointProperties(effort=2.0)))
    m.add_joint(Joint("rack_joint", "base", "rack", Vec3(0.05, 0, 0.1),
                      [Dof(TRANS, Vec3(1, 0, 0), 0.02, Limit(0.0, 0.1))], JointProperties(effort=50.0)))
    m.couplings.append(Coupling("gear", DofRef("wheel_joint"), DofRef("rack_joint"), 0.05, 0.001))
    return m


def ball_model() -> RobotModel:
    """Arm with an excluded ball joint and an excluded cylindrical joint (MULTI)."""
    m = RobotModel("ball_robot")
    r = Quat.from_axis_angle(Vec3(1, 2, 3), 0.6)
    ax = [r.rotate(Vec3(1, 0, 0)), r.rotate(Vec3(0, 1, 0)), r.rotate(Vec3(0, 0, 1))]
    m.add_link(link("base"))
    m.add_link(link("arm", (0, 0, 0.2)))
    m.add_link(link("ball", (0, 0.1, 0.4), Quat.from_axis_angle(Vec3(1, 0, 0), 0.3)))
    m.add_link(link("sleeve", (0.2, 0.1, 0.4), Quat.from_axis_angle(Vec3(0, 0, 1), 0.8)))
    m.base_link = "base"
    m.add_joint(Joint("hinge", "base", "arm", Vec3(0, 0, 0.2), [Dof(TRANS, Vec3(0, 0, 1), 0.0)]))
    m.add_joint(Joint("ball_joint", "arm", "ball", Vec3(0, 0.1, 0.4),
                      [Dof(ROT, ax[0], 0.2), Dof(ROT, ax[1], -0.3), Dof(ROT, -ax[2], 0.1, Limit(-1, 1))],
                      exclude_from_articulation=True))
    m.add_joint(Joint("cyl_joint", "arm", "sleeve", Vec3(0.2, 0.1, 0.4),
                      [Dof(ROT, Vec3(0, 0, 1), 0.25), Dof(TRANS, Vec3(0, 0, -1), 0.03, Limit(-0.05, 0.02))],
                      exclude_from_articulation=True))
    return m


ALL_MODELS = [arm_model, four_bar_model, mimic_model, ball_model]
DEG = 180.0 / math.pi
