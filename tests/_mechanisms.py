"""Hand-built raw models shaped like what the Fusion reader produces."""

from __future__ import annotations

import math

from fusion2usd.core.inertia import MassProperties
from fusion2usd.core.math3d import Transform, Vec3
from fusion2usd.model.robot import Coupling, Dof, DofRef, DofType, Joint, Limit, Link, Material, Mesh, RobotModel

Z = Vec3(0, 0, 1)
I3 = ((1e-3, 0, 0), (0, 1e-3, 0), (0, 0, 1e-3))


def box_mesh(name: str, half: float = 0.05, material: str = "steel") -> Mesh:
    h = half
    pts = [(-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h), (-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)]
    tris = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 3, 0, 4, 3, 4, 7]
    return Mesh(name, [c for p in pts for c in p], tris, None, material)


def link(name: str, x: float, y: float = 0.0, mass: float = 1.0) -> Link:
    return Link(name, Transform(translation=Vec3(x, y, 0)), MassProperties(mass, Vec3(), I3),
                visuals=[box_mesh("body")], source=f"{name}:1",
                metadata={"attributes": {}})


def revolute(name: str, parent: str, child: str, origin: Vec3, value: float = 0.0, limit=None) -> Joint:
    return Joint(name, parent, child, origin, [Dof(DofType.ROTATION, Z, value, limit, "rot")],
                 source=name, source_type="RevoluteJointType", metadata={"fusion_name": name, "attributes": {}})


def four_bar() -> RobotModel:
    """Ground + crank + coupler + rocker; Fusion joints point in arbitrary directions."""
    m = RobotModel("four_bar")
    m.add_material(Material("steel", (0.6, 0.6, 0.65)))
    for l in (link("ground", 0, mass=5.0), link("crank", 0, 0.5), link("coupler", 1, 1), link("rocker", 2, 0.5)):
        m.add_link(l)
    m.grounded_links = ["ground"]
    m.base_link = "ground"
    lim = Limit(-math.pi / 2, math.pi / 2)
    # Fusion's occurrenceOne (child) is the moving one; directions here deliberately mixed.
    m.add_joint(revolute("Revolute 1", "ground", "crank", Vec3(0, 0, 0), 0.1, lim))
    m.add_joint(revolute("Revolute 2", "coupler", "crank", Vec3(0, 1, 0), 0.2))   # reversed w.r.t. tree
    m.add_joint(revolute("Revolute 3", "coupler", "rocker", Vec3(2, 1, 0), -0.3))
    m.add_joint(revolute("Revolute 4", "ground", "rocker", Vec3(2, 0, 0), 0.4))
    return m


def gripper() -> RobotModel:
    """Floating palm with a driven finger and a motion-linked mirrored finger, plus a cylindrical joint."""
    m = RobotModel("gripper")
    for l in (link("palm", 0, mass=2.0), link("finger_l", -0.1), link("finger_r", 0.1), link("slider", 0, 0.3)):
        m.add_link(l)
    m.add_joint(revolute("Revolute 1", "palm", "finger_l", Vec3(-0.1, 0, 0), 0.2, Limit(0, 1)))
    m.add_joint(revolute("Revolute 2", "finger_r", "palm", Vec3(0.1, 0, 0), -0.2, Limit(-1, 0)))
    cyl = Joint("Cylindrical 1", "palm", "slider", Vec3(0, 0.3, 0),
                [Dof(DofType.ROTATION, Vec3(0, 1, 0), 0.5, None, "rot"),
                 Dof(DofType.TRANSLATION, Vec3(0, 1, 0), 0.01, Limit(0, 0.05), "slide")],
                source="Cylindrical 1", source_type="CylindricalJointType", metadata={"attributes": {}})
    m.add_joint(cyl)
    # finger_r mirrors finger_l: q_r(Fusion) = -q_l
    m.couplings.append(Coupling("Motion Link 1", DofRef("Revolute 1"), DofRef("Revolute 2"), -1.0, 0.0))
    # screw: slide = 0.02 * rot  (offset from current values)
    m.couplings.append(Coupling("Motion Link 2", DofRef("Cylindrical 1", 0), DofRef("Cylindrical 1", 1),
                                0.02, 0.01 - 0.02 * 0.5))
    return m
