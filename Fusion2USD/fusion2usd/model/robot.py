"""CAD-agnostic intermediate representation of an articulated mechanism.

Conventions (everything downstream relies on them):

* SI units: meters, kilograms, radians, seconds. Effort is N·m (rotation) or N (translation).
* ``Link.transform`` is the link frame in world coordinates. Meshes and mass properties are
  expressed in the link frame.
* A ``Joint`` connects ``parent`` → ``child``. Its DOFs are given in *world* coordinates at the
  exported pose; ``Dof.value`` is the CAD joint value at that pose and ``Dof.limit`` is in the
  same CAD convention (value 0 = CAD zero position). Positive motion moves the child relative to
  the parent along/about ``axis`` (right-hand rule).
* A joint with several DOFs is a CAD multi-DOF joint (cylindrical, planar, ball, ...). The
  decomposition pass turns it into a chain of single-DOF joints before USD writing.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..core.inertia import MassProperties
from ..core.math3d import Transform, Vec3


class DofType(str, Enum):
    ROTATION = "rotation"
    TRANSLATION = "translation"


class JointKind(str, Enum):
    FIXED = "fixed"
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    MULTI = "multi"


@dataclass
class Limit:
    lower: Optional[float] = None
    upper: Optional[float] = None

    def reversed(self) -> "Limit":
        return Limit(
            None if self.upper is None else -self.upper,
            None if self.lower is None else -self.lower,
        )

    def shifted(self, delta: float) -> "Limit":
        return Limit(
            None if self.lower is None else self.lower + delta,
            None if self.upper is None else self.upper + delta,
        )

    @property
    def is_bounded(self) -> bool:
        return self.lower is not None or self.upper is not None


@dataclass
class Dof:
    type: DofType
    axis: Vec3
    value: float = 0.0
    limit: Optional[Limit] = None
    label: str = ""  # e.g. "rot", "slide", "pitch"; used for decomposed joint names

    def reversed(self) -> "Dof":
        return Dof(self.type, self.axis, -self.value, self.limit.reversed() if self.limit else None, self.label)


@dataclass
class JointProperties:
    """Actuation/dynamics data. ``None`` means "use the export default"."""

    effort: Optional[float] = None        # N·m or N  -> drive maxForce
    velocity: Optional[float] = None      # rad/s or m/s -> physxJoint:maxJointVelocity
    stiffness: Optional[float] = None     # per rad or per m
    damping: Optional[float] = None       # per rad/s or per m/s
    armature: Optional[float] = None      # kg·m² or kg
    friction: Optional[float] = None      # unitless joint friction coefficient
    actuated: Optional[bool] = None       # False -> passive joint (no drive)
    drive_type: Optional[str] = None      # "force" | "acceleration"
    target: Optional[float] = None        # drive target position (CAD convention)

    FIELDS = ("effort", "velocity", "stiffness", "damping", "armature", "friction",
              "actuated", "drive_type", "target")

    def merged(self, other: "JointProperties") -> "JointProperties":
        """Values in ``other`` win where they are set."""
        out = copy.copy(self)
        for f in self.FIELDS:
            v = getattr(other, f)
            if v is not None:
                setattr(out, f, v)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {f: getattr(self, f) for f in self.FIELDS if getattr(self, f) is not None}


@dataclass
class Material:
    name: str
    color: Tuple[float, float, float] = (0.7, 0.7, 0.7)  # linear-ish RGB, 0..1
    opacity: float = 1.0
    roughness: float = 0.5
    metallic: float = 0.0


@dataclass
class Mesh:
    """Triangle mesh in its link's frame. Flat arrays keep large CAD meshes cheap."""

    name: str
    points: List[float]                  # x0, y0, z0, x1, ...
    indices: List[int]                   # 3 per triangle
    normals: Optional[List[float]] = None  # per-vertex, same layout as points
    material: Optional[str] = None

    @property
    def vertex_count(self) -> int:
        return len(self.points) // 3

    @property
    def triangle_count(self) -> int:
        return len(self.indices) // 3

    def transformed(self, t: Transform) -> "Mesh":
        pts: List[float] = []
        for i in range(0, len(self.points), 3):
            pts.extend(t.apply_point(Vec3(self.points[i], self.points[i + 1], self.points[i + 2])))
        nrm = None
        if self.normals is not None:
            nrm = []
            for i in range(0, len(self.normals), 3):
                nrm.extend(t.apply_vector(Vec3(self.normals[i], self.normals[i + 1], self.normals[i + 2])))
        return Mesh(self.name, pts, list(self.indices), nrm, self.material)

    def bounds(self) -> Tuple[Vec3, Vec3]:
        xs, ys, zs = self.points[0::3], self.points[1::3], self.points[2::3]
        return Vec3(min(xs), min(ys), min(zs)), Vec3(max(xs), max(ys), max(zs))


@dataclass
class Link:
    name: str
    transform: Transform = field(default_factory=Transform)
    mass: Optional[MassProperties] = None
    visuals: List[Mesh] = field(default_factory=list)
    collisions: List[Mesh] = field(default_factory=list)
    collision_approximation: Optional[str] = None  # overrides the export default
    collision_separate: List[str] = field(default_factory=list)  # meshes kept out of a link hull
    collision_into: Optional[str] = None  # link whose hull takes this link's meshes (link_hull)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""          # CAD identity (e.g. Fusion occurrence path)
    is_virtual: bool = False  # massless helper link from joint decomposition


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    origin: Vec3                       # world point on the motion axis / joint anchor
    dofs: List[Dof] = field(default_factory=list)
    properties: JointProperties = field(default_factory=JointProperties)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    source_type: str = ""              # CAD joint type, informational
    frame_hint: Optional[Transform] = None  # world orientation hint (used for fixed joints / roll)
    exclude_from_articulation: bool = False
    loop_closure_preferred: bool = False    # user hint: cut the loop here if possible
    enabled: bool = True

    @property
    def kind(self) -> JointKind:
        if not self.dofs:
            return JointKind.FIXED
        if len(self.dofs) > 1:
            return JointKind.MULTI
        return JointKind.REVOLUTE if self.dofs[0].type is DofType.ROTATION else JointKind.PRISMATIC

    @property
    def is_ball(self) -> bool:
        return len(self.dofs) == 3 and all(d.type is DofType.ROTATION for d in self.dofs)

    def reversed(self) -> "Joint":
        """Same constraint with parent and child swapped (motion signs and DOF order flip)."""
        j = copy.copy(self)
        j.parent, j.child = self.child, self.parent
        j.dofs = [d.reversed() for d in reversed(self.dofs)]
        if self.properties.target is not None:
            j.properties = copy.copy(self.properties)
            j.properties.target = -self.properties.target
        j.metadata = dict(self.metadata)
        j.metadata["reversed"] = not self.metadata.get("reversed", False)
        return j

    def other(self, link: str) -> str:
        return self.child if link == self.parent else self.parent


class SensorKind(str, Enum):
    CAMERA = "camera"    # USD Camera prim; frame uses the ROS optical convention (+Z forward, +X right, +Y down)
    IMU = "imu"
    CONTACT = "contact"  # body-level contact reporting on the sensor's link
    FRAME = "frame"      # named reference frame (tool tip, mount point, ...)


@dataclass
class Sensor:
    """A sensor or named frame rigidly attached to a link.

    ``transform`` is the sensor frame in world coordinates at the exported pose (like ``Link.transform``).
    ``params`` holds parsed SI values (see ``kinematics.sensors``); ``metadata`` is exported as customData.
    """

    name: str
    kind: SensorKind
    link: str
    transform: Transform = field(default_factory=Transform)
    params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""


@dataclass(frozen=True)
class DofRef:
    joint: str
    dof: int = 0


@dataclass
class Coupling:
    """Kinematic coupling (gear, rack & pinion, motion link): ``q_follower = ratio * q_leader + offset``.

    Values in SI (rad or m) and in the CAD sign convention of the referenced joints.
    """

    name: str
    leader: DofRef
    follower: DofRef
    ratio: float
    offset: float = 0.0
    source: str = ""


@dataclass
class RobotModel:
    name: str
    links: Dict[str, Link] = field(default_factory=dict)
    joints: List[Joint] = field(default_factory=list)
    couplings: List[Coupling] = field(default_factory=list)
    materials: Dict[str, Material] = field(default_factory=dict)
    base_link: Optional[str] = None
    fixed_base: bool = False
    grounded_links: List[str] = field(default_factory=list)  # links welded to the world in CAD
    root_pose: Transform = field(default_factory=Transform)  # world pose of the asset origin (= base link)
    sensors: List[Sensor] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- construction
    def add_link(self, link: Link) -> Link:
        if link.name in self.links:
            raise ValueError(f"duplicate link name {link.name!r}")
        self.links[link.name] = link
        return link

    def add_joint(self, joint: Joint) -> Joint:
        if any(j.name == joint.name for j in self.joints):
            raise ValueError(f"duplicate joint name {joint.name!r}")
        self.joints.append(joint)
        return joint

    def add_material(self, material: Material) -> Material:
        return self.materials.setdefault(material.name, material)

    # -- queries
    def joint(self, name: str) -> Joint:
        for j in self.joints:
            if j.name == name:
                return j
        raise KeyError(name)

    def find_joint(self, name: str) -> Optional[Joint]:
        return next((j for j in self.joints if j.name == name), None)

    def joints_of(self, link: str) -> Iterator[Joint]:
        return (j for j in self.joints if link in (j.parent, j.child))

    def copy(self) -> "RobotModel":
        return copy.deepcopy(self)

    # -- renaming keeps all cross references consistent
    def rename_link(self, old: str, new: str) -> None:
        if old == new:
            return
        link = self.links.pop(old)
        link.name = new
        self.links[new] = link
        for j in self.joints:
            if j.parent == old:
                j.parent = new
            if j.child == old:
                j.child = new
        if self.base_link == old:
            self.base_link = new
        self.grounded_links = [new if g == old else g for g in self.grounded_links]

    def reverse_joint(self, name: str) -> Joint:
        """Swap a joint's parent/child in place and re-express couplings that reference it."""
        idx = next(i for i, j in enumerate(self.joints) if j.name == name)
        old = self.joints[idx]
        self.joints[idx] = new = old.reversed()
        n = len(old.dofs)
        for c in self.couplings:
            # q' = -q, and DOF order is mirrored.
            if c.leader.joint == name:
                c.leader = DofRef(name, n - 1 - c.leader.dof)
                c.ratio = -c.ratio
            if c.follower.joint == name:
                c.follower = DofRef(name, n - 1 - c.follower.dof)
                c.ratio, c.offset = -c.ratio, -c.offset
        return new

    def rename_joint(self, old: str, new: str) -> None:
        if old == new:
            return
        self.joint(old).name = new
        for c in self.couplings:
            if c.leader.joint == old:
                c.leader = DofRef(new, c.leader.dof)
            if c.follower.joint == old:
                c.follower = DofRef(new, c.follower.dof)
