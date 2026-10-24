"""User-facing export settings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any, Dict, List, Optional


class ZeroPose(str, Enum):
    FUSION = "fusion"    # USD joint position 0 == Fusion joint value 0; limits kept as in Fusion
    CURRENT = "current"  # the exported pose becomes joint position 0; limits are shifted


class CollisionApproximation(str, Enum):
    NONE = "none"
    CONVEX_HULL = "convexHull"
    CONVEX_DECOMPOSITION = "convexDecomposition"
    MESH_SIMPLIFICATION = "meshSimplification"  # triangle mesh; only valid for kinematic/static bodies in PhysX
    BOUNDING_CUBE = "boundingCube"
    BOUNDING_SPHERE = "boundingSphere"
    SDF = "sdf"


class MeshQuality(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    VERY_HIGH = "very_high"


class DisconnectedPolicy(str, Enum):
    FIX_TO_BASE = "fix_to_base"  # weld unreachable parts to the base link (with a warning)
    FREE = "free"                # keep them as free rigid bodies outside the articulation
    DROP = "drop"                # omit them


@dataclass
class DriveDefaults:
    stiffness: float = 0.0        # per rad (revolute) / per m (prismatic)
    damping: float = 0.0
    max_force: Optional[float] = None  # None -> unlimited
    drive_type: str = "force"


@dataclass
class ExportOptions:
    robot_name: Optional[str] = None          # default: Fusion document name
    output_path: str = ""                     # .usda / .usd / .usdc
    zero_pose: ZeroPose = ZeroPose.FUSION
    fixed_base: Optional[bool] = None         # None: fixed iff something is grounded in Fusion
    base_link: Optional[str] = None           # Fusion occurrence/link name; None: auto
    source_up_axis: str = "auto"              # "auto" | "Y" | "Z"; output is always Z-up

    # geometry
    visual_quality: MeshQuality = MeshQuality.NORMAL
    collision_quality: MeshQuality = MeshQuality.LOW
    export_visuals: bool = True
    export_collisions: bool = True
    collision_approximation: CollisionApproximation = CollisionApproximation.CONVEX_DECOMPOSITION
    include_hidden: bool = False              # export bodies/occurrences whose light bulb is off

    # kinematics
    probe_joints: bool = True                 # nudge each joint in Fusion to measure true axis/sign
    disconnected: DisconnectedPolicy = DisconnectedPolicy.FIX_TO_BASE
    export_mimic_joints: bool = True          # Fusion motion links -> PhysxMimicJointAPI
    virtual_link_mass: float = 1e-3           # kg, for helper links of decomposed multi-DOF joints
    virtual_link_inertia: float = 1e-6        # kg·m²
    min_mass: float = 1e-4
    min_inertia: float = 1e-9

    # actuation
    actuate_by_default: bool = True           # add a DriveAPI to every non-mimic articulation joint
    drive: DriveDefaults = field(default_factory=DriveDefaults)
    self_collisions: bool = False

    # extras
    overrides_path: Optional[str] = None      # JSON file with per-joint/link properties
    write_model_json: bool = False            # dump the raw extracted model (re-export offline via the CLI)

    # post-export actions (see fusion2usd.actions); the flags are shortcuts for built-in actions
    write_isaaclab_cfg: bool = True
    write_urdf: bool = False
    post_actions: List[str] = field(default_factory=list)  # registered names or "pkg.module:Class"

    def action_names(self) -> List[str]:
        names = (["isaaclab_cfg"] if self.write_isaaclab_cfg else []) + (["urdf"] if self.write_urdf else [])
        return names + [a for a in self.post_actions if a not in names]

    def to_dict(self) -> Dict[str, Any]:
        def enc(v: Any) -> Any:
            return v.value if isinstance(v, Enum) else v
        d = asdict(self)
        return {k: (enc(getattr(self, k)) if k != "drive" else d["drive"]) for k in d}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExportOptions":
        opts = cls()
        types = {f.name: f.type for f in fields(cls)}
        for key, value in data.items():
            if key not in types:
                continue  # tolerate settings from other versions
            if key == "drive":
                opts.drive = DriveDefaults(**{k: v for k, v in value.items() if k in DriveDefaults.__dataclass_fields__})
                continue
            current = getattr(opts, key)
            if isinstance(current, Enum):
                value = type(current)(value)
            setattr(opts, key, value)
        return opts

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "ExportOptions":
        return cls.from_dict(json.loads(text))
