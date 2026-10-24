"""Joint roles shared by validation and the writers, so they cannot drift apart."""

from __future__ import annotations

from typing import Collection

from ..model.robot import Joint, JointKind
from ..options import ExportOptions

SINGLE_DOF = (JointKind.REVOLUTE, JointKind.PRISMATIC)


def in_articulation(joint: Joint) -> bool:
    """A single-DOF joint solved in reduced coordinates (not a loop closure / D6)."""
    return not joint.exclude_from_articulation and joint.kind in SINGLE_DOF


def is_driven(joint: Joint, followers: Collection[str], options: ExportOptions) -> bool:
    """Whether the joint gets a DriveAPI / Isaac Lab actuator."""
    if not in_articulation(joint) or joint.name in followers:
        return False
    actuated = joint.properties.actuated
    return options.actuate_by_default if actuated is None else actuated
