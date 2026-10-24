"""Decisions shared by the USD builder and the Isaac Lab config writer (prim names, drives, mimics)."""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from ..core.naming import NameRegistry, sanitize
from ..core.report import ExportReport
from ..model.robot import Coupling, Joint, JointKind, RobotModel
from ..options import ExportOptions
from ..kinematics import roles

LOOKS = "Looks"
JOINTS = "joints"
ROOT_JOINT = "root_joint"
SINGLE_DOF = roles.SINGLE_DOF


class ExportPlan:
    """Maps model names to prim names and decides which joints get drives or mimic constraints.

    Names coming out of the kinematic passes are normally kept verbatim; they are only changed on
    collisions with the reserved scopes (``joints``, ``Looks``, ``root_joint``).
    """

    def __init__(self, model: RobotModel, options: ExportOptions, report: Optional[ExportReport] = None) -> None:
        self.model = model
        self.options = options
        self._report = report
        self.root_name = sanitize(model.name or "Robot", "Robot")
        self.root_path = "/" + self.root_name

        links = NameRegistry({JOINTS, LOOKS})
        self.link_names: Dict[str, str] = {n: links.claim(n, "link") for n in model.links}
        joints = NameRegistry({ROOT_JOINT} if model.fixed_base else set())
        self.joint_names: Dict[str, str] = {j.name: joints.claim(j.name, "joint") for j in model.joints}
        mats = NameRegistry()
        self.material_names: Dict[str, str] = {n: mats.claim(n, "material") for n in model.materials}
        for kind, mapping in (("link", self.link_names), ("joint", self.joint_names)):
            for old, new in mapping.items():
                if old != new:
                    self._warn(f"{kind} {old!r} renamed to {new!r} for USD")

        self.mimics: List[Coupling] = self._valid_mimics() if options.export_mimic_joints else []
        self.followers: Set[str] = {c.follower.joint for c in self.mimics}
        self.mimic_joints: Set[str] = self.followers | {c.leader.joint for c in self.mimics}

    # -- paths
    def link_path(self, link: str) -> str:
        return f"{self.root_path}/{self.link_names[link]}"

    def joint_path(self, joint: str) -> str:
        return f"{self.root_path}/{JOINTS}/{self.joint_names[joint]}"

    def material_path(self, material: str) -> str:
        return f"{self.root_path}/{LOOKS}/{self.material_names[material]}"

    # -- joint classification
    in_articulation = staticmethod(roles.in_articulation)

    def is_driven(self, joint: Joint) -> bool:
        return roles.is_driven(joint, self.followers, self.options)

    def _valid_mimics(self) -> List[Coupling]:
        out: List[Coupling] = []
        seen: Set[str] = set()
        for c in self.model.couplings:
            leader = self.model.find_joint(c.leader.joint)
            follower = self.model.find_joint(c.follower.joint)
            if leader is None or follower is None:
                self._warn(f"coupling {c.name!r}: unknown joint, skipped")
            elif not (self.in_articulation(leader) and self.in_articulation(follower)):
                self._warn(f"coupling {c.name!r}: mimic joints must be single-DOF articulation joints, skipped")
            elif c.leader.dof != 0 or c.follower.dof != 0:
                self._warn(f"coupling {c.name!r}: references a DOF index other than 0, skipped")
            elif follower.name in seen:
                self._warn(f"coupling {c.name!r}: joint {follower.name!r} already follows another joint, skipped")
            elif leader.name == follower.name:
                self._warn(f"coupling {c.name!r}: joint follows itself, skipped")
            else:
                seen.add(follower.name)
                out.append(c)
        return out

    def _warn(self, msg: str) -> None:
        if self._report is not None:
            self._report.warn(msg)
