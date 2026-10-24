"""Assigns unique, USD-safe names to links, joints, meshes and materials."""

from __future__ import annotations

from typing import Dict

from ..core.naming import NameRegistry
from ..model.robot import DofRef, RobotModel
from .pipeline import ModelPass, PassContext

# Child prims the USD writer creates directly under the robot root.
RESERVED_ROOT_NAMES = {"joints", "Looks"}


class NamingPass(ModelPass):
    def run(self, model: RobotModel, ctx: PassContext) -> RobotModel:
        self._rename_links(model)
        self._rename_joints(model)
        self._rename_materials(model)
        for link in model.links.values():
            for meshes, fallback in ((link.visuals, "visual"), (link.collisions, "collision")):
                registry = NameRegistry()
                for mesh in meshes:
                    mesh.name = registry.claim(mesh.name, fallback)
        return model

    @staticmethod
    def _rename_links(model: RobotModel) -> None:
        registry = NameRegistry(reserved=RESERVED_ROOT_NAMES)
        mapping: Dict[str, str] = {
            old: registry.claim(link.metadata.get("export_name", old), "link") for old, link in model.links.items()
        }
        renamed = {}
        for old, link in model.links.items():
            link.name = mapping[old]
            renamed[link.name] = link
        model.links = renamed
        for j in model.joints:
            j.parent, j.child = mapping.get(j.parent, j.parent), mapping.get(j.child, j.child)
        if model.base_link:
            model.base_link = mapping.get(model.base_link, model.base_link)
        model.grounded_links = [mapping.get(g, g) for g in model.grounded_links]

    @staticmethod
    def _rename_joints(model: RobotModel) -> None:
        registry = NameRegistry(reserved={"root_joint"})
        mapping: Dict[str, str] = {}
        for j in model.joints:
            new = registry.claim(j.metadata.get("export_name", j.name), "joint")
            mapping.setdefault(j.name, new)  # duplicate raw names are ambiguous for couplings anyway
            j.name = new
        for c in model.couplings:
            c.leader = DofRef(mapping.get(c.leader.joint, c.leader.joint), c.leader.dof)
            c.follower = DofRef(mapping.get(c.follower.joint, c.follower.joint), c.follower.dof)

    @staticmethod
    def _rename_materials(model: RobotModel) -> None:
        registry = NameRegistry()
        mapping: Dict[str, str] = {}
        materials = {}
        for old, mat in model.materials.items():
            mat.name = mapping[old] = registry.claim(old, "material")
            materials[mat.name] = mat
        model.materials = materials
        for link in model.links.values():
            for mesh in link.visuals + link.collisions:
                if mesh.material is not None:
                    mesh.material = mapping.get(mesh.material, mesh.material)
