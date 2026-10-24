"""Turns a kinematically processed ``RobotModel`` into a USD Physics / PhysX articulation stage.

Layout (flat, like the Isaac Sim URDF importer)::

    /<Robot>            Xform, defaultPrim, ArticulationRootAPI
        /<link>         rigid bodies (world pose), with /visuals and /collisions
        /joints/<joint> USD Physics joints (+ root_joint for fixed bases)
        /Looks/<mat>    UsdPreviewSurface materials
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from ..core.math3d import Quat, Transform, Vec3
from ..core.naming import NameRegistry
from ..core.report import ExportReport
from ..model.robot import Coupling, Dof, DofType, Joint, JointKind, Link, Material, Mesh, RobotModel
from ..options import CollisionApproximation, ExportOptions
from .plan import JOINTS, LOOKS, ROOT_JOINT, ExportPlan
from .usda import Prim, UsdaStage

DEG = 180.0 / math.pi
INF = math.inf
_AXES = ("X", "Y", "Z")
_ALIGN_TOL = math.cos(1e-3)


def _pose_attrs(prim: Prim, index: int, pose: Transform) -> None:
    prim.attr(f"physics:localPos{index}", "point3f", pose.translation.as_tuple())
    prim.attr(f"physics:localRot{index}", "quatf", pose.rotation.normalized().canonical())


def _limit_pair(lower: Optional[float], upper: Optional[float], scale: float) -> Tuple[float, float]:
    return (-INF if lower is None else lower * scale, INF if upper is None else upper * scale)


def _dof_motion(dof_type: DofType, axis: Vec3, value: float) -> Transform:
    """Relative joint-frame motion for a DOF value about/along ``axis`` (joint-frame coordinates)."""
    if dof_type is DofType.ROTATION:
        return Transform(Quat.from_axis_angle(axis, value), Vec3())
    return Transform(Quat(), axis * value)


class UsdRobotBuilder:
    """Builds the in-memory USDA stage for a robot."""

    def __init__(self, options: ExportOptions, report: ExportReport) -> None:
        self.options = options
        self.report = report

    def build(self, model: RobotModel) -> UsdaStage:
        plan = ExportPlan(model, self.options, self.report)
        stage = UsdaStage(default_prim=plan.root_name, doc=f"{model.name} exported by fusion2usd")
        root = stage.define(plan.root_name, "Xform")
        root.metadata["kind"] = "component"
        if model.metadata:
            root.custom_data.update(model.metadata)

        _StageWriter(model, plan, self.options, self.report, stage, root).write()
        # Fixed base: the root Xform (the world joint pins the base). Floating base: the base link,
        # so PhysX does not pick some other link as the articulation's root.
        art = root if model.fixed_base else stage.get(plan.link_path(model.base_link))
        art.apply("PhysicsArticulationRootAPI", "PhysxArticulationAPI")
        art.attr("physxArticulation:enabledSelfCollisions", "bool", bool(self.options.self_collisions))
        return stage


class _StageWriter:
    """One-shot helper holding the per-build state."""

    def __init__(self, model: RobotModel, plan: ExportPlan, options: ExportOptions,
                 report: ExportReport, stage: UsdaStage, root: Prim) -> None:
        self.model = model
        self.plan = plan
        self.options = options
        self.report = report
        self.stage = stage
        self.root = root
        self.filtered: Dict[str, List[str]] = {}  # child link -> parent link paths

    def write(self) -> None:
        for link in self.model.links.values():
            self._write_link(link)
        joints = self.root.define(JOINTS, "Scope")
        if self.model.fixed_base:
            self._write_root_joint(joints)
        for joint in self.model.joints:
            self._write_joint(joints, joint)
        for coupling in self.plan.mimics:
            self._write_mimic(coupling)
        self._filter_decomposed_chains()
        for link, targets in self.filtered.items():
            prim = self.stage.get(self.plan.link_path(link))
            prim.apply("PhysicsFilteredPairsAPI")
            prim.rel("physics:filteredPairs", targets)
        if self.model.materials:
            looks = self.root.define(LOOKS, "Scope")
            for name, mat in self.model.materials.items():
                self._write_material(looks, name, mat)

    # ------------------------------------------------------------ links

    def _write_link(self, link: Link) -> None:
        prim = self.root.define(self.plan.link_names[link.name], "Xform")
        t = link.transform
        prim.attr("xformOp:translate", "double3", t.translation.as_tuple())
        prim.attr("xformOp:orient", "quatd", t.rotation.normalized().canonical())
        prim.attr("xformOp:scale", "double3", (1.0, 1.0, 1.0))
        prim.attr("xformOpOrder", "uniform token[]", ["xformOp:translate", "xformOp:orient", "xformOp:scale"])
        prim.apply("PhysicsRigidBodyAPI", "PhysicsMassAPI")
        if link.mass is None:
            self.report.warn(f"link {link.name!r} has no mass properties; PhysX will derive them from density")
        else:
            diag, axes = link.mass.principal()
            prim.attr("physics:mass", "float", link.mass.mass)
            prim.attr("physics:centerOfMass", "point3f", link.mass.com.as_tuple())
            prim.attr("physics:diagonalInertia", "float3", diag)
            prim.attr("physics:principalAxes", "quatf", axes)
        data: Dict[str, Any] = dict(link.metadata)
        if link.source:
            data["source"] = link.source
        if link.is_virtual:
            data["is_virtual"] = True
        prim.custom_data.update(data)

        if self.options.export_visuals and link.visuals:
            scope = prim.define("visuals", "Xform")
            names = NameRegistry()
            for mesh in link.visuals:
                self._write_mesh(scope, names.claim(mesh.name, "mesh"), mesh, visual=True)
        self._write_collisions(prim, link)

    def _write_collisions(self, prim: Prim, link: Link) -> None:
        if not self.options.export_collisions:
            return
        approx = link.collision_approximation or self.options.collision_approximation
        approx = approx.value if isinstance(approx, CollisionApproximation) else str(approx)
        if approx == CollisionApproximation.NONE.value:
            return
        meshes = link.collisions or link.visuals
        meshes = [m for m in meshes if m.vertex_count and m.triangle_count]
        if not meshes:
            return
        scope = prim.define("collisions", "Xform")
        scope.attr("purpose", "uniform token", "guide")
        names = NameRegistry()
        for mesh in meshes:
            m = self._write_mesh(scope, names.claim(mesh.name, "collision"), mesh, visual=False)
            if m is None:
                continue
            m.apply("PhysicsCollisionAPI", "PhysicsMeshCollisionAPI")
            m.attr("physics:approximation", "uniform token", approx)
            if approx == CollisionApproximation.SDF.value:
                m.apply("PhysxSDFMeshCollisionAPI")

    def _write_mesh(self, parent: Prim, name: str, mesh: Mesh, visual: bool) -> Optional[Prim]:
        if not mesh.vertex_count or not mesh.triangle_count:
            self.report.warn(f"mesh {mesh.name!r} under {parent.path} is empty, skipped")
            return None
        prim = parent.define(name, "Mesh")
        lo, hi = mesh.bounds()
        prim.attr("points", "point3f[]", mesh.points[: mesh.vertex_count * 3])
        prim.attr("faceVertexCounts", "int[]", [3] * mesh.triangle_count)
        prim.attr("faceVertexIndices", "int[]", mesh.indices[: mesh.triangle_count * 3])
        prim.attr("extent", "float3[]", [lo.as_tuple(), hi.as_tuple()])
        prim.attr("subdivisionScheme", "uniform token", "none")
        if visual and mesh.normals is not None and len(mesh.normals) == len(mesh.points):
            prim.attr("normals", "normal3f[]", mesh.normals, metadata={"interpolation": "vertex"})
        if visual and mesh.material:
            mat = self.model.materials.get(mesh.material)
            if mat is None:
                self.report.warn(f"mesh {mesh.name!r}: unknown material {mesh.material!r}")
            else:
                prim.apply("MaterialBindingAPI")
                prim.rel("material:binding", [self.plan.material_path(mesh.material)])
                prim.attr("primvars:displayColor", "color3f[]", [mat.color],
                          metadata={"interpolation": "constant"})
        return prim

    def _write_material(self, looks: Prim, name: str, mat: Material) -> None:
        prim = looks.define(self.plan.material_names[name], "Material")
        shader = prim.define("Shader", "Shader")
        shader.attr("info:id", "uniform token", "UsdPreviewSurface")
        shader.attr("inputs:diffuseColor", "color3f", tuple(mat.color))
        shader.attr("inputs:opacity", "float", mat.opacity)
        shader.attr("inputs:roughness", "float", mat.roughness)
        shader.attr("inputs:metallic", "float", mat.metallic)
        shader.attr("outputs:surface", "token")
        prim.attr("outputs:surface", "token", connect=f"{shader.path}.outputs:surface")

    # ------------------------------------------------------------ joints

    def _write_root_joint(self, joints: Prim) -> None:
        base = self.model.base_link
        if base is None or base not in self.model.links:
            self.report.warn("fixed base requested but no valid base link; root_joint not written")
            return
        prim = joints.define(ROOT_JOINT, "PhysicsFixedJoint")
        prim.rel("physics:body1", [self.plan.link_path(base)])
        _pose_attrs(prim, 0, self.model.links[base].transform)
        _pose_attrs(prim, 1, Transform.identity())

    def _write_joint(self, joints: Prim, joint: Joint) -> None:
        parent = self.model.links.get(joint.parent)
        child = self.model.links.get(joint.child)
        if parent is None or child is None:
            self.report.warn(f"joint {joint.name!r} references a missing link, skipped")
            return
        kind = joint.kind
        if kind is JointKind.MULTI and not joint.exclude_from_articulation:
            self.report.warn(f"joint {joint.name!r} has several DOFs but is part of the articulation; "
                             "writing it as an excluded D6 joint")
        type_name = {
            JointKind.FIXED: "PhysicsFixedJoint",
            JointKind.REVOLUTE: "PhysicsRevoluteJoint",
            JointKind.PRISMATIC: "PhysicsPrismaticJoint",
            JointKind.MULTI: "PhysicsJoint",
        }[kind]
        prim = joints.define(self.plan.joint_names[joint.name], type_name)
        prim.rel("physics:body0", [self.plan.link_path(joint.parent)])
        prim.rel("physics:body1", [self.plan.link_path(joint.child)])

        if kind is JointKind.MULTI:
            frame, motion = self._write_d6(prim, joint)
        else:
            frame = self._joint_frame(joint)
            motion = _dof_motion(joint.dofs[0].type, Vec3(1.0, 0.0, 0.0), joint.dofs[0].value) \
                if joint.dofs else Transform.identity()
        _pose_attrs(prim, 0, parent.transform.inverse() * frame)
        _pose_attrs(prim, 1, child.transform.inverse() * frame * motion)

        excluded = joint.exclude_from_articulation or kind is JointKind.MULTI
        prim.attr("physics:jointEnabled", "bool", bool(joint.enabled))
        prim.attr("physics:collisionEnabled", "bool", False)
        prim.attr("physics:excludeFromArticulation", "bool", excluded)
        if kind in (JointKind.REVOLUTE, JointKind.PRISMATIC):
            self._write_single_dof(prim, joint, joint.dofs[0], kind is JointKind.REVOLUTE)

        data: Dict[str, Any] = dict(joint.metadata)
        if joint.source:
            data["source"] = joint.source
        if joint.source_type:
            data["source_type"] = joint.source_type
        props = joint.properties.to_dict()
        if props:
            data["properties"] = props
        prim.custom_data.update(data)

        self._filter(joint.parent, joint.child)

    def _filter(self, a: str, b: str) -> None:
        targets = self.filtered.setdefault(b, [])
        path = self.plan.link_path(a)
        if a != b and path not in targets:
            targets.append(path)

    def _filter_decomposed_chains(self) -> None:
        """Bodies of one split multi-DOF joint (incl. helpers) must never collide with each other."""
        chains: Dict[str, List[str]] = {}
        for joint in self.model.joints:
            origin = joint.metadata.get("decomposed_from")
            if origin:
                chain = chains.setdefault(origin, [])
                chain += [n for n in (joint.parent, joint.child) if n not in chain]
        for chain in chains.values():
            for i, a in enumerate(chain):
                for b in chain[i + 1:]:
                    self._filter(a, b)

    def _joint_frame(self, joint: Joint) -> Transform:
        if not joint.dofs:
            rot = joint.frame_hint.rotation if joint.frame_hint else Quat()
            return Transform(rot, joint.origin)
        y_hint = joint.frame_hint.axis(1) if joint.frame_hint else None
        return Transform.with_x_axis(joint.origin, joint.dofs[0].axis, y_hint)

    def _write_single_dof(self, prim: Prim, joint: Joint, dof: Dof, angular: bool) -> None:
        scale = DEG if angular else 1.0
        dof_name = "angular" if angular else "linear"
        prim.attr("physics:axis", "uniform token", "X")
        if dof.limit is not None and dof.limit.is_bounded:
            lo, hi = _limit_pair(dof.limit.lower, dof.limit.upper, scale)
            prim.attr("physics:lowerLimit", "float", lo)
            prim.attr("physics:upperLimit", "float", hi)

        props = joint.properties
        if self.plan.is_driven(joint):
            drive = self.options.drive
            stiffness = drive.stiffness if props.stiffness is None else props.stiffness
            damping = drive.damping if props.damping is None else props.damping
            effort = drive.max_force if props.effort is None else props.effort
            target = dof.value if props.target is None else props.target
            per_unit = 1.0 / scale  # USD angular gains are per degree
            ns = f"drive:{dof_name}:physics"
            prim.apply(f"PhysicsDriveAPI:{dof_name}")
            prim.attr(f"{ns}:type", "uniform token", props.drive_type or drive.drive_type)
            prim.attr(f"{ns}:stiffness", "float", stiffness * per_unit)
            prim.attr(f"{ns}:damping", "float", damping * per_unit)
            prim.attr(f"{ns}:maxForce", "float", INF if effort is None else effort)
            prim.attr(f"{ns}:targetPosition", "float", target * scale)
            prim.attr(f"{ns}:targetVelocity", "float", 0.0)

        # Also on loop-closure joints: PhysX ignores it there, but it records the position the limits refer to.
        prim.apply(f"PhysicsJointStateAPI:{dof_name}")
        prim.attr(f"state:{dof_name}:physics:position", "float", dof.value * scale)
        prim.attr(f"state:{dof_name}:physics:velocity", "float", 0.0)

        physx = {
            "physxJoint:maxJointVelocity": None if props.velocity is None else props.velocity * scale,
            "physxJoint:armature": props.armature,
            "physxJoint:jointFriction": props.friction,
        }
        physx = {k: v for k, v in physx.items() if v is not None}
        if physx:
            prim.apply("PhysxJointAPI")
            for name, value in physx.items():
                prim.attr(name, "float", value)

    def _write_d6(self, prim: Prim, joint: Joint) -> Tuple[Transform, Transform]:
        """Generic joint: free DOFs map to frame axes, every other axis is locked."""
        dofs = joint.dofs
        x = dofs[0].axis.normalized()
        y_hint = next((d.axis for d in dofs[1:] if x.cross(d.axis.normalized()).norm() > 1e-6), None)
        if y_hint is None and joint.frame_hint is not None:
            y_hint = joint.frame_hint.axis(1)
        frame = Transform.with_x_axis(joint.origin, x, y_hint)
        frame_axes = [frame.axis(i) for i in range(3)]

        free: Dict[str, Tuple[Dof, float]] = {}
        translation = Vec3()
        rotations = [d for d in dofs if d.type is DofType.ROTATION]
        if len(rotations) == 3:
            # Ball joint: rotation is entirely free, its axes need not be orthogonal away from zero.
            for i, axis in enumerate(_AXES):
                free["rot" + axis] = (rotations[i], 1.0)
            dofs = [d for d in dofs if d.type is not DofType.ROTATION]
        for dof in dofs:
            a = dof.axis.normalized()
            dots = [a.dot(e) for e in frame_axes]
            i = max(range(3), key=lambda k: abs(dots[k]))
            token = ("rot" if dof.type is DofType.ROTATION else "trans") + _AXES[i]
            if abs(dots[i]) < _ALIGN_TOL:
                self.report.warn(f"joint {joint.name!r}: DOF axis {a.as_tuple()} is not aligned with "
                                 "the joint frame; DOF skipped")
                continue
            if token in free:
                self.report.warn(f"joint {joint.name!r}: duplicate DOF {token}, skipped")
                continue
            sign = 1.0 if dots[i] > 0 else -1.0
            free[token] = (dof, sign)
            local = Vec3(*(sign if k == i else 0.0 for k in range(3)))
            if dof.type is DofType.TRANSLATION:
                translation = translation + local * dof.value

        for kind in ("trans", "rot"):
            for axis in _AXES:
                token = kind + axis
                if token in free:
                    dof, sign = free[token]
                    if dof.limit is None or not dof.limit.is_bounded:
                        continue
                    if kind == "rot":
                        self.report.warn(f"joint {joint.name!r}: rotational limits on {token} are not "
                                         "exported for multi-DOF joints")
                        continue
                    limit = dof.limit if sign > 0 else dof.limit.reversed()
                    low, high = _limit_pair(limit.lower, limit.upper, 1.0)
                else:
                    low, high = 1.0, -1.0  # low > high locks the axis
                prim.apply(f"PhysicsLimitAPI:{token}")
                prim.attr(f"limit:{token}:physics:low", "float", low)
                prim.attr(f"limit:{token}:physics:high", "float", high)
        # Free rotations are unconstrained, so their values need not be encoded: the body frames
        # coincide in orientation at the export pose. Translations are encoded so limits keep the
        # CAD convention.
        return frame, Transform(Quat(), translation)

    def _write_mimic(self, coupling: Coupling) -> None:
        follower = self.model.joint(coupling.follower.joint)
        leader = self.model.joint(coupling.leader.joint)
        inst = "rotX" if follower.kind is JointKind.REVOLUTE else "transX"
        ref_axis = "rotX" if leader.kind is JointKind.REVOLUTE else "transX"
        prim = self.stage.get(self.plan.joint_path(follower.name))
        if prim is None:
            return
        ns = f"physxMimicJoint:{inst}"
        prim.apply(f"PhysxMimicJointAPI:{inst}")
        prim.rel(f"{ns}:referenceJoint", [self.plan.joint_path(leader.name)])
        prim.attr(f"{ns}:referenceJointAxis", "uniform token", ref_axis)
        f_scale = DEG if follower.kind is JointKind.REVOLUTE else 1.0
        l_scale = DEG if leader.kind is JointKind.REVOLUTE else 1.0
        # q_f = r q_l + c  in SI  ->  q_f' + gearing q_l' + offset = 0  in USD units (deg / m)
        prim.attr(f"{ns}:gearing", "float", -coupling.ratio * f_scale / l_scale)
        prim.attr(f"{ns}:offset", "float", -coupling.offset * f_scale)
