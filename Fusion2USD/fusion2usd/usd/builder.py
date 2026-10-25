"""Turns a kinematically processed ``RobotModel`` into a USD Physics articulation stage.

Engine-specific schemas (PhysX or Newton) come from ``engines``; everything written here is plain UsdPhysics.

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
from ..model.robot import (
    Coupling, Dof, DofType, Joint, JointKind, Link, Material, Mesh, RobotModel, Sensor, SensorKind,
)
from ..options import CollisionApproximation, ExportOptions, PhysicsEngine, ZeroPose
from .engines import EngineSchemas, engine_schemas
from .plan import JOINTS, LOOKS, ROOT_JOINT, ExportPlan
from .usda import Prim, UsdaStage

DEG = 180.0 / math.pi
MIMIC_LIMIT_DEG = 3600.0  # finite stand-in for continuous revolute joints in mimic pairs
INF = math.inf
_AXES = ("X", "Y", "Z")
_ALIGN_EPS = 1e-3                 # rad; D6 DOF axes closer than this to a frame axis are aligned
_SNAP_TOL = math.cos(math.radians(2.0))  # up to this, a DOF axis is snapped to the frame axis with a warning


def _pose_attrs(prim: Prim, index: int, pose: Transform) -> None:
    prim.attr(f"physics:localPos{index}", "point3f", pose.translation.as_tuple())
    prim.attr(f"physics:localRot{index}", "quatf", pose.rotation.normalized().canonical())


def _xform_attrs(prim: Prim, t: Transform) -> None:
    prim.attr("xformOp:translate", "double3", t.translation.as_tuple())
    prim.attr("xformOp:orient", "quatd", t.rotation.normalized().canonical())
    prim.attr("xformOp:scale", "double3", (1.0, 1.0, 1.0))
    prim.attr("xformOpOrder", "uniform token[]", ["xformOp:translate", "xformOp:orient", "xformOp:scale"])


# CAD camera frames use the ROS optical convention (+Z forward, +Y down); USD cameras look down -Z with +Y up.
ROS_TO_USD_CAMERA = Transform(Quat.from_axis_angle(Vec3(1.0, 0.0, 0.0), math.pi))


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

        engine = engine_schemas(self.options.physics_engine)
        _StageWriter(model, plan, self.options, self.report, stage, root, engine).write()
        # Fixed base: the root Xform (the world joint pins the base). Floating base: the base link,
        # so the simulator does not pick some other link as the articulation's root.
        art = root if model.fixed_base else stage.get(plan.link_path(model.base_link))
        art.apply("PhysicsArticulationRootAPI")
        engine.articulation_root(art, bool(self.options.self_collisions))
        return stage


class _StageWriter:
    """One-shot helper holding the per-build state."""

    def __init__(self, model: RobotModel, plan: ExportPlan, options: ExportOptions,
                 report: ExportReport, stage: UsdaStage, root: Prim, engine: EngineSchemas) -> None:
        self.model = model
        self.plan = plan
        self.options = options
        self.report = report
        self.stage = stage
        self.root = root
        self.engine = engine
        self.newton = options.physics_engine is PhysicsEngine.NEWTON
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
        if self.newton:
            self._newton_notes()
        for sensor in self.model.sensors:
            self._write_sensor(sensor)
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
        _xform_attrs(prim, t)
        prim.apply("PhysicsRigidBodyAPI", "PhysicsMassAPI")
        if link.mass is None:
            self.report.warn(f"link {link.name!r} has no mass properties; the simulator derives them from its colliders")
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
            self.engine.collision(m, approx)

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
        # Newton solves multi-DOF joints in reduced coordinates; PhysX gets them split (see decompose).
        spherical = self.newton and joint.is_ball
        native_multi = self.newton and kind is JointKind.MULTI
        if kind is JointKind.MULTI and not native_multi and not joint.exclude_from_articulation:
            self.report.warn(f"joint {joint.name!r} has several DOFs but is part of the articulation; "
                             "writing it as an excluded D6 joint")
        type_name = "PhysicsSphericalJoint" if spherical else {
            JointKind.FIXED: "PhysicsFixedJoint",
            JointKind.REVOLUTE: "PhysicsRevoluteJoint",
            JointKind.PRISMATIC: "PhysicsPrismaticJoint",
            JointKind.MULTI: "PhysicsJoint",
        }[kind]
        prim = joints.define(self.plan.joint_names[joint.name], type_name)
        prim.rel("physics:body0", [self.plan.link_path(joint.parent)])
        prim.rel("physics:body1", [self.plan.link_path(joint.child)])

        if spherical:
            # No limits, so the frames only need to coincide at the export pose (like free D6 rotations).
            frame, motion = self._joint_frame(joint), Transform.identity()
            prim.attr("physics:axis", "uniform token", "X")
            if any(d.limit is not None and d.limit.is_bounded for d in joint.dofs):
                self.report.warn(f"joint {joint.name!r}: ball-joint limits are not exported (a spherical joint "
                                 "only has a cone limit)")
        elif kind is JointKind.MULTI:
            frame, motion = self._write_d6(prim, joint)
        else:
            frame = self._joint_frame(joint)
            motion = _dof_motion(joint.dofs[0].type, Vec3(1.0, 0.0, 0.0), joint.dofs[0].value) \
                if joint.dofs else Transform.identity()
        if native_multi:
            self._check_passive_multi(joint)
        _pose_attrs(prim, 0, parent.transform.inverse() * frame)
        _pose_attrs(prim, 1, child.transform.inverse() * frame * motion)

        excluded = joint.exclude_from_articulation or (kind is JointKind.MULTI and not native_multi)
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

    def _write_sensor(self, sensor: Sensor) -> None:
        link = self.model.links[sensor.link]
        link_prim = self.stage.get(self.plan.link_path(sensor.link))
        local = link.transform.inverse() * sensor.transform
        data: Dict[str, Any] = {"sensor": sensor.kind.value, **sensor.metadata}
        if sensor.source:
            data["source"] = sensor.source

        if sensor.kind is SensorKind.CONTACT:
            self.engine.contact_sensor(link_prim, float(sensor.params.get("force_threshold") or 0.0))
            link_prim.custom_data.setdefault("contact_sensors", []).append(sensor.name)
            return
        if sensor.kind is SensorKind.CAMERA:
            p = sensor.params
            prim = link_prim.define(sensor.name, "Camera")
            _xform_attrs(prim, local * ROS_TO_USD_CAMERA)
            prim.attr("projection", "token", "perspective")
            prim.attr("focalLength", "float", p["focal_length"])
            prim.attr("horizontalAperture", "float", p["horizontal_aperture"])
            prim.attr("verticalAperture", "float", p["vertical_aperture"])
            prim.attr("clippingRange", "float2", tuple(p["clipping_range"]))
            data.update(width=p["width"], height=p["height"], data_types=list(p["data_types"]))
        else:  # IMU and named frames are plain transform markers
            prim = link_prim.define(sensor.name, "Xform")
            _xform_attrs(prim, local)
        if "update_period" in sensor.params:
            data["update_period"] = sensor.params["update_period"]
        prim.custom_data.update({"fusion2usd": data})

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
        lo, hi = _limit_pair(dof.limit.lower if dof.limit else None, dof.limit.upper if dof.limit else None, scale)
        if (angular and self.engine.mimic_needs_finite_limits and joint.name in self.plan.mimic_joints
                and not (math.isfinite(lo) and math.isfinite(hi))):
            lo, hi = max(lo, -MIMIC_LIMIT_DEG), min(hi, MIMIC_LIMIT_DEG)
            self.report.warn(f"joint {joint.name!r} is part of a mimic coupling; PhysX needs finite limits, "
                             f"so it was limited to ±{MIMIC_LIMIT_DEG:g}°")
        if dof.limit is not None and dof.limit.is_bounded or (math.isfinite(lo) and math.isfinite(hi)):
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

        # Also on loop-closure joints: solvers ignore it there, but it records the position the limits refer to.
        _joint_state(prim, dof_name, dof.value * scale)
        self.engine.joint_dynamics(prim, props.armature, props.friction,
                                   None if props.velocity is None else props.velocity * scale)

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
        rotations = [d for d in dofs if d.type is DofType.ROTATION]
        ball = len(rotations) == 3
        if ball:
            # Rotation is entirely free, its axes need not be orthogonal away from zero.
            for i, axis in enumerate(_AXES):
                free["rot" + axis] = (rotations[i], 1.0)
            dofs = [d for d in dofs if d.type is not DofType.ROTATION]
        for dof in dofs:
            a = dof.axis.normalized()
            dots = [a.dot(e) for e in frame_axes]
            i = max(range(3), key=lambda k: abs(dots[k]))
            token = ("rot" if dof.type is DofType.ROTATION else "trans") + _AXES[i]
            off = math.degrees(math.acos(min(1.0, abs(dots[i]))))
            if abs(dots[i]) < _SNAP_TOL:
                self.report.warn(f"joint {joint.name!r}: DOF axis {a.as_tuple()} is {off:.1f}° off the joint "
                                 "frame; DOF skipped")
                continue
            if token in free:
                self.report.warn(f"joint {joint.name!r}: duplicate DOF {token}, skipped")
                continue
            if abs(dots[i]) < math.cos(_ALIGN_EPS):
                self.report.warn(f"joint {joint.name!r}: DOF axis is {off:.2f}° off the joint frame's "
                                 f"{_AXES[i]} axis; snapped to it")
            free[token] = (dof, 1.0 if dots[i] > 0 else -1.0)

        rotation, translation = Quat(), Vec3()
        for kind in ("trans", "rot"):
            for i, axis in enumerate(_AXES):
                token = kind + axis
                if token not in free:
                    prim.apply(f"PhysicsLimitAPI:{token}")
                    prim.attr(f"limit:{token}:physics:low", "float", 1.0)  # low > high locks the axis
                    prim.attr(f"limit:{token}:physics:high", "float", -1.0)
                    continue
                dof, sign = free[token]
                if ball:  # free rotations need no encoding: the frames coincide at the export pose
                    if kind == "rot":
                        if dof.limit is not None and dof.limit.is_bounded:
                            self.report.warn(f"joint {joint.name!r}: rotational limits on {token} are not "
                                             "exported for ball joints")
                        continue
                # Encode the value (translations first, then the single rotation), so limits keep the CAD
                # convention.
                local = Vec3(*(1.0 if k == i else 0.0 for k in range(3)))
                value = sign * dof.value
                scale = DEG if kind == "rot" else 1.0
                if kind == "rot":
                    rotation = Quat.from_axis_angle(local, value)
                else:
                    translation = translation + local * value
                _joint_state(prim, token, value * scale)
                bounded = dof.limit is not None and dof.limit.is_bounded
                if bounded and kind == "rot" and not self.engine.d6_rotation_limits:
                    self.report.warn(f"joint {joint.name!r}: rotational limits on {token} are not exported for "
                                     "multi-DOF joints")
                    bounded = False
                if bounded:
                    limit = dof.limit if sign > 0 else dof.limit.reversed()
                    low, high = _limit_pair(limit.lower, limit.upper, scale)
                elif self.engine.d6_free_axes_need_limits:
                    low, high = -INF, INF
                else:
                    continue
                prim.apply(f"PhysicsLimitAPI:{token}")
                prim.attr(f"limit:{token}:physics:low", "float", low)
                prim.attr(f"limit:{token}:physics:high", "float", high)
        return frame, Transform(rotation, translation)

    def _check_passive_multi(self, joint: Joint) -> None:
        """Native multi-DOF joints are written without drives or per-joint dynamics; say what is dropped."""
        props = joint.properties
        if props.actuated is None and self.options.actuate_by_default:
            self.report.info(f"Joint {joint.name!r} is a passive multi-DOF joint; mark it actuated to drive it")
        dropped = [f for f in ("effort", "velocity", "stiffness", "damping", "armature", "friction", "target")
                   if getattr(props, f) is not None]
        if dropped:
            self.report.warn(f"joint {joint.name!r}: {', '.join(dropped)} not exported for multi-DOF joints")

    def _write_mimic(self, coupling: Coupling) -> None:
        follower = self.model.joint(coupling.follower.joint)
        leader = self.model.joint(coupling.leader.joint)
        prim = self.stage.get(self.plan.joint_path(follower.name))
        if prim is None:
            return
        f_scale = DEG if follower.kind is JointKind.REVOLUTE else 1.0
        l_scale = DEG if leader.kind is JointKind.REVOLUTE else 1.0
        self.engine.mimic(prim, self.plan.joint_path(leader.name), coupling, f_scale, l_scale)

    def _newton_notes(self) -> None:
        """Newton behaviors (verified with Newton 1.7) that change how the export has to be loaded."""
        if self.plan.mimics:
            self.report.warn("Newton's Kamino solver ignores mimic joints; motion links only couple with the "
                             "MuJoCo, Featherstone or XPBD solvers")
        if self.options.zero_pose is ZeroPose.FUSION and any(d.value for j in self.model.joints for d in j.dofs):
            if any(j.exclude_from_articulation for j in self.model.joints):
                self.report.warn("Newton's MuJoCo solver anchors loop closures at joint zero, so with Joint zero = "
                                 "Fusion they drift open; export with Joint zero = Current pose")
            self.report.warn("Newton's add_usd ignores joint positions (JointStateAPI) unless SchemaResolverPhysx "
                             "is passed, so joints load at Fusion's joint zero, and the Isaac Lab cfg's init_state "
                             "only covers single-DOF joints; choose Joint zero = Current pose")


def _joint_state(prim: Prim, dof_name: str, position: float) -> None:
    prim.apply(f"PhysicsJointStateAPI:{dof_name}")
    prim.attr(f"state:{dof_name}:physics:position", "float", position)
    prim.attr(f"state:{dof_name}:physics:velocity", "float", 0.0)
