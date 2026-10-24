"""URDF output derived from the processed RobotModel (the one the USD was written from).

URDF is a strict tree whose child-link frame coincides with the joint frame at q = 0, so each
link gets a URDF frame at its incoming joint's origin and all geometry is re-expressed in it.
Loop-closure joints, D6 joints and free bodies cannot be represented and are left out (with warnings).
"""

from __future__ import annotations

import math
import os
import struct
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

from ..core.math3d import Quat, Transform, Vec3
from ..core.report import ExportReport
from ..model.robot import Joint, JointKind, Link, Mesh, RobotModel, Sensor, SensorKind
from ..options import ExportOptions

_FALLBACK_EFFORT = 1000.0
_FALLBACK_VELOCITY = 100.0
_UNBOUNDED = 1e3  # stands in for a missing side of a limit (URDF needs both)


def _fmt(x: float) -> str:
    return f"{x:.9g}"


def _xyz(v: Vec3) -> str:
    return " ".join(_fmt(c) for c in v)


def rpy_from_quat(q: Quat) -> Tuple[float, float, float]:
    """URDF fixed-axis roll/pitch/yaw: R = Rz(yaw) · Ry(pitch) · Rx(roll)."""
    r = q.to_matrix()
    pitch = math.asin(max(-1.0, min(1.0, -r[2][0])))
    if abs(r[2][0]) < 1.0 - 1e-9:
        roll = math.atan2(r[2][1], r[2][2])
        yaw = math.atan2(r[1][0], r[0][0])
    else:  # gimbal lock: fold everything into yaw
        roll = 0.0
        yaw = math.atan2(-r[0][1], r[1][1])
    return roll, pitch, yaw


def _origin(parent: ET.Element, t: Transform) -> None:
    ET.SubElement(parent, "origin", xyz=_xyz(t.translation), rpy=" ".join(_fmt(a) for a in rpy_from_quat(t.rotation)))


def _motion(joint: Joint, frame: Transform, value: float) -> Transform:
    """Joint motion by ``value`` expressed in ``frame``."""
    dof = joint.dofs[0]
    axis = frame.rotation.conjugate().rotate(dof.axis)
    if joint.kind is JointKind.REVOLUTE:
        return Transform(Quat.from_axis_angle(axis, value))
    return Transform(translation=axis * value)


class StlWriter:
    @staticmethod
    def write(path: str, mesh: Mesh, transform: Transform) -> None:
        pts = [transform.apply_point(Vec3(*mesh.points[i:i + 3])) for i in range(0, len(mesh.points), 3)]
        tris = [mesh.indices[i:i + 3] for i in range(0, len(mesh.indices), 3)]
        with open(path, "wb") as f:
            f.write(b"fusion2usd".ljust(80, b"\0"))
            f.write(struct.pack("<I", len(tris)))
            for a, b, c in tris:
                pa, pb, pc = pts[a], pts[b], pts[c]
                n = (pb - pa).cross(pc - pa)
                n = n / n.norm() if n.norm() > 0 else Vec3()
                f.write(struct.pack("<12fH", *n, *pa, *pb, *pc, 0))


class UrdfWriter:
    def __init__(self, options: ExportOptions, report: ExportReport) -> None:
        self.options = options
        self.report = report

    def write(self, model: RobotModel, path: str) -> List[str]:
        mesh_dir_name = os.path.splitext(os.path.basename(path))[0] + "_meshes"
        mesh_dir = os.path.join(os.path.dirname(os.path.abspath(path)), mesh_dir_name)
        os.makedirs(mesh_dir, exist_ok=True)
        written: List[str] = []

        joints = self._tree_joints(model)
        frames = self._urdf_frames(model, joints)
        robot = ET.Element("robot", name=model.name)

        for mat in model.materials.values():
            m = ET.SubElement(robot, "material", name=mat.name)
            ET.SubElement(m, "color", rgba=" ".join(_fmt(c) for c in (*mat.color, mat.opacity)))

        if model.fixed_base:
            ET.SubElement(robot, "link", name="world")
            j = ET.SubElement(robot, "joint", name="root_joint", type="fixed")
            ET.SubElement(j, "parent", link="world")
            ET.SubElement(j, "child", link=model.base_link)
            _origin(j, frames[model.base_link])

        for name in frames:
            written += self._write_link(robot, model.links[name], frames[name], mesh_dir, mesh_dir_name)
        for joint in joints:
            self._write_joint(robot, model, joint, frames)
        for sensor in model.sensors:
            self._write_sensor_frame(robot, sensor, frames)

        ET.indent(robot)
        ET.ElementTree(robot).write(path, encoding="utf-8", xml_declaration=True)
        self.report.info(f"URDF written with {len(frames)} links and {len(joints)} joints")
        return [path] + written

    # ------------------------------------------------------------------ structure

    def _tree_joints(self, model: RobotModel) -> List[Joint]:
        joints = []
        for j in model.joints:
            if j.exclude_from_articulation:
                self.report.warn(f"URDF: joint {j.name!r} closes a loop (or joins free bodies); URDF cannot represent it")
            elif j.kind is JointKind.MULTI:
                self.report.warn(f"URDF: multi-DOF joint {j.name!r} skipped")
            else:
                joints.append(j)
        return joints

    def _urdf_frames(self, model: RobotModel, joints: List[Joint]) -> Dict[str, Transform]:
        """World pose (at the exported configuration) of every reachable link's URDF frame."""
        frames: Dict[str, Transform] = {model.base_link: model.links[model.base_link].transform}  # type: ignore[dict-item]
        by_parent: Dict[str, List[Joint]] = {}
        for j in joints:
            by_parent.setdefault(j.parent, []).append(j)
        queue = [model.base_link]
        while queue:
            cur = queue.pop(0)
            for j in by_parent.get(cur, []):
                frames[j.child] = Transform(model.links[j.child].transform.rotation, j.origin)
                queue.append(j.child)
        for name in model.links:
            if name not in frames:
                self.report.warn(f"URDF: link {name!r} is not part of the tree; skipped")
        return frames

    # ------------------------------------------------------------------ elements

    def _write_link(self, robot: ET.Element, link: Link, frame: Transform, mesh_dir: str, mesh_rel: str) -> List[str]:
        el = ET.SubElement(robot, "link", name=link.name)
        to_urdf = frame.inverse() * link.transform  # model link frame -> URDF link frame
        if link.mass is not None:
            mp = link.mass.transformed(to_urdf)
            inertial = ET.SubElement(el, "inertial")
            ET.SubElement(inertial, "origin", xyz=_xyz(mp.com), rpy="0 0 0")
            ET.SubElement(inertial, "mass", value=_fmt(mp.mass))
            i = mp.inertia
            ET.SubElement(inertial, "inertia", ixx=_fmt(i[0][0]), ixy=_fmt(i[0][1]), ixz=_fmt(i[0][2]),
                          iyy=_fmt(i[1][1]), iyz=_fmt(i[1][2]), izz=_fmt(i[2][2]))

        written = []
        collisions = link.collisions or link.visuals
        for kind, meshes in (("visual", link.visuals), ("collision", collisions)):
            if kind == "collision" and not self.options.export_collisions:
                continue
            for mesh in meshes:
                if mesh.triangle_count == 0:
                    continue
                file_name = f"{link.name}_{kind}_{mesh.name}.stl"
                path = os.path.join(mesh_dir, file_name)
                StlWriter.write(path, mesh, to_urdf)
                written.append(path)
                shape = ET.SubElement(el, kind, name=mesh.name)
                geom = ET.SubElement(shape, "geometry")
                ET.SubElement(geom, "mesh", filename=f"{mesh_rel}/{file_name}")
                if kind == "visual" and mesh.material:
                    ET.SubElement(shape, "material", name=mesh.material)
        return written

    def _write_joint(self, robot: ET.Element, model: RobotModel, joint: Joint, frames: Dict[str, Transform]) -> None:
        parent_frame, child_frame = frames[joint.parent], frames[joint.child]
        kind = joint.kind
        if kind is JointKind.FIXED:
            el = ET.SubElement(robot, "joint", name=joint.name, type="fixed")
            ET.SubElement(el, "parent", link=joint.parent)
            ET.SubElement(el, "child", link=joint.child)
            _origin(el, parent_frame.inverse() * child_frame)
            return

        dof = joint.dofs[0]
        lim = dof.limit
        if kind is JointKind.REVOLUTE:
            urdf_type = "revolute" if lim is not None and lim.is_bounded else "continuous"
        else:
            urdf_type = "prismatic"
        el = ET.SubElement(robot, "joint", name=joint.name, type=urdf_type)
        ET.SubElement(el, "parent", link=joint.parent)
        ET.SubElement(el, "child", link=joint.child)
        # child(q) = parent * origin * M(q); at q = value it must equal the exported child frame.
        _origin(el, parent_frame.inverse() * child_frame * _motion(joint, child_frame, dof.value).inverse())
        ET.SubElement(el, "axis", xyz=_xyz(child_frame.rotation.conjugate().rotate(dof.axis).normalized()))

        props = joint.properties
        effort = props.effort if props.effort is not None else (self.options.drive.max_force or _FALLBACK_EFFORT)
        velocity = props.velocity if props.velocity is not None else _FALLBACK_VELOCITY
        limit_attrs = {"effort": _fmt(effort), "velocity": _fmt(velocity)}
        if urdf_type != "continuous":
            lower = lim.lower if lim and lim.lower is not None else -_UNBOUNDED
            upper = lim.upper if lim and lim.upper is not None else _UNBOUNDED
            limit_attrs.update(lower=_fmt(lower), upper=_fmt(upper))
        ET.SubElement(el, "limit", **limit_attrs)
        if props.friction is not None:
            ET.SubElement(el, "dynamics", friction=_fmt(props.friction))

        coupling = self._coupling_for(model, joint.name)
        if coupling is not None:
            ET.SubElement(el, "mimic", joint=coupling.leader.joint, multiplier=_fmt(coupling.ratio),
                          offset=_fmt(coupling.offset))

    @staticmethod
    def _write_sensor_frame(robot: ET.Element, sensor: Sensor, frames: Dict[str, Transform]) -> None:
        """Cameras, IMUs and named frames become massless fixed child links (the usual ROS pattern)."""
        if sensor.kind is SensorKind.CONTACT or sensor.link not in frames:
            return
        name = f"{sensor.name}_frame"
        ET.SubElement(robot, "link", name=name)
        joint = ET.SubElement(robot, "joint", name=f"{name}_joint", type="fixed")
        ET.SubElement(joint, "parent", link=sensor.link)
        ET.SubElement(joint, "child", link=name)
        _origin(joint, frames[sensor.link].inverse() * sensor.transform)

    def _coupling_for(self, model: RobotModel, joint_name: str):
        if not self.options.export_mimic_joints:
            return None
        return next((c for c in model.couplings if c.follower.joint == joint_name), None)


__all__ = ["UrdfWriter", "StlWriter", "rpy_from_quat"]
