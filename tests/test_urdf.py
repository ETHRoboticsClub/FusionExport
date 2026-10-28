import math
import xml.etree.ElementTree as ET

import pytest

from _mechanisms import four_bar, gripper
from fusion2usd.core.math3d import Quat, Transform, Vec3
from fusion2usd.core.report import ExportReport
from fusion2usd.kinematics import PassContext, default_pipeline
from fusion2usd.options import ExportOptions, ZeroPose
from fusion2usd.urdf.writer import UrdfWriter, rpy_from_quat


def quat_from_rpy(r, p, y):
    x, yv, z = Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)
    return Quat.from_axis_angle(z, y) * Quat.from_axis_angle(yv, p) * Quat.from_axis_angle(x, r)


def origin_tf(el):
    o = el.find("origin")
    xyz = [float(v) for v in o.get("xyz").split()]
    rpy = [float(v) for v in o.get("rpy").split()]
    return Transform(quat_from_rpy(*rpy), Vec3(*xyz))


def urdf_fk(root, base, base_tf, q):
    frames = {base: base_tf}
    pending = [j for j in root.findall("joint") if j.get("name") != "root_joint"]
    for _ in range(len(pending) + 1):
        for j in list(pending):
            parent = j.find("parent").get("link")
            if parent not in frames:
                continue
            child = j.find("child").get("link")
            tf = frames[parent] * origin_tf(j)
            if j.get("type") in ("revolute", "continuous", "prismatic"):
                axis = Vec3(*[float(v) for v in j.find("axis").get("xyz").split()])
                val = q[j.get("name")]
                tf = tf * (Transform(Quat.from_axis_angle(axis, val)) if j.get("type") != "prismatic"
                           else Transform(translation=axis * val))
            frames[child] = tf
            pending.remove(j)
    assert not pending, "URDF is not a tree rooted at the base"
    return frames


@pytest.mark.parametrize("uri,prefix", [("", "robot_meshes/"),
                                        ("package://pkg/urdf/", "package://pkg/urdf/robot_meshes/")])
def test_mesh_paths_follow_the_uri_option(tmp_path, uri, prefix):
    opts = ExportOptions(urdf_mesh_uri=uri)
    report = ExportReport()
    model = default_pipeline().run(gripper(), PassContext(opts, report))
    UrdfWriter(opts, report).write(model, str(tmp_path / "robot.urdf"))
    files = [m.get("filename") for m in ET.parse(tmp_path / "robot.urdf").getroot().iter("mesh")]
    assert files and all(f.startswith(prefix) for f in files)
    assert all((tmp_path / "robot_meshes" / f.rsplit("/", 1)[1]).exists() for f in files)


def test_rpy_round_trip():
    q = quat_from_rpy(0.3, -0.4, 1.2)
    assert quat_from_rpy(*rpy_from_quat(q)).angle_to(q) < 1e-6


@pytest.mark.parametrize("factory", [four_bar, gripper])
@pytest.mark.parametrize("zero", [ZeroPose.FUSION, ZeroPose.CURRENT])
def test_urdf_reproduces_export_pose(tmp_path, factory, zero):
    opts = ExportOptions(zero_pose=zero, fixed_base=None)
    report = ExportReport()
    model = default_pipeline().run(factory(), PassContext(opts, report))
    path = tmp_path / "robot.urdf"
    written = UrdfWriter(opts, report).write(model, str(path))
    assert all((tmp_path / p).exists() for p in written)
    root = ET.parse(path).getroot()

    q = {j.name: j.dofs[0].value for j in model.joints if len(j.dofs) == 1}
    base = model.links[model.base_link]
    frames = urdf_fk(root, model.base_link, base.transform, q)
    # URDF link frames sit on the incoming joint origin; compare the origin positions of model links
    # by mapping the model link frame through the URDF frame (inertial origin is the COM in URDF frame).
    for link_el in root.findall("link"):
        name = link_el.get("name")
        if name == "world":
            continue
        inertial = link_el.find("inertial/origin")
        com_urdf = Vec3(*[float(v) for v in inertial.get("xyz").split()])
        model_link = model.links[name]
        expected = model_link.transform.apply_point(model_link.mass.com)
        assert frames[name].apply_point(com_urdf).is_close(expected, 1e-6), name

    if factory is four_bar:
        assert len(root.findall("joint")) == 1 + 3  # root + tree joints; loop joint dropped
        assert any("closes a loop" in w for w in report.warnings)
    else:
        mimic = root.find("joint[@name='Revolute_2']/mimic")
        assert mimic is not None and float(mimic.get("multiplier")) == pytest.approx(1.0)
