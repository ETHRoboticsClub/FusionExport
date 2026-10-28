import ast
import math
import xml.etree.ElementTree as ET

import pytest

from _mechanisms import gripper
from fusion2usd.core.math3d import Quat, Transform, Vec3
from fusion2usd.core.report import ExportReport
from fusion2usd.isaaclab.cfg_writer import IsaacLabCfgWriter
from fusion2usd.kinematics import PassContext, default_pipeline
from fusion2usd.kinematics.sensors import camera_horizontal_fov
from fusion2usd.model.robot import Sensor, SensorKind
from fusion2usd.options import ExportOptions, PhysicsEngine
from fusion2usd.serialization import ModelSerializer
from fusion2usd.urdf.writer import UrdfWriter
from fusion2usd.usd.writer import UsdWriter

pxr = pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom  # noqa: E402

# Camera on the palm at (0.2, 0, 0.1), optical axis (+Z, ROS convention) along world +X, image "down" (+Y) = world -Z.
CAM_POSE = Transform.from_axes(Vec3(0.2, 0.0, 0.1), Vec3(0, -1, 0), Vec3(0, 0, -1))  # x=-Y, y=-Z -> z = +X


def with_sensors(y_up: bool = False):
    m = gripper()
    if y_up:
        m.metadata["source_up_axis"] = "Y"
    m.sensors = [
        Sensor("Front Cam", SensorKind.CAMERA, "palm", CAM_POSE,
               metadata={"attributes": {"sensor": "camera", "width": "640", "height": "480", "fov": "90",
                                        "rate": "30 Hz", "data_types": "rgb,depth", "mount": "bracket A"}}),
        Sensor("imu", SensorKind.IMU, "finger_l", Transform(translation=Vec3(-0.1, 0.0, 0.02)),
               metadata={"attributes": {"sensor": "imu", "rate": "200"}}),
        Sensor("tip contact", SensorKind.CONTACT, "finger_r", Transform(),
               metadata={"attributes": {"sensor": "contact", "history_length": "3", "force_threshold": "0.5 N"}}),
        Sensor("tool tip", SensorKind.FRAME, "slider", Transform(translation=Vec3(0.0, 0.4, 0.0)),
               metadata={"attributes": {"sensor": "frame"}}),
    ]
    return m


def process(model):
    opts = ExportOptions(physics_engine=PhysicsEngine.PHYSX)
    report = ExportReport()
    return default_pipeline().run(model, PassContext(opts, report)), opts, report


def test_camera_params_parsed():
    model, _, report = process(with_sensors())
    cam = next(s for s in model.sensors if s.kind is SensorKind.CAMERA)
    assert cam.name == "Front_Cam"
    assert math.degrees(camera_horizontal_fov(cam)) == pytest.approx(90.0)
    assert cam.params["vertical_aperture"] / cam.params["horizontal_aperture"] == pytest.approx(480 / 640)
    assert cam.params["update_period"] == pytest.approx(1 / 30)
    assert cam.params["data_types"] == ["rgb", "depth"]
    assert cam.metadata["user"] == {"mount": "bracket A"}
    contact = next(s for s in model.sensors if s.kind is SensorKind.CONTACT)
    assert contact.params == {"history_length": 3, "force_threshold": 0.5}
    assert not report.warnings or all("default values" in w or "continuous" in w for w in report.warnings)


@pytest.mark.parametrize("y_up", [False, True])
def test_usd_camera_looks_along_cad_z(tmp_path, y_up):
    model, opts, report = process(with_sensors(y_up))
    path = str(tmp_path / "g.usda")
    UsdWriter(opts, report).write(model, path)
    stage = Usd.Stage.Open(path)
    cam_prim = stage.GetPrimAtPath("/gripper/palm/Front_Cam")
    assert cam_prim.IsA(UsdGeom.Camera)
    world = UsdGeom.Xformable(cam_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    forward = world.TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized()   # USD camera looks down -Z
    up = world.TransformDir(Gf.Vec3d(0, 1, 0)).GetNormalized()
    # Expected world directions after the exporter's frame changes (up-axis conversion + base re-centering).
    cam = next(s for s in model.sensors if s.kind is SensorKind.CAMERA)
    exp_forward, exp_down = cam.transform.axis(2), cam.transform.axis(1)
    if not y_up:
        assert exp_forward.is_close(Vec3(1, 0, 0), 1e-9)
    assert list(forward) == pytest.approx(list(exp_forward), abs=1e-6)
    assert list(up) == pytest.approx(list(-exp_down), abs=1e-6)
    if y_up:  # CAD +X stays +X under the Y-up -> Z-up rotation about X
        assert list(forward) == pytest.approx([1, 0, 0], abs=1e-6)
    camera = UsdGeom.Camera(cam_prim)
    assert camera.GetFocalLengthAttr().Get() == pytest.approx(24.0)
    assert camera.GetClippingRangeAttr().Get()[0] == pytest.approx(0.01)
    assert stage.GetPrimAtPath("/gripper/finger_l/imu").IsA(UsdGeom.Xform)
    # usd-core has no PhysX schemas registered, so check the authored list rather than GetAppliedSchemas()
    authored = stage.GetPrimAtPath("/gripper/finger_r").GetMetadata("apiSchemas").GetAddedOrExplicitItems()
    assert "PhysxContactReportAPI" in authored


def test_isaaclab_sensors_factory(tmp_path):
    model, opts, report = process(with_sensors())
    src = IsaacLabCfgWriter(opts, report).render(model, str(tmp_path / "g.usda"), str(tmp_path / "g_cfg.py"))
    tree = ast.parse(src)
    assert any(isinstance(n, ast.FunctionDef) and n.name == "sensors" for n in tree.body)
    assert "activate_contact_sensors=True" in src
    assert 'CameraCfg(prim_path=f"{robot_prim_path}/palm/Front_Cam", spawn=None, width=640, height=480' in src
    # IMU offset: identity rotation in XYZW order is (0, 0, 0, 1)
    imu_line = next(l for l in src.splitlines() if "ImuCfg(prim_path" in l)
    assert "rot=(0, 0, 0, 1)" in imu_line.replace(".0", "")
    assert "FRAMES = {" in src and "'tool_tip': ('slider'," in src


def test_urdf_sensor_frames_and_serialization(tmp_path):
    model, opts, report = process(with_sensors())
    renamed = {"imu": "imu_link", "tool_tip": "slider"}  # slider is also a link: that frame gets _frame
    for sensor in model.sensors:
        sensor.name = renamed.get(sensor.name, sensor.name)
    UrdfWriter(opts, report).write(model, str(tmp_path / "g.urdf"))
    root = ET.parse(tmp_path / "g.urdf").getroot()
    names = {l.get("name") for l in root.findall("link")}
    assert {"Front_Cam", "imu_link", "slider_frame"} <= names and "tip_contact" not in names
    mounts = {j.get("name"): j.find("child").get("link") for j in root.findall("joint")}
    assert mounts["imu_joint"] == "imu_link" and mounts["Front_Cam_joint"] == "Front_Cam"
    assert mounts["slider_frame_joint"] == "slider_frame"

    raw = with_sensors()
    path = str(tmp_path / "m.json")
    ModelSerializer().save(raw, path)
    back = ModelSerializer().load(path)
    assert [(s.name, s.kind, s.link) for s in back.sensors] == [(s.name, s.kind, s.link) for s in raw.sensors]
    assert back.sensors[0].transform.is_close(CAM_POSE)
