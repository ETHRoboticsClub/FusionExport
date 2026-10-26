"""Import every Fusion-facing module against a stub ``adsk`` and run the reader on a fake design."""

import importlib
import json
import math
import pkgutil
from types import SimpleNamespace

import pytest

import _adsk_stub
from fusion2usd.core.math3d import Quat, Transform, Vec3
from fusion2usd.core.report import ExportReport
from fusion2usd.options import ExportOptions


@pytest.fixture
def adsk():
    mods = _adsk_stub.install()
    yield mods
    _adsk_stub.uninstall()


def test_all_fusion_and_ui_modules_import(adsk):
    import fusion2usd.fusion
    import fusion2usd.ui

    for pkg in (fusion2usd.fusion, fusion2usd.ui):
        for info in pkgutil.iter_modules(pkg.__path__):
            importlib.import_module(f"{pkg.__name__}.{info.name}")


# ---------------------------------------------------------------- fakes


class Coll(list):
    @property
    def count(self):
        return len(self)

    def item(self, i):
        return self[i]


def pt(v: Vec3, scale=1.0):
    return SimpleNamespace(x=v.x * scale, y=v.y * scale, z=v.z * scale)


class Matrix:
    def __init__(self, t: Transform):
        self.t = t

    def getAsCoordinateSystem(self):
        return (pt(self.t.translation, 100.0), pt(self.t.axis(0)), pt(self.t.axis(1)), pt(self.t.axis(2)))


class Attributes:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def itemsByGroup(self, group):
        return [SimpleNamespace(name=k, value=v) for k, v in self.values.items()] if group == "fusion2usd" else []


def rotation_about(axis, point, angle):
    return Transform(translation=point) * Transform(Quat.from_axis_angle(axis, angle)) * Transform(translation=-point)


class Occurrence:
    def __init__(self, name, pose_fn, bodies=(), grounded=False):
        self.name = name
        self.fullPathName = name
        self._pose = pose_fn
        self.isGrounded = grounded
        self.isGroundToParent = False
        self.childOccurrences = Coll()
        self.attributes = Attributes()
        self.nativeObject = None
        self.appearance = None
        self.component = SimpleNamespace(name=name.split(":")[0], partNumber="PN-" + name, description="",
                                         material=SimpleNamespace(name="Steel"), motionLinks=Coll())
        self.bRepBodies = Coll()
        for b in bodies:
            b.assemblyContext = self
            self.bRepBodies.append(b)

    @property
    def transform2(self):
        return Matrix(self._pose())


class ColorProperty:
    def __init__(self, r, g, b, a):
        self.id = "generic_diffuse"
        self.value = SimpleNamespace(red=r, green=g, blue=b, opacity=a)

    @staticmethod
    def cast(obj):
        return obj if isinstance(obj, ColorProperty) else None


class FloatProperty:
    @staticmethod
    def cast(obj):
        return None


class Body:
    """A unit-ish box given in world coordinates (cm)."""

    def __init__(self, name, lo: Vec3, hi: Vec3, mass, com: Vec3):
        self.name = name
        self.isVisible = True
        self.lo, self.hi = lo, hi
        self.mass, self.com = mass, com
        self.appearance = SimpleNamespace(id="app1", name="Red", appearanceProperties=Coll([ColorProperty(255, 0, 0, 255)]))
        self.boundingBox = SimpleNamespace(minPoint=pt(lo, 100.0), maxPoint=pt(hi, 100.0))
        self.meshManager = SimpleNamespace(createMeshCalculator=lambda: self)

    def setQuality(self, q):
        self.quality = q

    def calculate(self):
        lo, hi = self.lo * 100.0, self.hi * 100.0
        pts = [lo.x, lo.y, lo.z, hi.x, lo.y, lo.z, hi.x, hi.y, lo.z, lo.x, lo.y, hi.z]
        return SimpleNamespace(nodeCount=4, nodeCoordinatesAsDouble=pts, nodeIndices=[0, 1, 2, 0, 2, 3],
                               normalVectorsAsDouble=[0.0, 0.0, 1.0] * 4)

    def getPhysicalProperties(self, accuracy):
        return Phys(self.mass, self.com)


class Phys:
    def __init__(self, mass, com: Vec3):
        self.mass, self.centerOfMass = mass, pt(com, 100.0)

    def getPrincipalMomentsOfInertia(self):
        return True, 10.0, 20.0, 30.0  # kg·cm²

    def getPrincipalAxes(self):
        return True, pt(Vec3(1, 0, 0)), pt(Vec3(0, 1, 0)), pt(Vec3(0, 0, 1))


class Limits:
    def __init__(self, lo=None, hi=None):
        self.isMinimumValueEnabled, self.minimumValue = lo is not None, lo or 0.0
        self.isMaximumValueEnabled, self.maximumValue = hi is not None, hi or 0.0


class RevoluteMotion:
    def __init__(self, jt, axis, value=0.0, limits=None):
        self.jointType = jt
        self.rotationAxisVector = pt(axis)
        self.rotationAxis = None
        self.rotationValue = value
        self.rotationLimits = limits or Limits()


class FakeJoint:
    def __init__(self, name, one, two, motion, frame: Transform):
        self.name = name
        self.occurrenceOne, self.occurrenceTwo = one, two
        self.jointMotion = motion
        self.geometryTwoTransform = Matrix(frame)
        self.isSuppressed = False
        self.assemblyContext = None
        self.nativeObject = None
        self.attributes = Attributes({"effort": "5", "json": json.dumps({"note": "hi"})})


# ---------------------------------------------------------------- scenario


AXIS1_POINT = Vec3(0.1, 0.0, 0.2)
AXIS2_POINT = Vec3(0.3, 0.0, 0.2)
BASE_POSE = Transform(translation=Vec3(0.0, 0.0, 0.1))
ARM_POSE = Transform(Quat.from_axis_angle(Vec3(0, 0, 1), 0.3), Vec3(0.2, 0.0, 0.2))
WHEEL_POSE = Transform(translation=Vec3(0.3, 0.0, 0.25))


def build_design(adsk):
    jt = adsk["adsk.fusion"].JointTypes
    m1 = RevoluteMotion(jt.RevoluteJointType, Vec3(0, 0, 1), value=0.1, limits=Limits(-1.0, 1.0))
    m2 = RevoluteMotion(jt.RevoluteJointType, Vec3(1, 0, 0), value=0.4)
    z, x = Vec3(0, 0, 1), Vec3(1, 0, 0)

    def arm_pose():  # Fusion rotates rev1 about -Z for increasing values (sign the API does not reveal)
        return rotation_about(z, AXIS1_POINT, -(m1.rotationValue - 0.1)) * ARM_POSE

    def wheel_pose():
        r1 = rotation_about(z, AXIS1_POINT, -(m1.rotationValue - 0.1))
        return r1 * rotation_about(x, AXIS2_POINT, m2.rotationValue - 0.4) * WHEEL_POSE

    base = Occurrence("base:1", lambda: BASE_POSE,
                      [Body("plate", Vec3(-0.1, -0.1, 0.05), Vec3(0.1, 0.1, 0.1), 2.0, Vec3(0, 0, 0.075))],
                      grounded=True)
    arm = Occurrence("arm:1", arm_pose, [Body("beam", Vec3(0.1, -0.02, 0.18), Vec3(0.3, 0.02, 0.22), 0.5,
                                              Vec3(0.2, 0, 0.2))])
    wheel = Occurrence("wheel:1", wheel_pose, [Body("disc", Vec3(0.29, -0.05, 0.2), Vec3(0.31, 0.05, 0.3), 0.1,
                                                    Vec3(0.3, 0, 0.25))])
    # API origin is 5 cm off the true axis; the probe must move it back onto the axis.
    rev1 = FakeJoint("rev1", arm, base, m1, Transform(translation=AXIS1_POINT + Vec3(0, 0.05, 0)))
    rev2 = FakeJoint("rev2", wheel, arm, m2, Transform(Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2), AXIS2_POINT))
    mt = adsk["adsk.fusion"].JointMotionTypes
    link = SimpleNamespace(name="ml", jointOne=rev1, jointTwo=rev2, isSuppressed=False, isReversed=True,
                           motionOne=mt.RevoluteJointRotateMotionType, motionTwo=mt.RevoluteJointRotateMotionType,
                           valueOne=SimpleNamespace(value=1.0), valueTwo=SimpleNamespace(value=2.0))
    root = SimpleNamespace(
        name="Robot", occurrences=Coll([base, arm, wheel]), bRepBodies=Coll(), allJoints=Coll([rev1, rev2]),
        allAsBuiltJoints=Coll(), allRigidGroups=Coll(), allJointOrigins=[], motionLinks=Coll([link]),
        attributes=Attributes(),
        material=None, partNumber="", description="",
    )
    design = SimpleNamespace(rootComponent=root, parentDocument=SimpleNamespace(name="Doc"))
    return design, (m1, m2)


def test_reader_on_fake_design(adsk):
    core = adsk["adsk.core"]
    core.ColorProperty, core.FloatProperty = ColorProperty, FloatProperty
    from fusion2usd.fusion.reader import FusionDesignReader

    design, (m1, m2) = build_design(adsk)
    report = ExportReport()
    options = ExportOptions(source_up_axis="Z")
    model = FusionDesignReader(design, options, report).read()

    assert model.name == "Doc"
    assert model.metadata["source_up_axis"] == "Z"
    assert sorted(model.links) == ["arm:1", "base:1", "wheel:1"]
    assert model.base_link == "base:1" and model.grounded_links == ["base:1"]
    base, arm = model.links["base:1"], model.links["arm:1"]
    assert base.transform.is_close(BASE_POSE) and arm.transform.is_close(ARM_POSE)

    # Mass: ObjectCollection is a stub, so the per-body fallback runs; result is in the link frame.
    assert arm.mass.mass == pytest.approx(0.5)
    assert arm.mass.com.is_close(ARM_POSE.inverse().apply_point(Vec3(0.2, 0, 0.2)), 1e-9)
    assert sorted(arm.mass.principal()[0]) == pytest.approx([1e-3, 2e-3, 3e-3])

    # Meshes are in the link frame (meters) and carry the appearance material.
    lo, hi = arm.visuals[0].transformed(arm.transform).bounds()
    assert lo.is_close(Vec3(0.1, -0.02, 0.18), 1e-9) and hi.is_close(Vec3(0.3, 0.02, 0.22), 1e-9)
    assert arm.visuals[0].material == "Red" and model.materials["Red"].color == (1.0, 0.0, 0.0)
    assert len(arm.collisions) == 1 and arm.collisions[0].material is None
    assert arm.metadata["part_number"] == "PN-arm:1" and arm.metadata["material"] == "Steel"

    j1, j2 = model.joint("rev1"), model.joint("rev2")
    assert (j1.parent, j1.child) == ("base:1", "arm:1")
    assert (j2.parent, j2.child) == ("arm:1", "wheel:1")
    assert j1.dofs[0].value == pytest.approx(0.1) and j1.dofs[0].limit.lower == pytest.approx(-1.0)
    # Probe fixed the sign and moved the origin onto the measured axis.
    assert j1.dofs[0].axis.is_close(Vec3(0, 0, -1), 1e-6)
    assert j1.origin.is_close(AXIS1_POINT, 1e-6)
    assert j2.dofs[0].axis.is_close(Vec3(1, 0, 0), 1e-6)
    assert j1.metadata["attributes"] == {"effort": "5", "note": "hi"}
    assert any("off the measured rotation axis" in w for w in report.warnings)
    # Joint values restored after probing.
    assert m1.rotationValue == 0.1 and m2.rotationValue == 0.4

    (c,) = model.couplings
    assert (c.leader.joint, c.follower.joint) == ("rev1", "rev2")
    assert c.ratio == pytest.approx(-2.0)
    assert c.offset == pytest.approx(0.4 - (-2.0) * 0.1)


def test_probe_keeps_api_axis_when_other_dofs_move_too(adsk):
    """In a closed loop Fusion re-solves the other DOFs, so the measured motion is not this DOF's alone."""
    from fusion2usd.fusion.joints import FusionJoint
    from fusion2usd.fusion.probe import JointProbe
    from fusion2usd.model.robot import Dof, DofType, Joint

    jt = adsk["adsk.fusion"].JointTypes
    motion = RevoluteMotion(jt.RevoluteJointType, Vec3(0, 0, 1))
    y = Vec3(0, 1, 0)
    arm = Occurrence("arm:1", lambda: rotation_about(y, AXIS1_POINT, 10 * motion.rotationValue))
    base = Occurrence("base:1", lambda: BASE_POSE)
    entity = FakeJoint("rev", arm, base, motion, Transform(translation=AXIS1_POINT))
    fj = FusionJoint(entity, "rev", "rev", "", False, jt.RevoluteJointType, arm, base, None)
    joint = Joint("rev", "base:1", "arm:1", AXIS1_POINT, [Dof(DofType.ROTATION, Vec3(0, 0, 1))])
    report = ExportReport()
    JointProbe(report).run([(fj, joint)])

    assert joint.dofs[0].axis.is_close(Vec3(0, 0, 1))
    assert any("for a commanded" in w and "using API axis" in w for w in report.warnings)
    assert motion.rotationValue == 0.0


def test_internal_joint_dropped_and_hidden_body_skipped(adsk):
    core = adsk["adsk.core"]
    core.ColorProperty, core.FloatProperty = ColorProperty, FloatProperty
    from fusion2usd.fusion.reader import FusionDesignReader

    design, _ = build_design(adsk)
    root = design.rootComponent
    base, arm, wheel = root.occurrences
    arm.isGrounded = True  # arm is now welded to the base: rev1 is internal
    wheel.bRepBodies[0].isVisible = False
    report = ExportReport()
    model = FusionDesignReader(design, ExportOptions(source_up_axis="Y", probe_joints=False), report).read()
    assert [j.name for j in model.joints] == ["rev2"]
    assert any("rev1" in w and "dropped" in w for w in report.warnings)
    assert model.links["wheel:1"].visuals == [] and model.links["wheel:1"].mass is None
    assert model.couplings == []  # leader joint not exported


def test_srgb_to_linear(adsk):
    from fusion2usd.fusion.geometry import srgb_to_linear

    assert srgb_to_linear(0.0) == 0.0 and srgb_to_linear(1.0) == pytest.approx(1.0)
    assert srgb_to_linear(0.5) == pytest.approx(0.21404, abs=1e-5)
    assert srgb_to_linear(0.04) == pytest.approx(0.04 / 12.92)


def test_slider_fallback_axis_is_frame_z(adsk):
    from fusion2usd.fusion.joints import DOF_SPECS, JT, JointBuilder

    frame = Transform(Quat.from_axis_angle(Vec3(1, 0, 0), math.pi / 2))  # frame Z -> world -Y
    motion = SimpleNamespace(slideDirection=None)
    (spec,) = DOF_SPECS[JT.SliderJointType]
    assert JointBuilder._fallback_axis(spec, motion, frame).is_close(Vec3(0, -1, 0), 1e-9)


# ---------------------------------------------------------------- sensors


class FakeJointOrigin:
    def __init__(self, name, context, frame: Transform, attributes=None, geometry_origin=None):
        self.name = name
        self.assemblyContext = context
        self.transform = Matrix(frame)
        self.attributes = Attributes(attributes)
        self.nativeObject = None
        origin = frame.translation if geometry_origin is None else geometry_origin
        self.geometry = SimpleNamespace(origin=pt(origin, 100.0))


CAMERA_FRAME = Transform(Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2), Vec3(0.25, 0.0, 0.22))  # Z -> +X


def _read_with_origins(adsk, origins):
    core = adsk["adsk.core"]
    core.ColorProperty, core.FloatProperty = ColorProperty, FloatProperty
    from fusion2usd.fusion.reader import FusionDesignReader

    design, _ = build_design(adsk)
    design.rootComponent.allJointOrigins = list(origins)  # a SWIG vector in Fusion: no count/item
    report = ExportReport()
    model = FusionDesignReader(design, ExportOptions(source_up_axis="Z", probe_joints=False), report).read()
    return design, model, report


def test_tagged_joint_origin_becomes_sensor(adsk):
    design, _ = build_design(adsk)
    arm = design.rootComponent.occurrences[1]
    camera = FakeJointOrigin("cam", arm, CAMERA_FRAME, {"sensor": "camera", "width": "640",
                                                         "json": json.dumps({"note": "front"})})
    untagged = FakeJointOrigin("plain", arm, Transform(), {"other": "x"})
    root_frame = FakeJointOrigin("tip", None, Transform(translation=Vec3(0, 0, 0.3)), {"sensor": "Frame"},
                                 geometry_origin=Vec3(0, 0, 0.0))
    from fusion2usd.model.robot import SensorKind

    _, model, report = _read_with_origins(adsk, [camera, untagged, root_frame])
    cam, tip = model.sensors
    assert (cam.name, cam.kind, cam.link) == ("cam", SensorKind.CAMERA, "arm:1")
    assert cam.transform.is_close(CAMERA_FRAME) and cam.transform.axis(2).is_close(Vec3(1, 0, 0), 1e-9)
    assert cam.metadata == {"attributes": {"sensor": "camera", "width": "640", "note": "front"}, "fusion_name": "cam"}
    assert cam.source == "arm:1/cam" and cam.params == {}
    # Root-level origin: falls back to the grounded/base link; geometry origin mismatch is reported.
    assert (tip.kind, tip.link, tip.source) == (SensorKind.FRAME, "base:1", "tip")
    assert any("'tip'" in w and "mm off" in w for w in report.warnings)


def test_unknown_sensor_kind_warns_and_duplicate_names_get_suffix(adsk):
    design, _ = build_design(adsk)
    arm, wheel = design.rootComponent.occurrences[1:]
    origins = [FakeJointOrigin("s", arm, Transform(), {"sensor": "lidar"}),
               FakeJointOrigin("s", arm, Transform(), {"sensor": "imu"}),
               FakeJointOrigin("s", wheel, Transform(), {"sensor": "contact"})]
    _, model, report = _read_with_origins(adsk, origins)
    assert [(s.name, s.link) for s in model.sensors] == [("s", "arm:1"), ("s_2", "wheel:1")]
    assert any("unknown sensor kind 'lidar'" in w for w in report.warnings)


def test_grounded_group_with_root_bodies_is_named_after_its_occurrence(adsk):
    """Root bodies put the root frame into the grounded group; the link must still be called base:1
    (live Fusion named it after the document, "(Unsaved)")."""
    core = adsk["adsk.core"]
    core.ColorProperty, core.FloatProperty = ColorProperty, FloatProperty
    from fusion2usd.fusion.reader import FusionDesignReader

    design, _ = build_design(adsk)
    design.rootComponent.name = "(Unsaved)"
    design.rootComponent.bRepBodies = Coll([Body("floor", Vec3(-1, -1, -0.1), Vec3(1, 1, 0), 5.0, Vec3(0, 0, -0.05))])
    model = FusionDesignReader(design, ExportOptions(probe_joints=False), ExportReport()).read()
    base = model.links["base:1"]
    assert base.source == "base:1" and base.transform.is_close(Transform())  # root frame anchors it
    assert "(Unsaved)" not in model.links
    assert sorted(m.name for m in base.visuals) == ["floor", "plate"]  # root bodies joined the base link


def test_rotation_values_wrap_into_the_limits(adsk):
    """Fusion reports -0.01 rad as 6.27; left unwrapped, a rezeroed limit lands a full turn away."""
    from fusion2usd.fusion.joints import wrap_angle
    from fusion2usd.model.robot import Limit

    tau = 2 * math.pi
    assert wrap_angle(tau - 0.01, Limit(-0.35, 0.35)) == pytest.approx(-0.01)
    assert wrap_angle(tau - 0.01, None) == pytest.approx(-0.01)
    assert wrap_angle(math.radians(200), Limit(math.radians(-60), math.radians(210))) == pytest.approx(math.radians(200))
    assert wrap_angle(math.radians(200), None) == pytest.approx(math.radians(-160))
    assert wrap_angle(0.1, Limit(-1.0, 1.0)) == pytest.approx(0.1)


def test_resolve_key_prefixes_from_the_assembly_context():
    from fusion2usd.fusion.occurrences import resolve_key

    known = {"hand_r:1+I-R:1+I-PP:1": 0, "hand_l:1+I-R:1+I-PP:1": 0, "hand_r:1+R-Carpals:1": 0}
    assert resolve_key("I-R:1+I-PP:1", "hand_r:1+I-R:1", known) == "hand_r:1+I-R:1+I-PP:1"
    assert resolve_key("I-R:1+I-PP:1", "hand_l:1+I-R:1", known) == "hand_l:1+I-R:1+I-PP:1"
    assert resolve_key("R-Carpals:1", "hand_r:1", known) == "hand_r:1+R-Carpals:1"
    assert resolve_key("R-Carpals:1", "", known) is None
    assert resolve_key(None, "hand_r:1", known) is None


def test_joint_inside_external_reference_is_kept(adsk):
    """Fusion reports a referenced design's joint occurrences relative to that design."""
    core = adsk["adsk.core"]
    core.ColorProperty, core.FloatProperty = ColorProperty, FloatProperty
    from fusion2usd.fusion.reader import FusionDesignReader

    design, _ = build_design(adsk)
    root = design.rootComponent
    base, arm, wheel = root.occurrences
    hand = Occurrence("hand:1", lambda: Transform())
    hand.childOccurrences = Coll([wheel])
    root.occurrences = Coll([base, arm, hand])
    wheel.fullPathName = "hand:1+wheel:1"
    rev2 = root.allJoints[1]
    rev2.occurrenceOne = SimpleNamespace(fullPathName="wheel:1", transform2=Matrix(Transform()))
    rev2.assemblyContext = SimpleNamespace(fullPathName="hand:1", name="hand:1")
    # The proxy's geometry is useless. The native joint holds the frame in the referenced design's space,
    # where the parent part (arm) sits in a pose of its own, not the one it has in the assembly.
    hand._pose = lambda: Transform(Quat.from_axis_angle(Vec3(0, 0, 1), math.pi / 2), Vec3(0.5, 0.0, 0.0))
    rev2.geometryTwoTransform = Matrix(Transform(translation=Vec3(9, 9, 9)))
    arm_native = Transform(Quat.from_axis_angle(Vec3(1, 1, 0).normalized(), 0.7), Vec3(-0.2, 0.1, 0.3))
    to_native = arm_native * ARM_POSE.inverse()
    rev2.nativeObject = SimpleNamespace(
        geometryTwoTransform=Matrix(to_native * Transform(Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2),
                                                          AXIS2_POINT)),
        occurrenceTwo=SimpleNamespace(transform2=Matrix(arm_native)),
        jointMotion=RevoluteMotion(adsk["adsk.fusion"].JointTypes.RevoluteJointType,
                                   to_native.apply_vector(Vec3(1, 0, 0)), value=0.4))
    report = ExportReport()
    opts = ExportOptions(source_up_axis="Z", probe_joints=False)
    model = FusionDesignReader(design, opts, report).read()

    assert not any("unknown occurrence" in w for w in report.warnings)
    j2 = model.joint("rev2")
    assert j2.child == "wheel:1" and j2.parent == arm.name
    assert j2.origin.is_close(AXIS2_POINT, 1e-9)
    assert j2.dofs[0].axis.is_close(Vec3(1, 0, 0), 1e-9)
