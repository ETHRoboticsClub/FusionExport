"""Live self-test: builds a test mechanism in a NEW Fusion design, exports it and checks the result.

Run from Utilities > Scripts and Add-Ins. The test document is left open (unsaved) for inspection;
the report is written to ``<tmp>/selftest_report.txt`` and shown in a message box.
"""

from __future__ import annotations

import math
import os
import re
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import adsk.core
import adsk.fusion

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # realpath: works when symlinked
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
# Fusion keeps one interpreter alive: drop stale copies so each run tests the current code.
for _name in [m for m in sys.modules if m == "fusion2usd" or m.startswith("fusion2usd.")]:
    del sys.modules[_name]

from fusion2usd.core.math3d import Vec3  # noqa: E402
from fusion2usd.core.report import ExportReport  # noqa: E402
from fusion2usd.fusion.joints import JointCollector  # noqa: E402
from fusion2usd.fusion.metadata import AttributeWriter  # noqa: E402
from fusion2usd.model.robot import RobotModel  # noqa: E402

XY = Tuple[float, float]
CM = 0.01

# Test mechanism, in cm. Four-bar: ground A-D, crank A-B, coupler B-C, rocker D-C (Grashof crank-rocker).
A, B, C, D = (0.0, 0.0), (0.0, 5.0), (18.0, 8.0), (20.0, 0.0)
GEAR_CENTER, GEAR_RADIUS = (55.0, 0.0), 3.0
SLIDER_POINT, BALL_POINT = (30.0, 0.0), (33.0, 0.0)
SLIDE_LIMITS_CM = (0.0, 10.0)
CRANK_LIMITS = (-1.0, 1.0)
GEAR_RATIO = 2.0
BAR_WIDTH, PART_HEIGHT = 1.0, 1.0
CRANK_ATTRIBUTES = {"effort": "2.5", "velocity": "3.0"}
# Sensors (joint origins at sketch points, z = 0): a camera on the crank looking along world +X, an IMU on the coupler.
CAMERA_POINT, IMU_POINT = (0.0, 2.5), (9.0, 6.5)
SENSORS: Dict[str, Tuple[str, str, XY, Dict[str, str]]] = {  # name -> (kind, part, point, attributes)
    "camera": ("camera", "crank", CAMERA_POINT, {"sensor": "camera", "width": "640", "height": "480", "fov": "90"}),
    "imu": ("imu", "coupler", IMU_POINT, {"sensor": "imu"}),
}

# Expected pivots (cm) of the revolute/ball joints by Fusion joint name.
PIVOTS: Dict[str, XY] = {"crank": A, "coupler_joint": B, "loop_joint": C, "rocker_joint": D,
                         "gear": GEAR_CENTER, "ball": BALL_POINT}
REVOLUTES = ("crank", "coupler_joint", "loop_joint", "rocker_joint", "gear")
PARTS = ("base", "crank", "coupler", "rocker", "slider", "gear", "ball_arm")


def _p(xy: XY, z: float = 0.0) -> Any:
    return adsk.core.Point3D.create(xy[0], xy[1], z)


def bar_corners(p: XY, q: XY, width: float) -> List[XY]:
    """Rectangle of ``width`` around the segment p-q, extended by width/2 at both ends."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    n = math.hypot(dx, dy)
    ux, uy = dx / n, dy / n
    h = width / 2.0
    ex, ey = ux * h, uy * h      # along the bar
    nx, ny = -uy * h, ux * h     # across the bar
    return [(p[0] - ex + nx, p[1] - ey + ny), (p[0] - ex - nx, p[1] - ey - ny),
            (q[0] + ex - nx, q[1] + ey - ny), (q[0] + ex + nx, q[1] + ey + ny)]


@dataclass
class Part:
    occurrence: Any
    sketch: Any
    bbox_cm: Tuple[Vec3, Vec3]          # expected world bounding box
    points: Dict[XY, Any] = field(default_factory=dict)

    def joint_geometry(self, xy: XY) -> Any:
        proxy = self.points[xy].createForAssemblyContext(self.occurrence)
        return adsk.fusion.JointGeometry.createByPoint(proxy)


class MechanismBuilder:
    """Creates the test parts and joints in the given (new) design."""

    def __init__(self, design: Any) -> None:
        self.design = design
        self.root = design.rootComponent
        self.parts: Dict[str, Part] = {}
        self.joints: Dict[str, Any] = {}

    def build(self) -> None:
        base = self._polygon_part("base", [(-5.0, -8.0), (60.0, -8.0), (60.0, 15.0), (-5.0, 15.0)],
                                  [A, D, SLIDER_POINT, GEAR_CENTER], height=-PART_HEIGHT)
        base.occurrence.isGrounded = True
        self._polygon_part("crank", bar_corners(A, B, BAR_WIDTH), [A, B, CAMERA_POINT])
        self._polygon_part("coupler", bar_corners(B, C, BAR_WIDTH), [B, C, IMU_POINT])
        self._polygon_part("rocker", bar_corners(D, C, BAR_WIDTH), [D, C])
        x0, x1 = SLIDER_POINT[0] - 2.0, SLIDER_POINT[0] + 4.0
        self._polygon_part("slider", [(x0, -1.0), (x1, -1.0), (x1, 1.0), (x0, 1.0)], [SLIDER_POINT, BALL_POINT])
        self._polygon_part("ball_arm", bar_corners(BALL_POINT, (BALL_POINT[0], 6.0), BAR_WIDTH), [BALL_POINT])
        self._disc_part("gear", GEAR_CENTER, GEAR_RADIUS)

        jd = adsk.fusion.JointDirections
        # The four-bar uses as-built joints: the parts are already in a consistent closed-loop pose.
        crank = self._as_built("crank", "crank", "base", A)
        self._as_built("coupler_joint", "coupler", "crank", B)
        loop = self._as_built("loop_joint", "rocker", "coupler", C)
        self._as_built("rocker_joint", "rocker", "base", D)
        _set_limits(crank.jointMotion.rotationLimits, *CRANK_LIMITS)

        slider = self._joint("slider", "slider", "base", SLIDER_POINT,
                             lambda i: i.setAsSliderJointMotion(jd.XAxisJointDirection))
        _set_limits(slider.jointMotion.slideLimits, *SLIDE_LIMITS_CM)
        gear = self._joint("gear", "gear", "base", GEAR_CENTER,
                           lambda i: i.setAsRevoluteJointMotion(jd.ZAxisJointDirection))
        self._joint("ball", "ball_arm", "slider", BALL_POINT,
                    lambda i: i.setAsBallJointMotion(jd.ZAxisJointDirection, jd.XAxisJointDirection))

        mt = adsk.fusion.JointMotionTypes
        link_input = self.root.motionLinks.createInput(crank, gear)
        link_input.motionOne = mt.RevoluteJointRotateMotionType
        link_input.motionTwo = mt.RevoluteJointRotateMotionType
        link_input.valueOne = adsk.core.ValueInput.createByReal(1.0)          # rad
        link_input.valueTwo = adsk.core.ValueInput.createByReal(GEAR_RATIO)   # rad
        link_input.isReversed = False
        if self.root.motionLinks.add(link_input) is None:
            raise RuntimeError("creating the motion link failed")

        writer = AttributeWriter()
        writer.write(crank, CRANK_ATTRIBUTES)
        writer.write(loop, {"loop_closure": "true"})  # keep the crank (mimic leader) in the tree

        crank_axis_x = self.parts["crank"].occurrence.component.xConstructionAxis  # crank occurrence is identity
        for name, (_, part, at, attributes) in SENSORS.items():
            origin = self._joint_origin(name, part, at, crank_axis_x if name == "camera" else None)
            writer.write(origin, attributes)

    # -- parts
    def _new_component(self, name: str) -> Tuple[Any, Any]:
        occ = self.root.occurrences.addNewComponent(adsk.core.Matrix3D.create())
        occ.component.name = name
        sketch = occ.component.sketches.add(occ.component.xYConstructionPlane)
        return occ, sketch

    def _polygon_part(self, name: str, corners: Sequence[XY], points: Sequence[XY],
                      height: float = PART_HEIGHT) -> Part:
        occ, sketch = self._new_component(name)
        lines = sketch.sketchCurves.sketchLines
        first = prev = lines.addByTwoPoints(_p(corners[0]), _p(corners[1]))
        for c in corners[2:]:
            prev = lines.addByTwoPoints(prev.endSketchPoint, _p(c))
        lines.addByTwoPoints(prev.endSketchPoint, first.startSketchPoint)
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
        bbox = (Vec3(min(xs), min(ys), min(0.0, height)), Vec3(max(xs), max(ys), max(0.0, height)))
        return self._finish(name, occ, sketch, bbox, points, height)

    def _disc_part(self, name: str, center: XY, radius: float) -> Part:
        occ, sketch = self._new_component(name)
        sketch.sketchCurves.sketchCircles.addByCenterRadius(_p(center), radius)
        bbox = (Vec3(center[0] - radius, center[1] - radius, 0.0),
                Vec3(center[0] + radius, center[1] + radius, PART_HEIGHT))
        return self._finish(name, occ, sketch, bbox, [center], PART_HEIGHT)

    def _finish(self, name: str, occ: Any, sketch: Any, bbox: Tuple[Vec3, Vec3], points: Sequence[XY],
                height: float) -> Part:
        part = Part(occ, sketch, bbox)
        for xy in points:
            part.points[xy] = sketch.sketchPoints.add(_p(xy))
        extrudes = occ.component.features.extrudeFeatures
        extrudes.addSimple(sketch.profiles.item(0), adsk.core.ValueInput.createByReal(height),
                           adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        self.parts[name] = part
        return part

    # -- joints (occurrence one = moving part, two = parent)
    def _as_built(self, name: str, one: str, two: str, at: XY) -> Any:
        joints = self.root.asBuiltJoints
        inp = joints.createInput(self.parts[one].occurrence, self.parts[two].occurrence,
                                 self.parts[two].joint_geometry(at))
        inp.setAsRevoluteJointMotion(adsk.fusion.JointDirections.ZAxisJointDirection)
        return self._named(name, joints.add(inp))

    def _joint(self, name: str, one: str, two: str, at: XY, motion: Callable[[Any], Any]) -> Any:
        joints = self.root.joints
        inp = joints.createInput(self.parts[one].joint_geometry(at), self.parts[two].joint_geometry(at))
        motion(inp)
        return self._named(name, joints.add(inp))

    def _joint_origin(self, name: str, part: str, at: XY, z_axis: Any = None) -> Any:
        """A joint origin in the part's component (read back as a root-context proxy)."""
        origins = self.parts[part].occurrence.component.jointOrigins
        inp = origins.createInput(adsk.fusion.JointGeometry.createByPoint(self.parts[part].points[at]))
        if z_axis is not None:
            inp.zAxisEntity = z_axis
        origin = origins.add(inp)
        if origin is None:
            raise RuntimeError(f"creating joint origin {name!r} failed")
        origin.name = name
        return origin

    def _named(self, name: str, joint: Any) -> Any:
        if joint is None:
            raise RuntimeError(f"creating joint {name!r} failed")
        joint.name = name
        self.joints[name] = joint
        return joint


def _set_limits(limits: Any, lower: float, upper: float) -> None:
    limits.minimumValue = lower
    limits.maximumValue = upper
    limits.isMinimumValueEnabled = True
    limits.isMaximumValueEnabled = True


# ---------------------------------------------------------------- checks


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""

    def line(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}" + (f": {self.detail}" if self.detail else "")


def joint_values(root: Any) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for fj in JointCollector(root, ExportReport()).collect():
        out[fj.name] = [getattr(fj.motion, s.value_attr) for s in fj.specs]
    return out


def _by_source(model: RobotModel, name: str) -> List[Any]:
    return [j for j in model.joints if j.source == name or j.metadata.get("fusion_name") == name]


def _distance_to_line(p: Vec3, origin: Vec3, axis: Vec3) -> float:
    u = axis.normalized()
    d = p - origin
    return (d - u * d.dot(u)).norm()


class ResultChecker:
    """Assertions on the raw model (Fusion reader output), processed model, report and files."""

    def __init__(self, raw: RobotModel, processed: RobotModel, report: ExportReport, usda_text: str,
                 bboxes_cm: Dict[str, Tuple[Vec3, Vec3]], values_before: Dict[str, List[float]],
                 values_after: Dict[str, List[float]], urdf_path: str) -> None:
        self.raw, self.processed, self.report = raw, processed, report
        self.usda, self.bboxes = usda_text, bboxes_cm
        self.before, self.after = values_before, values_after
        self.urdf_path = urdf_path

    def run(self) -> List[Check]:
        checks: List[Check] = []
        for fn in (self.link_count, self.masses, self.loop_joint, self.mimic, self.slider_limits,
                   self.probe_warnings, self.pivots, self.mesh_bounds, self.values_restored,
                   self.attributes, self.sensors, self.outputs):
            try:
                checks.extend(fn())
            except Exception:
                checks.append(Check(fn.__name__, False, "raised:\n" + traceback.format_exc()))
        return checks

    def link_count(self) -> List[Check]:
        real = [l for l in self.processed.links.values() if not l.is_virtual]
        return [Check("raw link count", len(self.raw.links) == len(PARTS), f"{sorted(self.raw.links)}"),
                Check("processed non-virtual link count", len(real) == len(PARTS), f"{len(real)}")]

    def masses(self) -> List[Check]:
        missing = [n for n, l in self.raw.links.items() if l.mass is None or l.mass.mass <= 0]
        return [Check("all links have mass", not missing, f"missing: {missing}" if missing else "")]

    def loop_joint(self) -> List[Check]:
        excluded = [j for j in self.processed.joints if j.exclude_from_articulation]
        names = [j.source or j.name for j in excluded]
        return [Check("exactly one excluded loop joint", len(excluded) == 1, f"{names}"),
                Check("loop cut at the preferred joint", names == ["loop_joint"], f"{names}")]

    def mimic(self) -> List[Check]:
        gear = {j.name for j in _by_source(self.processed, "gear")}
        couplings = [c for c in self.processed.couplings if c.follower.joint in gear]
        ratio = couplings[0].ratio if couplings else None
        gearings = [float(g) for g in re.findall(r"physxMimicJoint:\w+:gearing\s*=\s*([-+0-9.eE]+)", self.usda)]
        ok_ratio = ratio is not None and abs(abs(ratio) - GEAR_RATIO) < 1e-6
        ok_usd = len(gearings) == 1 and abs(abs(gearings[0]) - GEAR_RATIO) < 1e-4
        return [Check("motion link coupling present", ok_ratio, f"ratio={ratio}"),
                Check("USD mimic joint gearing ±2", ok_usd, f"gearing={gearings} (sign reported, not asserted)")]

    def slider_limits(self) -> List[Check]:
        (j,) = _by_source(self.raw, "slider")
        lim = j.dofs[0].limit
        lo, hi = SLIDE_LIMITS_CM[0] * CM, SLIDE_LIMITS_CM[1] * CM
        ok = lim is not None and abs(lim.lower - lo) < 1e-9 and abs(lim.upper - hi) < 1e-9
        return [Check("slider limits in meters", ok, f"{lim}")]

    def probe_warnings(self) -> List[Check]:
        bad = [w for w in self.report.warnings if "deviates" in w or "probing failed" in w or "off the measured" in w]
        movable = [j for j in self.raw.joints if j.dofs]
        unprobed = [j.name for j in movable if len(j.metadata.get("probed", [])) != len(j.dofs)]
        return [Check("probe: no large deviations or failures", not bad, "; ".join(bad)),
                Check("probe: every DOF measured", not unprobed, f"unprobed: {unprobed}" if unprobed else "")]

    def pivots(self) -> List[Check]:
        out = []
        for name, xy in PIVOTS.items():
            (j,) = _by_source(self.raw, name)
            expected = Vec3(xy[0] * CM, xy[1] * CM, 0.0)
            err = (j.origin - expected).norm()
            line_err = max(_distance_to_line(expected, j.origin, d.axis) for d in j.dofs
                           if d.type.value == "rotation")
            out.append(Check(f"joint {name}: origin at pivot and on measured axis",
                             err < 1e-4 and line_err < 1e-4, f"origin error {err:.2e} m, axis distance {line_err:.2e} m"))
            if name in REVOLUTES:
                z = abs(j.dofs[0].axis.normalized().z)
                out.append(Check(f"joint {name}: axis is ±Z", abs(z - 1.0) < 1e-6, f"axis={j.dofs[0].axis}"))
        return out

    def mesh_bounds(self) -> List[Check]:
        out = []
        for part, (lo_cm, hi_cm) in self.bboxes.items():
            links = [l for l in self.raw.links.values() if l.source.split("+")[-1].startswith(part + ":")]
            if len(links) != 1 or not links[0].visuals:
                out.append(Check(f"mesh bounds {part}", False, f"links={[l.name for l in links]}"))
                continue
            link = links[0]
            pts = [p for m in link.visuals for p in m.transformed(link.transform).points]
            lo = Vec3(min(pts[0::3]), min(pts[1::3]), min(pts[2::3]))
            hi = Vec3(max(pts[0::3]), max(pts[1::3]), max(pts[2::3]))
            err = max((lo - lo_cm * CM).norm(), (hi - hi_cm * CM).norm())
            local = [p for m in link.visuals for p in m.points]
            extent = max(abs(v) for v in local)
            out.append(Check(f"mesh bounds {part} (link frame -> world)", err < 2e-3,
                             f"error {err:.2e} m, max |local coordinate| {extent:.3f} m"))
        return out

    def values_restored(self) -> List[Check]:
        diffs = {n: (v, self.after.get(n)) for n, v in self.before.items()
                 if self.after.get(n) is None or any(abs(a - b) > 1e-9 for a, b in zip(v, self.after[n]))}
        return [Check("joint values restored after probing", not diffs, f"{diffs}" if diffs else "")]

    def attributes(self) -> List[Check]:
        joints = _by_source(self.processed, "crank")
        props = joints[0].properties if joints else None
        ok = props is not None and props.effort == float(CRANK_ATTRIBUTES["effort"]) \
            and props.velocity == float(CRANK_ATTRIBUTES["velocity"])
        return [Check("fusion2usd attributes -> joint properties", ok,
                      f"effort={getattr(props, 'effort', None)}, velocity={getattr(props, 'velocity', None)}")]

    def sensors(self) -> List[Check]:
        by_name = {s.metadata.get("fusion_name", s.name): s for s in self.raw.sensors}
        out = [Check("raw model has 2 sensors", len(self.raw.sensors) == len(SENSORS),
                     f"{[(s.name, s.kind, s.link) for s in self.raw.sensors]}")]
        for name, (kind, part, xy, _) in SENSORS.items():
            s = by_name.get(name)
            link = self.raw.links.get(s.link) if s is not None else None
            ok_link = link is not None and link.source.split("+")[-1].startswith(part + ":")
            out.append(Check(f"sensor {name}: kind {kind} on the {part} link",
                             s is not None and getattr(s.kind, "value", s.kind) == kind and ok_link,
                             f"kind={getattr(s, 'kind', None)}, link={getattr(s, 'link', None)}"))
            if s is not None:
                err = (s.transform.translation - Vec3(xy[0] * CM, xy[1] * CM, 0.0)).norm()
                out.append(Check(f"sensor {name}: world origin", err < 1e-4, f"error {err:.2e} m"))
        camera = by_name.get("camera")
        z = camera.transform.axis(2) if camera is not None else None
        out.append(Check("camera +Z axis is world +X", z is not None and z.dot(Vec3(1.0, 0.0, 0.0)) > 1 - 1e-6,
                         f"z={z}"))
        out.append(Check("USDA contains a Camera prim", re.search(r"\bdef Camera\b", self.usda) is not None))
        return out

    def outputs(self) -> List[Check]:
        return [Check("USDA written", bool(self.usda)),
                Check("URDF written", os.path.isfile(self.urdf_path), self.urdf_path)]


# ---------------------------------------------------------------- driver


class SelfTest:
    def __init__(self, out_dir: Optional[str] = None) -> None:
        self.app = adsk.core.Application.get()
        # Fixed location so the report is easy to find after each run (overwritten every time).
        self.out_dir = out_dir or os.path.join(tempfile.gettempdir(), "fusion2usd_selftest")
        os.makedirs(self.out_dir, exist_ok=True)
        self.usda_path = os.path.join(self.out_dir, "selftest.usda")
        self.report_path = os.path.join(self.out_dir, "selftest_report.txt")

    def run(self) -> str:
        from fusion2usd.exporter import Exporter
        from fusion2usd.kinematics import OverrideSet, PassContext, default_pipeline
        from fusion2usd.options import ExportOptions
        from fusion2usd.serialization import ModelSerializer

        doc = self.app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
        design = adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType"))
        design.designType = adsk.fusion.DesignTypes.ParametricDesignType
        builder = MechanismBuilder(design)
        builder.build()

        before = joint_values(design.rootComponent)
        options = ExportOptions(output_path=self.usda_path, write_urdf=True, write_model_json=True,
                                robot_name="selftest")
        report = Exporter(options).export_design(design)
        after = joint_values(design.rootComponent)

        raw = ModelSerializer().load(os.path.splitext(self.usda_path)[0] + ".model.json")
        processed = default_pipeline(OverrideSet.load(None)).run(raw.copy(), PassContext(options, ExportReport()))
        with open(self.usda_path, "r", encoding="utf-8") as f:
            usda = f.read()
        bboxes = {name: part.bbox_cm for name, part in builder.parts.items()}
        checks = ResultChecker(raw, processed, report, usda, bboxes, before, after,
                               os.path.splitext(self.usda_path)[0] + ".urdf").run()
        return self._write(checks, report)

    def _write(self, checks: List[Check], report: ExportReport) -> str:
        failed = sum(not c.passed for c in checks)
        lines = [f"Fusion2USD self-test: {len(checks) - failed}/{len(checks)} checks passed",
                 f"Output directory: {self.out_dir}", ""]
        lines += [c.line() for c in checks]
        lines += ["", "Export report:", report.summary()]
        text = "\n".join(lines)
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(text)
        return text


def run(context: Any) -> None:
    ui = adsk.core.Application.get().userInterface
    try:
        text = SelfTest().run()
    except Exception:
        text = f"Fusion2USD self-test crashed:\n\n{traceback.format_exc()}"
    ui.messageBox(text, "Fusion2USD self-test")
