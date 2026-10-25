from __future__ import annotations

import math

import pytest

from fusion2usd.core.math3d import Vec3

from _usd_models import ALL_MODELS, DEG, arm_model, ball_model, four_bar_model, mimic_model
from fusion2usd.core.report import ExportReport
from fusion2usd.kinematics import JointDecompositionPass, PassContext
from fusion2usd.options import CollisionApproximation, ExportOptions, PhysicsEngine
from fusion2usd.usd.builder import UsdRobotBuilder
from fusion2usd.usd.writer import UsdWriter

pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

TOL = 1e-5


def build(model, options=None):
    report = ExportReport()
    text = UsdRobotBuilder(options or ExportOptions(), report).build(model).to_string()
    layer = Sdf.Layer.CreateAnonymous(".usda")
    assert layer.ImportFromString(text), text
    stage = Usd.Stage.Open(layer)
    stage._layer = layer  # keep alive
    return stage, report


def world(stage, path) -> Gf.Matrix4d:
    return UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def local_pose(prim, i) -> Gf.Matrix4d:
    pos = prim.GetAttribute(f"physics:localPos{i}").Get()
    q = prim.GetAttribute(f"physics:localRot{i}").Get()
    m = Gf.Matrix4d(1.0)
    m.SetRotateOnly(Gf.Quatd(q.GetReal(), Gf.Vec3d(q.GetImaginary())))
    m.SetTranslateOnly(Gf.Vec3d(pos))
    return m


def body_world(stage, prim, i) -> Gf.Matrix4d:
    targets = prim.GetRelationship(f"physics:body{i}").GetTargets()
    return world(stage, targets[0]) if targets else Gf.Matrix4d(1.0)


def assert_matrix_close(a: Gf.Matrix4d, b: Gf.Matrix4d) -> None:
    for r in range(4):
        for c in range(4):
            assert a[r][c] == pytest.approx(b[r][c], abs=TOL), f"\n{a}\n!=\n{b}"


def joint_motion(prim) -> Gf.Matrix4d:
    m = Gf.Matrix4d(1.0)
    if prim.IsA(UsdPhysics.RevoluteJoint):
        q = prim.GetAttribute("state:angular:physics:position").Get()
        m.SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), q))
    elif prim.IsA(UsdPhysics.PrismaticJoint):
        m.SetTranslate(Gf.Vec3d(prim.GetAttribute("state:linear:physics:position").Get(), 0, 0))
    return m


def joint_frames(stage, prim):
    """World frames (row-vector convention) of the joint on body0 and body1."""
    f0 = local_pose(prim, 0) * body_world(stage, prim, 0)
    f1 = local_pose(prim, 1) * body_world(stage, prim, 1)
    return f0, f1


@pytest.mark.parametrize("make", ALL_MODELS)
def test_stage_valid_and_kinematics_consistent(make):
    model = make()
    stage, _ = build(model)
    root = stage.GetDefaultPrim()
    assert root.IsValid() and root.GetMetadata("kind") == "component"
    art_roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    expected_root = root if model.fixed_base else stage.GetPrimAtPath(f"{root.GetPath()}/{model.base_link}")
    assert art_roots == [expected_root]
    assert stage.GetMetadata("upAxis") == "Z" and stage.GetMetadata("metersPerUnit") == 1.0
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    assert len(joints) == len(model.joints) + (1 if model.fixed_base else 0)
    for prim in joints:
        f0, f1 = joint_frames(stage, prim)
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint) \
                or prim.IsA(UsdPhysics.FixedJoint):
            assert_matrix_close(f1, joint_motion(prim) * f0)
        j = model.find_joint(prim.GetName())
        if j is None or not j.dofs:
            continue
        # Joint frame sits on the anchor and its X axis follows the first DOF axis.
        assert Gf.Vec3d(*j.origin) == pytest.approx(f0.ExtractTranslation(), abs=TOL)
        x = f0.TransformDir(Gf.Vec3d(1, 0, 0))
        assert list(x) == pytest.approx(list(j.dofs[0].axis.normalized()), abs=TOL)


def test_arm():
    model = arm_model()
    stage, report = build(model)
    root = stage.GetPrimAtPath("/Test_Arm")
    assert stage.GetDefaultPrim() == root
    assert root.GetCustomData()["source_document"] == "arm.f3d"
    assert root.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is False

    link1 = stage.GetPrimAtPath("/Test_Arm/link1")
    assert link1.HasAPI(UsdPhysics.RigidBodyAPI) and link1.HasAPI(UsdPhysics.MassAPI)
    mass = UsdPhysics.MassAPI(link1)
    assert mass.GetMassAttr().Get() == pytest.approx(1.0)
    assert list(mass.GetCenterOfMassAttr().Get()) == pytest.approx([0.01, 0.0, 0.02])
    assert sorted(mass.GetDiagonalInertiaAttr().Get()) == pytest.approx([0.01, 0.02, 0.03])
    assert stage.GetPrimAtPath("/Test_Arm/base").GetCustomData()["occurrence"] == "Base:1"

    # root joint welds the base to the world
    rj = UsdPhysics.FixedJoint(stage.GetPrimAtPath("/Test_Arm/joints/root_joint"))
    assert rj and rj.GetBody0Rel().GetTargets() == []
    assert rj.GetBody1Rel().GetTargets() == [Sdf.Path("/Test_Arm/base")]

    sh = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/Test_Arm/joints/shoulder"))
    assert sh and sh.GetAxisAttr().Get() == "X"
    assert sh.GetLowerLimitAttr().Get() == pytest.approx(-DEG)
    assert sh.GetUpperLimitAttr().Get() == pytest.approx(2 * DEG)
    assert sh.GetExcludeFromArticulationAttr().Get() is False
    assert sh.GetCollisionEnabledAttr().Get() is False
    drive = UsdPhysics.DriveAPI(sh.GetPrim(), "angular")
    assert drive.GetStiffnessAttr().Get() == pytest.approx(100 * math.pi / 180)
    assert drive.GetDampingAttr().Get() == pytest.approx(5 * math.pi / 180)
    assert drive.GetMaxForceAttr().Get() == pytest.approx(10.0)
    assert drive.GetTargetPositionAttr().Get() == pytest.approx(0.3 * DEG, rel=1e-6)
    assert drive.GetTypeAttr().Get() == "force"
    p = sh.GetPrim()
    assert p.GetAttribute("state:angular:physics:position").Get() == pytest.approx(0.3 * DEG, rel=1e-6)
    assert p.GetAttribute("physxJoint:maxJointVelocity").Get() == pytest.approx(3 * DEG, rel=1e-6)
    assert p.GetAttribute("physxJoint:armature").Get() == pytest.approx(0.01)
    cd = p.GetCustomData()
    assert cd["source"] == "Rev1" and cd["properties"]["effort"] == 10.0

    el = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/Test_Arm/joints/elbow"))
    assert el.GetLowerLimitAttr().Get() == -math.inf
    assert el.GetUpperLimitAttr().Get() == pytest.approx(1.5 * DEG)
    assert el.GetPrim().GetCustomData()["nested"]["b"] == [1.5, 2.0]
    assert UsdPhysics.DriveAPI(el.GetPrim(), "angular").GetMaxForceAttr().Get() == math.inf

    # filtered pairs between joint bodies
    fp = UsdPhysics.FilteredPairsAPI(stage.GetPrimAtPath("/Test_Arm/link2"))
    assert fp.GetFilteredPairsRel().GetTargets() == [Sdf.Path("/Test_Arm/link1")]

    # visuals / collisions / materials
    body = stage.GetPrimAtPath("/Test_Arm/base/visuals/body")
    mesh = UsdGeom.Mesh(body)
    assert len(mesh.GetPointsAttr().Get()) == 8 and list(mesh.GetFaceVertexCountsAttr().Get()) == [3] * 12
    assert mesh.GetNormalsInterpolation() == "vertex"
    assert len(mesh.GetExtentAttr().Get()) == 2
    mat, _ = UsdShade.MaterialBindingAPI(body).ComputeBoundMaterial()
    assert mat.GetPath() == Sdf.Path("/Test_Arm/Looks/steel")
    assert mat.GetPath().HasPrefix(root.GetPath())
    shader = mat.ComputeSurfaceSource()[0]
    assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
    assert shader.GetInput("metallic").Get() == pytest.approx(1.0)

    coll = stage.GetPrimAtPath("/Test_Arm/link1/collisions/c")
    assert coll.HasAPI(UsdPhysics.CollisionAPI) and coll.HasAPI(UsdPhysics.MeshCollisionAPI)
    assert UsdPhysics.MeshCollisionAPI(coll).GetApproximationAttr().Get() == "convexDecomposition"
    assert UsdGeom.Imageable(coll).ComputePurpose() == UsdGeom.Tokens.guide
    # link2 has no collision meshes: falls back to visuals, with its own approximation
    coll2 = stage.GetPrimAtPath("/Test_Arm/link2/collisions/v")
    assert UsdPhysics.MeshCollisionAPI(coll2).GetApproximationAttr().Get() == "convexHull"
    assert not report.warnings, report.warnings


def test_collision_none_and_disabled():
    model = arm_model()
    model.links["link1"].collision_approximation = "none"
    stage, _ = build(model)
    assert not stage.GetPrimAtPath("/Test_Arm/link1/collisions").IsValid()
    assert stage.GetPrimAtPath("/Test_Arm/link2/collisions").IsValid()
    stage, _ = build(arm_model(), ExportOptions(export_collisions=False))
    assert not any(p.HasAPI(UsdPhysics.CollisionAPI) for p in stage.Traverse())


def test_four_bar_loop_closure():
    stage, _ = build(four_bar_model())
    loop = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/four_bar/joints/loop_joint"))
    assert loop.GetExcludeFromArticulationAttr().Get() is True
    assert not loop.GetPrim().HasAPI(UsdPhysics.DriveAPI)
    assert not stage.GetPrimAtPath("/four_bar/joints/root_joint").IsValid()
    assert UsdPhysics.DriveAPI(stage.GetPrimAtPath("/four_bar/joints/crank_joint"), "angular")
    assert not stage.GetPrimAtPath("/four_bar/joints/coupler_joint").HasAPI(UsdPhysics.DriveAPI)
    fp = UsdPhysics.FilteredPairsAPI(stage.GetPrimAtPath("/four_bar/ground"))
    assert fp.GetFilteredPairsRel().GetTargets() == [Sdf.Path("/four_bar/rocker")]


def test_prismatic_mimic():
    model = mimic_model()
    stage, report = build(model)
    rack = stage.GetPrimAtPath("/rack_pinion/joints/rack_joint")
    pj = UsdPhysics.PrismaticJoint(rack)
    assert pj.GetLowerLimitAttr().Get() == pytest.approx(0.0)
    assert pj.GetUpperLimitAttr().Get() == pytest.approx(0.1)
    assert rack.GetAttribute("state:linear:physics:position").Get() == pytest.approx(0.02)
    # Isaac Sim 6.2 PhysX only parses rot* mimic tokens, also on prismatic joints (verified on GPU)
    assert "PhysxMimicJointAPI:rotX" in rack.GetPrimTypeInfo().GetAppliedAPISchemas()
    assert rack.GetRelationship("physxMimicJoint:rotX:referenceJoint").GetTargets() == \
        [Sdf.Path("/rack_pinion/joints/wheel_joint")]
    assert rack.GetAttribute("physxMimicJoint:rotX:referenceJointAxis").Get() == "rotX"
    # prismatic follower, revolute leader: PhysX wants m/deg
    assert rack.GetAttribute("physxMimicJoint:rotX:gearing").Get() == pytest.approx(-0.05 / DEG)
    assert rack.GetAttribute("physxMimicJoint:rotX:offset").Get() == pytest.approx(-0.001)
    # the continuous wheel leads a mimic: PhysX needs finite limits on it
    wheel = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/rack_pinion/joints/wheel_joint"))
    assert (wheel.GetLowerLimitAttr().Get(), wheel.GetUpperLimitAttr().Get()) == (-3600.0, 3600.0)
    assert any("finite limits" in w for w in report.warnings)
    assert not rack.HasAPI(UsdPhysics.DriveAPI)
    assert UsdPhysics.DriveAPI(stage.GetPrimAtPath("/rack_pinion/joints/wheel_joint"), "angular")
    # free body: plain rigid body without mass values
    loose = stage.GetPrimAtPath("/rack_pinion/loose")
    assert loose.HasAPI(UsdPhysics.RigidBodyAPI) and loose.HasAPI(UsdPhysics.MassAPI)
    assert not loose.GetAttribute("physics:mass").HasAuthoredValue()
    assert any("no mass" in w for w in report.warnings)

    stage, _ = build(mimic_model(), ExportOptions(export_mimic_joints=False))
    rack = stage.GetPrimAtPath("/rack_pinion/joints/rack_joint")
    assert not rack.GetRelationship("physxMimicJoint:transX:referenceJoint")
    assert rack.HasAPI(UsdPhysics.DriveAPI)


def _rel(stage, path):
    prim = stage.GetPrimAtPath(path)
    f0, f1 = joint_frames(stage, prim)
    return prim, f1 * f0.GetInverse()  # frame1 expressed in frame0 (row-vector convention)


def _limit(prim, axis):
    return (prim.GetAttribute(f"limit:{axis}:physics:low").Get(),
            prim.GetAttribute(f"limit:{axis}:physics:high").Get())


def test_multi_dof_d6():
    model = ball_model()
    stage, report = build(model)
    ball, rel = _rel(stage, "/ball_robot/joints/ball_joint")
    assert ball.GetTypeName() == "PhysicsJoint"
    assert UsdPhysics.Joint(ball).GetExcludeFromArticulationAttr().Get() is True
    assert not ball.HasAPI(UsdPhysics.DriveAPI)
    for axis in ("transX", "transY", "transZ"):
        assert _limit(ball, axis) == (1.0, -1.0)
    for axis in ("rotX", "rotY", "rotZ"):
        assert not ball.GetAttribute(f"limit:{axis}:physics:low")
    assert list(rel.ExtractTranslation()) == pytest.approx([0, 0, 0], abs=TOL)
    assert any("rotational limits" in w for w in report.warnings)

    cyl, rel = _rel(stage, "/ball_robot/joints/cyl_joint")
    for axis in ("transY", "transZ", "rotY", "rotZ"):
        assert _limit(cyl, axis) == (1.0, -1.0)
    # trans DOF axis is anti-aligned with frame X -> limits and motion flip sign
    assert _limit(cyl, "transX") == pytest.approx((-0.02, 0.05))
    assert list(rel.ExtractTranslation()) == pytest.approx([-0.03, 0, 0], abs=TOL)
    # a single rotation is encoded like the translation, so its limits keep the CAD convention
    assert rel.ExtractRotation().GetAngle() == pytest.approx(math.degrees(0.25), abs=0.05)  # quatf noise
    assert cyl.GetAttribute("state:rotX:physics:position").Get() == pytest.approx(math.degrees(0.25), abs=1e-4)
    assert cyl.GetAttribute("state:transX:physics:position").Get() == pytest.approx(-0.03)


def test_d6_ball_with_skewed_axes_frees_all_rotations():
    model = ball_model()
    ball = model.joint("ball_joint")
    ball.dofs[2].axis = Vec3(0.3, 0.2, 0.93).normalized()  # roll axis not orthogonal away from zero
    stage, report = build(model)
    prim = stage.GetPrimAtPath("/ball_robot/joints/ball_joint")
    for axis in ("rotX", "rotY", "rotZ"):
        assert not prim.HasAPI(UsdPhysics.LimitAPI, axis)
    assert not any("DOF skipped" in w for w in report.warnings)


def test_writer_formats(tmp_path):
    report = ExportReport()
    w = UsdWriter(ExportOptions(), report)
    for ext in ("usda", "usd", "usdc"):
        path = str(tmp_path / f"arm.{ext}")
        assert w.write(arm_model(), path) == [path]
        stage = Usd.Stage.Open(path)
        assert stage.GetDefaultPrim().GetName() == "Test_Arm"
        assert any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in stage.Traverse())
    assert Sdf.FileFormat.FindByExtension("usdc").CanRead(str(tmp_path / "arm.usd"))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["arm.usd", "arm.usda", "arm.usdc"]


def test_reserved_names_do_not_collide():
    model = four_bar_model()
    model.rename_link("crank", "joints")
    stage, report = build(model)
    assert stage.GetPrimAtPath("/four_bar/joints_2").HasAPI(UsdPhysics.RigidBodyAPI)
    assert stage.GetPrimAtPath("/four_bar/joints").GetTypeName() == "Scope"
    assert any("renamed" in w for w in report.warnings)



# ------------------------------------------------------------------ Newton target

NEWTON = ExportOptions(physics_engine=PhysicsEngine.NEWTON)


def _authored(stage, prefix):
    names = set()
    for prim in stage.Traverse():
        names |= {p.GetName() for p in prim.GetAuthoredProperties() if p.GetName().startswith(prefix)}
        names |= {str(a) for a in prim.GetPrimTypeInfo().GetAppliedAPISchemas() if str(a).startswith(prefix.title())}
    return names


@pytest.mark.parametrize("make", ALL_MODELS)
def test_each_engine_gets_only_its_own_schemas(make):
    physx, _ = build(make())
    newton, _ = build(make(), NEWTON)
    assert not _authored(physx, "newton")
    assert not _authored(newton, "physx")


def test_newton_multi_dof_joints_stay_native():
    model = ball_model()
    for name in ("ball_joint", "cyl_joint"):
        model.joint(name).exclude_from_articulation = False
    model = JointDecompositionPass().run(model, PassContext(NEWTON, ExportReport()))
    assert not any(link.is_virtual for link in model.links.values())  # no helper bodies
    stage, report = build(model, NEWTON)
    ball, rel = _rel(stage, "/ball_robot/joints/ball_joint")
    assert ball.IsA(UsdPhysics.SphericalJoint)
    assert UsdPhysics.Joint(ball).GetExcludeFromArticulationAttr().Get() is False
    assert list(rel.ExtractTranslation()) == pytest.approx([0, 0, 0], abs=TOL)
    assert any("ball-joint limits are not exported" in w for w in report.warnings)
    cyl = stage.GetPrimAtPath("/ball_robot/joints/cyl_joint")
    assert cyl.GetTypeName() == "PhysicsJoint"
    assert UsdPhysics.Joint(cyl).GetExcludeFromArticulationAttr().Get() is False


def test_newton_loop_closing_ball_is_spherical():
    stage, _ = build(ball_model(), NEWTON)
    ball = stage.GetPrimAtPath("/ball_robot/joints/ball_joint")
    assert ball.IsA(UsdPhysics.SphericalJoint)
    assert UsdPhysics.Joint(ball).GetExcludeFromArticulationAttr().Get() is True


def test_newton_mimic_and_continuous_leader():
    stage, report = build(mimic_model(), NEWTON)
    rack = stage.GetPrimAtPath("/rack_pinion/joints/rack_joint")
    assert "NewtonMimicAPI" in rack.GetPrimTypeInfo().GetAppliedAPISchemas()
    assert rack.GetRelationship("newton:mimicJoint").GetTargets() == [Sdf.Path("/rack_pinion/joints/wheel_joint")]
    # q_f = coef1 q_l + coef0: coef1 in joint coordinates (m/rad), coef0 in m
    assert rack.GetAttribute("newton:mimicCoef1").Get() == pytest.approx(0.05)
    assert rack.GetAttribute("newton:mimicCoef0").Get() == pytest.approx(0.001)
    # Newton couples continuous leaders, so no ±3600° stand-in
    wheel = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/rack_pinion/joints/wheel_joint"))
    assert not wheel.GetLowerLimitAttr().HasAuthoredValue()
    assert any("Kamino" in w for w in report.warnings)


def test_newton_articulation_and_joint_dynamics():
    stage, _ = build(arm_model(), NEWTON)
    root = stage.GetDefaultPrim()
    assert "NewtonArticulationRootAPI" in root.GetPrimTypeInfo().GetAppliedAPISchemas()
    assert root.GetAttribute("newton:selfCollisionEnabled").Get() is False  # Newton defaults to True
    shoulder = stage.GetPrimAtPath("/Test_Arm/joints/shoulder")
    assert "NewtonJointAPI" in shoulder.GetPrimTypeInfo().GetAppliedAPISchemas()
    assert shoulder.GetAttribute("newton:velocityLimit").Get() == pytest.approx(3 * DEG, rel=1e-6)
    assert shoulder.GetAttribute("newton:armature").Get() == pytest.approx(0.01)


def test_newton_sdf_collision():
    model = arm_model()
    stage, _ = build(model, ExportOptions(physics_engine=PhysicsEngine.NEWTON,
                                          collision_approximation=CollisionApproximation.SDF))
    mesh = stage.GetPrimAtPath("/Test_Arm/link1/collisions/c")
    assert "NewtonSDFCollisionAPI" in mesh.GetPrimTypeInfo().GetAppliedAPISchemas()
    assert not mesh.GetAttribute("physics:approximation").HasAuthoredValue()


def test_newton_d6_free_axes_get_infinite_limits():
    model = ball_model()
    model.joint("cyl_joint").exclude_from_articulation = False
    stage, _ = build(model, NEWTON)
    cyl = stage.GetPrimAtPath("/ball_robot/joints/cyl_joint")
    assert _limit(cyl, "rotX") == (-math.inf, math.inf)  # Newton only makes DOFs of axes with a limit
    assert _limit(cyl, "transX") == pytest.approx((-0.02, 0.05))
    physx, _ = build(ball_model())
    assert not physx.GetPrimAtPath("/ball_robot/joints/cyl_joint").HasAPI(UsdPhysics.LimitAPI, "rotX")


def _pin_slot(reversed_=False, actuated=None):
    from fusion2usd.model.robot import Dof, DofType, Joint, JointProperties
    m = arm_model()
    m.joints = [j for j in m.joints if j.name != "elbow"]
    m.add_joint(Joint("slot", "link1", "link2", Vec3(0.3, 0.2, 0.3),
                      [Dof(DofType.ROTATION, Vec3(0, 0, 1), 0.1), Dof(DofType.TRANSLATION, Vec3(1, 0, 0), 0.02)],
                      JointProperties(actuated=actuated), metadata={"reversed": True} if reversed_ else {}))
    return m


@pytest.mark.parametrize("reversed_, actuated, split", [
    (False, None, False), (True, None, True), (False, True, True)])
def test_newton_splits_only_what_a_d6_cannot_express(reversed_, actuated, split):
    report = ExportReport()
    model = JointDecompositionPass().run(_pin_slot(reversed_, actuated), PassContext(NEWTON, report))
    assert any(link.is_virtual for link in model.links.values()) is split
    assert any("is split because" in i for i in report.infos) is split


def test_newton_reversed_cylindrical_stays_whole():
    model = ball_model()
    cyl = model.joint("cyl_joint")
    cyl.exclude_from_articulation, cyl.metadata["reversed"] = False, True
    model = JointDecompositionPass().run(model, PassContext(NEWTON, ExportReport()))
    assert not any(link.is_virtual for link in model.links.values())  # coaxial slide: order does not matter
