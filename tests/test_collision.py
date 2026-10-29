from __future__ import annotations

import math
import random

import pytest
from _usd_models import arm_model, ball_model, box_mesh, four_bar_model
from fusion2usd.collision import (
    convex_hull,
    fibonacci_directions,
    joint_neighbours,
    link_hull,
    support_points,
)
from fusion2usd.core.math3d import Vec3
from fusion2usd.model.robot import JointProperties
from fusion2usd.options import CollisionGeometry, ExportOptions, PhysicsEngine

pytest.importorskip("pxr")
from pxr import UsdGeom, UsdPhysics  # noqa: E402
from test_usd_builder import build  # noqa: E402

NEWTON = ExportOptions(physics_engine=PhysicsEngine.NEWTON)


def _inside(point, vertices, faces, tol=1e-9):
    for i, j, k in faces:
        a, b, c = vertices[i], vertices[j], vertices[k]
        n = Vec3(*b) - Vec3(*a)
        n = n.cross(Vec3(*c) - Vec3(*a))
        if n.dot(Vec3(*point) - Vec3(*a)) > tol * max(1.0, n.norm()):
            return False
    return True


def test_cube_hull_is_its_eight_corners_and_twelve_outward_triangles():
    cube = box_mesh("c", 0.2).points
    corners = [tuple(cube[i:i + 3]) for i in range(0, len(cube), 3)]
    vertices, faces = convex_hull(corners + [(0.0, 0.0, 0.0), (0.05, -0.02, 0.01)])
    assert sorted(vertices) == sorted(corners) and len(faces) == 12
    assert _inside((0.0, 0.0, 0.0), vertices, faces) and not _inside((0.0, 0.0, 0.11), vertices, faces)


def test_link_hull_vertices_are_cloud_points_and_enclose_the_supports():
    rng = random.Random(3)
    cloud = [c for _ in range(4000) for c in (rng.gauss(0, 0.1), rng.gauss(0, 0.05), rng.gauss(0, 0.2))]
    vertices, faces = link_hull([cloud[:6000], cloud[6000:]], directions=128)
    points = {tuple(cloud[i:i + 3]) for i in range(0, len(cloud), 3)}
    assert set(vertices) <= points and len(vertices) <= 128
    supports = support_points(cloud, fibonacci_directions(128))
    assert all(_inside(p, vertices, faces, tol=1e-6) for p in supports)
    # Each direction's extreme is reached exactly (the hull is the polytope through them).
    for d in fibonacci_directions(128):
        best = max(cloud[i] * d[0] + cloud[i + 1] * d[1] + cloud[i + 2] * d[2] for i in range(0, len(cloud), 3))
        assert max(v[0] * d[0] + v[1] * d[1] + v[2] * d[2] for v in vertices) == pytest.approx(best)


def test_flat_cloud_has_no_hull():
    with pytest.raises(ValueError):
        convex_hull([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)])


def test_joint_neighbours_counts_joints_including_loops():
    edges = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("d", "e")]
    near = joint_neighbours(edges, 2)
    assert frozenset("ac") in near and frozenset("ae") in near and frozenset("be") not in near
    assert joint_neighbours(edges, 1) == {frozenset(e) for e in edges}


def test_link_hull_export_writes_one_convex_hull_per_link_and_keeps_separate_meshes():
    model = arm_model()
    model.links["link1"].collisions.append(box_mesh("plate", 0.03))
    model.links["link1"].collision_separate = ["plate"]
    stage, report = build(model, ExportOptions(physics_engine=PhysicsEngine.NEWTON,
                                               collision_geometry=CollisionGeometry.LINK_HULL))
    for name, expected in (("base", ["hull"]), ("link1", ["hull", "plate"]), ("link2", ["hull"])):
        scope = stage.GetPrimAtPath(f"/Test_Arm/{name}/collisions")
        colliders = [p for p in scope.GetChildren() if p.HasAPI(UsdPhysics.CollisionAPI)]
        assert sorted(p.GetName() for p in colliders) == expected
        for p in colliders:
            assert UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() == "convexHull"
    hull = UsdGeom.Mesh(stage.GetPrimAtPath("/Test_Arm/link1/collisions/hull"))
    assert len(hull.GetPointsAttr().Get()) == 8 and len(hull.GetFaceVertexCountsAttr().Get()) == 12
    assert not [w for w in report.warnings if "collision" in w or "hull" in w]


def test_filter_joint_distance_filters_links_two_joints_apart_across_a_loop():
    def pairs(stage):
        out = set()
        for p in stage.Traverse():
            if p.HasAPI(UsdPhysics.FilteredPairsAPI):
                out |= {frozenset((p.GetName(), t.name))
                        for t in UsdPhysics.FilteredPairsAPI(p).GetFilteredPairsRel().GetTargets()}
        return out

    model = four_bar_model()
    one = pairs(build(model, NEWTON)[0])
    two = pairs(build(four_bar_model(), ExportOptions(physics_engine=PhysicsEngine.NEWTON,
                                                      filter_joint_distance=2))[0])
    joints = {frozenset((j.parent, j.child)) for j in model.joints}
    assert one == joints
    assert two == joint_neighbours([(j.parent, j.child) for j in model.joints], 2) and two > one


def test_newton_ball_joint_carries_armature_and_friction():
    model = ball_model()
    model.joints[1].properties = JointProperties(armature=1e-4, friction=0.02)
    stage, _ = build(model, NEWTON)
    ball = stage.GetPrimAtPath("/ball_robot/joints/ball_joint")
    assert ball.IsA(UsdPhysics.SphericalJoint)
    assert ball.GetAttribute("newton:armature").Get() == pytest.approx(1e-4)
    assert ball.GetAttribute("newton:friction").Get() == pytest.approx(0.02)


def test_fibonacci_directions_are_unit_and_spread():
    dirs = fibonacci_directions(256)
    assert all(math.isclose(sum(c * c for c in d), 1.0) for d in dirs)
    assert min(max(sum(a * b for a, b in zip(d, e)) for e in dirs if e is not d) for d in dirs) < 0.99


def _collision_summary(path):
    from pxr import Usd

    stage = Usd.Stage.Open(str(path))
    colliders, pairs, balls = {}, set(), {}
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.CollisionAPI):
            pts = sorted(tuple(round(c, 5) for c in v) for v in UsdGeom.Mesh(p).GetPointsAttr().Get())
            colliders[str(p.GetPath())] = (UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get(), pts)
        if p.HasAPI(UsdPhysics.FilteredPairsAPI):
            pairs |= {frozenset((str(p.GetPath()), str(t)))
                      for t in UsdPhysics.FilteredPairsAPI(p).GetFilteredPairsRel().GetTargets()}
        if p.IsA(UsdPhysics.SphericalJoint):
            balls[p.GetName()] = (p.GetAttribute("newton:armature").Get(), p.GetAttribute("newton:friction").Get())
    return colliders, pairs, balls


@pytest.mark.parametrize("make", [arm_model, four_bar_model, ball_model])
def test_rebake_of_a_layered_export_equals_a_link_hull_export(tmp_path, make):
    """Old options plus ``rebake`` give the colliders, filtered pairs and ball dynamics a link-hull export writes."""
    import json

    from fusion2usd.core.report import ExportReport
    from fusion2usd.kinematics import PassContext
    from fusion2usd.kinematics.properties import OverrideSet, PropertiesPass
    from fusion2usd.rebake import rebake
    from fusion2usd.usd.writer import UsdWriter

    overrides = {"links": {"base": {"collision": "none"}, "link1": {"collision_separate": ["c"]}},
                 "joints": {"ball_joint": {"armature": 1e-4, "friction": 0.02}}}
    (tmp_path / "o.json").write_text(json.dumps(overrides))

    def export(name, **kw):
        model = make()
        for link in model.links.values():  # a second mesh per link, so a hull merges something
            link.collisions = (link.collisions or list(link.visuals)) + [box_mesh("extra", 0.08)]
        report = ExportReport()
        opts = ExportOptions(physics_engine=PhysicsEngine.NEWTON, mesh_layer=True, **kw)
        model = PropertiesPass(OverrideSet.load(str(tmp_path / "o.json"))).run(model, PassContext(opts, report))
        if name == "old":  # exports before this change wrote no ball-joint dynamics
            for joint in model.joints:
                joint.properties.armature = joint.properties.friction = None
        out = tmp_path / name / "robot.usda"
        out.parent.mkdir()
        UsdWriter(opts, report).write(model, str(out))
        return out

    old = export("old")
    rebake(str(old), str(tmp_path / "o.json"), 2, 64, None)
    new = export("new", collision_geometry=CollisionGeometry.LINK_HULL, filter_joint_distance=2)
    a, b = _collision_summary(old), _collision_summary(new)
    assert set(a[0]) == set(b[0]) and a[0]
    for path in a[0]:
        assert a[0][path][0] == b[0][path][0] == "convexHull"
        assert a[0][path][1] == pytest.approx(b[0][path][1], abs=1e-5)
    assert a[1] == b[1]
    assert a[2] == pytest.approx(b[2]) if a[2] else a[2] == b[2]
