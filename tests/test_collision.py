"""Occupied-volume and exporter regression tests for conservative reduction."""
import math
from dataclasses import replace
import xml.etree.ElementTree as ET

import pytest

from _mechanisms import box_mesh
from fusion2usd.collision import convex_planes, contains, fit_primitive, optimize_collisions
from fusion2usd.core.math3d import Quat, Transform, Vec3
from fusion2usd.core.report import ExportReport
from fusion2usd.model.robot import Link, Mesh, RobotModel
from fusion2usd.options import ExportOptions
from fusion2usd.serialization import ModelSerializer
from fusion2usd.usd.builder import UsdRobotBuilder
from fusion2usd.urdf.writer import UrdfWriter


def box(name, lo, hi):
    m = box_mesh(name, 1, material=None)
    m.points = [(lo[i % 3] + hi[i % 3]) / 2 + x * (hi[i % 3] - lo[i % 3]) / 2
                for i, x in enumerate(m.points)]
    return m


def combine(name, meshes):
    points, indices = [], []
    for m in meshes:
        indices += [i + len(points) // 3 for i in m.indices]
        points += m.points
    return Mesh(name, points, indices)


def rounded(name, radius=0.01, height=0.0, rings=16, segments=32):
    # Shared poles/seams and outward triangles, including an equatorial shaft.
    points = [(0, 0, radius + height / 2)]
    rows = []
    for i in range(1, rings):
        theta = math.pi * i / rings
        offsets = [height / 2 if i <= rings // 2 else -height / 2]
        if i == rings // 2 and height:
            offsets.append(-height / 2)
        for offset in offsets:
            row = []
            for j in range(segments):
                phi = 2 * math.pi * j / segments
                row.append(len(points))
                points.append((radius * math.sin(theta) * math.cos(phi),
                               radius * math.sin(theta) * math.sin(phi), radius * math.cos(theta) + offset))
            rows.append(row)
    bottom = len(points)
    points.append((0, 0, -radius - height / 2))
    faces = []
    for j in range(segments):
        k = (j + 1) % segments
        faces += [(0, rows[0][j], rows[0][k]), (bottom, rows[-1][k], rows[-1][j])]
        for a, b in zip(rows, rows[1:]):
            faces += [(a[j], b[j], b[k]), (a[j], b[k], a[k])]
    return Mesh(name, [x for p in points for x in p], [i for f in faces for i in f])


def optimize(meshes, **kwargs):
    model = RobotModel('test', base_link='base')
    model.add_link(Link('base', visuals=meshes))
    report = ExportReport()
    optimize_collisions(model, ExportOptions(**kwargs), report)
    return model, report


def test_duplicates_and_actual_containment_preserve_visuals():
    outer = box('outer', (-1, -1, -1), (1, 1, 1))
    inner = box('inner', (-.1, -.1, -.1), (.1, .1, .1))
    duplicate = replace(outer, name='duplicate', indices=outer.indices[3:] + outer.indices[:3])
    model, _ = optimize([outer, inner, duplicate])
    link = model.links['base']
    assert len(link.collisions) == 1
    assert len(link.visuals) == 3
    assert all(m.primitive is None for m in link.visuals)
    assert link.collisions[0].primitive['kind'] == 'box'


def test_convex_union_merges_different_shared_face_triangulations():
    a = box('a', (0, 0, 0), (1, 1, 1))
    b = box('b', (1, 0, 0), (2, 1, 1))
    model, _ = optimize([a, b])
    result = model.links['base'].collisions
    assert len(result) == 1
    assert sorted(result[0].primitive['size']) == [1, 1, 2]
    assert contains(convex_planes(result[0]), a)
    assert contains(convex_planes(result[0]), b)


def test_gap_and_concave_union_never_bridged():
    a = box('a', (0, 0, 0), (1, 1, 1))
    for b in (box('gap', (1.00001, 0, 0), (2, 1, 1)),
              box('l', (1, 0, 0), (2, .5, 1))):
        model, _ = optimize([a, b])
        assert len(model.links['base'].collisions) == 2


def test_socket_does_not_contain_screw_and_cavity_is_not_fitted():
    # Hollow housing: external shell plus inward inner shell. Its AABB contains
    # the screw, but its occupied material does not. Separate shells fail closed.
    outer = box('outside', (-1, -1, -1), (1, 1, 1))
    hole = box('hole', (-.5, -.5, -.5), (.5, .5, .5))
    hole.indices = [i for j in range(0, len(hole.indices), 3) for i in hole.indices[j:j+3][::-1]]
    socket = combine('socket', [outer, hole])
    screw = box('screw', (-.1, -.1, -.1), (.1, .1, .1))
    assert convex_planes(socket) is None
    model, report = optimize([socket, screw])
    assert len(model.links['base'].collisions) == 2
    assert model.links['base'].collisions[0] is socket
    assert socket.primitive is None
    assert 'socket clearances' in report.summary()


def test_screw_and_socket_on_distinct_links_are_never_compared():
    m = box('body', (-1, -1, -1), (1, 1, 1))
    model, _ = optimize([m])
    model.add_link(Link('moving', visuals=[m]))
    optimize_collisions(model, ExportOptions(), ExportReport())
    assert all(len(link.collisions) == 1 for link in model.links.values())


@pytest.mark.parametrize('height,kind', [(0, 'sphere'), (.04, 'capsule')])
def test_round_primitives_inside_every_support_plane_and_bounded_error(height, kind):
    m = rounded(kind, height=height)
    planes = convex_planes(m)
    assert planes is not None
    shape = fit_primitive(m, planes, .0002)
    assert shape['kind'] == kind
    c = Vec3.of(shape['center'])
    axis = Quat(*shape['rotation']).rotate(Vec3(0, 0, 1))
    for n, a in planes:
        support = n.dot(c - a) + shape['radius'] + shape.get('height', 0) / 2 * abs(n.dot(axis))
        assert support <= 1e-12
    assert 0 <= shape['error'] <= .0002
    assert fit_primitive(m, planes, 0) is None


def test_rotated_box_and_capsule():
    pose = Transform(Quat.from_axis_angle(Vec3(1, 2, 3), .73), Vec3(.2, -.3, .5))
    for m, kind in ((box_mesh('box'), 'box'), (rounded('capsule', height=.04), 'capsule')):
        m = m.transformed(pose)
        planes = convex_planes(m)
        shape = fit_primitive(m, planes, .0002)
        assert shape is not None and shape['kind'] == kind


def test_open_and_nonmanifold_meshes_are_not_certificates():
    closed = box_mesh('box')
    for indices in (closed.indices[:-3], closed.indices + closed.indices[:3],
                    [999] + closed.indices[1:]):
        assert convex_planes(replace(closed, indices=indices)) is None
    open_mesh = replace(closed, indices=closed.indices[:-3])
    model, _ = optimize([open_mesh, box_mesh('inner', .01)])
    assert len(model.links['base'].collisions) == 2
    assert model.links['base'].collisions[0] is open_mesh


def test_disable_and_explicit_approximation_preserved():
    for settings in ({'optimize_collisions': False}, {'collision_approximation': 'sdf'},
                     {'export_collisions': False}):
        model, _ = optimize([box_mesh('box')], **settings)
        assert model.links['base'].collisions == []


@pytest.mark.parametrize('value', [-1, math.nan, math.inf])
def test_invalid_tolerance(value):
    with pytest.raises(ValueError):
        optimize([box_mesh('box')], collision_primitive_tolerance=value)


def test_primitives_roundtrip_and_transform():
    model, _ = optimize([box_mesh('box')])
    restored = ModelSerializer().from_dict(ModelSerializer().to_dict(model))
    before = restored.links['base'].collisions[0]
    assert before.primitive == model.links['base'].collisions[0].primitive
    pose = Transform(Quat.from_axis_angle(Vec3(0, 1, 0), .5), Vec3(1, 2, 3))
    after = before.transformed(pose)
    assert Vec3.of(after.primitive['center']).is_close(pose.apply_point(Vec3.of(before.primitive['center'])))


def test_native_usd_primitives_and_urdf_capsule_union(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics
    meshes = [box_mesh('box'), rounded('sphere').transformed(Transform(translation=Vec3(.2, 0, 0))),
              rounded('capsule', height=.04).transformed(Transform(translation=Vec3(.4, 0, 0)))]
    model, report = optimize(meshes, collision_primitive_tolerance=.0002)
    opts = ExportOptions()
    layer = Sdf.Layer.CreateAnonymous('.usda')
    assert layer.ImportFromString(UsdRobotBuilder(opts, report).build(model).to_string())
    stage = Usd.Stage.Open(layer)
    collisions = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert sorted(p.GetTypeName() for p in collisions) == ['Capsule', 'Cube', 'Sphere']
    for p in collisions:
        assert not p.HasAPI(UsdPhysics.MeshCollisionAPI)
        assert UsdGeom.Boundable(p).GetExtentAttr().Get()
    path = tmp_path / 'test.urdf'
    UrdfWriter(opts, report).write(model, str(path))
    root = ET.parse(path).getroot()
    shapes = root.findall('link/collision/geometry')
    assert sorted(next(iter(s)).tag for s in shapes) == ['box', 'cylinder', 'sphere', 'sphere', 'sphere']


def through_socket():
    # A single watertight genus-one shell, unlike the nested-shell cavity above.
    rings = []
    points = []
    for z, radius in ((-1, 1), (1, 1), (-1, .5), (1, .5)):
        rings.append(list(range(len(points), len(points) + 4)))
        points += [(-radius, -radius, z), (radius, -radius, z),
                   (radius, radius, z), (-radius, radius, z)]
    ob, ot, ib, it = rings
    faces = []
    for i in range(4):
        j = (i + 1) % 4
        for a, b, c, d in ((ob[i], ob[j], ot[j], ot[i]), (ib[j], ib[i], it[i], it[j]),
                           (ot[i], ot[j], it[j], it[i]), (ob[j], ob[i], ib[i], ib[j])):
            faces.extend((a, b, c, a, c, d))
    return Mesh('through_socket', [x for p in points for x in p], faces)


def test_connected_through_socket_never_filled_or_used_as_container():
    socket = through_socket()
    screw = box('screw', (-.1, -.1, -1.1), (.1, .1, 1.1))
    assert convex_planes(socket) is None
    model, _ = optimize([socket, screw])
    result = model.links['base'].collisions
    assert len(result) == 2
    assert result[0] is socket and result[0].primitive is None


@pytest.mark.parametrize('mesh_layer', [False, True])
def test_export_pipeline_enables_reduction_and_preserves_mass(tmp_path, mesh_layer):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdPhysics
    from _mechanisms import link
    from fusion2usd.exporter import Exporter, ModelFileSource
    model = RobotModel('robot', base_link='base')
    body = link('base', 0)
    body.visuals.append(box_mesh('internal', .01))
    model.add_link(body)
    src = tmp_path / 'source.model.json'
    ModelSerializer().save(model, str(src))
    dest = tmp_path / 'robot.usda'
    report = Exporter(ExportOptions(output_path=str(dest), mesh_layer=mesh_layer,
                                    write_urdf=True, write_isaaclab_cfg=False)).export(ModelFileSource(str(src)))
    assert not report.errors
    stage = Usd.Stage.Open(str(dest))
    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(colliders) == 1 and colliders[0].GetTypeName() == 'Cube'
    rigid = next(p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI))
    assert UsdPhysics.MassAPI(rigid).GetMassAttr().Get() == pytest.approx(body.mass.mass)
    assert '1 contained' in report.summary()
