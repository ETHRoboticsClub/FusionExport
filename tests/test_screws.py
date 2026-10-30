"""tools/screws: hole inventory on synthetic meshes and the Fusion script's placement math."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom  # noqa: E402

import _adsk_stub  # noqa: E402

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "screws"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


inventory = _load("screw_inventory", TOOLS / "inventory.py")


def block(z0, z1, half=10.0, hole_r=None, hole_depth=None, n=24, csk_r=None):
    """Square block [mm] centered on the Z axis, optionally with a hole from the top (blind if hole_depth,
    with a 90 deg countersink if csk_r)."""
    ang = [2 * math.pi * i / n for i in range(n)]
    outer = [(math.cos(a) * half / max(abs(math.cos(a)), abs(math.sin(a))),
              math.sin(a) * half / max(abs(math.cos(a)), abs(math.sin(a)))) for a in ang]
    pts, tris = [], []

    def ring(xy, z):
        start = len(pts)
        pts.extend((x, y, z) for x, y in xy)
        return list(range(start, start + len(xy)))

    def quad(a, b, c, d):  # counter-clockwise seen from outside
        tris.extend([(a, b, c), (a, c, d)])

    ot, ob = ring(outer, z1), ring(outer, z0)
    for i in range(n):
        j = (i + 1) % n
        quad(ob[i], ob[j], ot[j], ot[i])  # outer wall
    if hole_r is None:
        ct, cb = ring([(0, 0)], z1)[0], ring([(0, 0)], z0)[0]
        for i in range(n):
            j = (i + 1) % n
            tris.extend([(ct, ot[i], ot[j]), (cb, ob[j], ob[i])])
        return np.array(pts), np.array(tris)
    inner = [(hole_r * math.cos(a), hole_r * math.sin(a)) for a in ang]
    zb = z0 if hole_depth is None else z1 - hole_depth
    if csk_r is None:
        top = it = ring(inner, z1)
    else:
        top = ring([(csk_r * math.cos(a), csk_r * math.sin(a)) for a in ang], z1)
        it = ring(inner, z1 - (csk_r - hole_r))
    ib = ring(inner, zb)
    for i in range(n):
        j = (i + 1) % n
        quad(top[i], ot[i], ot[j], top[j])  # top annulus
        if csk_r is not None:
            quad(it[i], top[i], top[j], it[j])  # countersink cone
        quad(it[j], ib[j], ib[i], it[i])  # hole wall, facing the axis
    if hole_depth is None:
        for i in range(n):
            j = (i + 1) % n
            quad(ib[j], ob[j], ob[i], ib[i])  # bottom annulus
    else:
        hb, cb = ring([(0, 0)], zb)[0], ring([(0, 0)], z0)[0]
        for i in range(n):
            j = (i + 1) % n
            tris.extend([(hb, ib[i], ib[j]), (cb, ob[j], ob[i])])
    return np.array(pts), np.array(tris)


def signed_volume(pts, tris):
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6


def stage_with(tmp_path, bodies):
    stage = Usd.Stage.CreateNew(str(tmp_path / "robot.usda"))
    UsdGeom.Xform.Define(stage, "/V1/link/visuals")
    for name, (pts, tris) in bodies.items():
        assert signed_volume(pts, tris) > 0
        mesh = UsdGeom.Mesh.Define(stage, f"/V1/link/visuals/{name}")
        mesh.CreatePointsAttr([Gf.Vec3f(*(p / 1000.0)) for p in pts])
        mesh.CreateFaceVertexCountsAttr([3] * len(tris))
        mesh.CreateFaceVertexIndicesAttr(tris.reshape(-1).tolist())
    stage.Save()
    return str(tmp_path / "robot.usda")


def test_clearance_hole_on_blank_part(tmp_path):
    path = stage_with(tmp_path, {"Plate": block(10, 15, hole_r=2.25), "Base": block(0, 10)})
    holes, _, sites = inventory.run(path)
    assert len(holes) == 1
    (s,) = sites
    assert (s["size"], s["kind"], s["confidence"], s["length"]) == ("M4", "thread-not-modeled", "medium", 12)
    assert s["clamp_mm"] == pytest.approx(5.0, abs=0.01)
    assert s["head_m"] == pytest.approx([0, 0, 0.015], abs=1e-5)
    assert s["axis"] == pytest.approx([0, 0, -1], abs=1e-6)


def test_clearance_over_tap_drill(tmp_path):
    path = stage_with(tmp_path, {"Plate": block(10, 15, hole_r=2.25),
                                 "Base": block(0, 10, hole_r=1.65, hole_depth=8)})
    _, _, sites = inventory.run(path)
    (s,) = sites
    assert (s["size"], s["kind"], s["confidence"], s["length"]) == ("M4", "tapped", "high", 12)
    assert s["pattern"] == "4.5x5.0 | 3.3x8.0"
    assert s["head_m"] == pytest.approx([0, 0, 0.015], abs=1e-5)


def test_countersunk_hole(tmp_path):
    path = stage_with(tmp_path, {"Plate": block(10, 15, hole_r=2.25, csk_r=4.5), "Base": block(0, 10)})
    (s,) = inventory.run(path)[2]
    # the bore starts 2.25 mm below the surface; shank 2.75 + 6 mm into the base, head 2.48 -> 12
    assert (s["size"], s["head"], s["length"]) == ("M4", "countersunk", 12)
    assert s["head_m"] == pytest.approx([0, 0, 0.01275], abs=1e-5)
    assert s["note"].startswith("countersunk (ISO 10642)")


def test_blind_clearance_hole_in_merged_body(tmp_path):
    path = stage_with(tmp_path, {"Weldment": block(0, 15, hole_r=2.25, hole_depth=10)})
    (s,) = inventory.run(path)[2]
    assert (s["size"], s["kind"], s["confidence"], s["length"]) == ("M4", "one-body-blind", "medium", 8)
    assert s["head_m"] == pytest.approx([0, 0, 0.015], abs=1e-5)
    assert s["axis"] == pytest.approx([0, 0, -1], abs=1e-6)


def test_lone_hole_is_not_a_site(tmp_path):
    path = stage_with(tmp_path, {"Plate": block(10, 15, hole_r=2.25)})
    assert inventory.run(path)[2] == []


def test_actuator_interface_from_datasheet():
    def site(angle_deg, circle, size, hole_d, axis=(0.6, 0.0, -0.8)):  # tilted axis, mixed signs
        a = np.array(axis)
        u = np.cross(a, [0, 1, 0])
        u /= np.linalg.norm(u)
        v = np.cross(a, u)
        t = math.radians(angle_deg)
        face = circle / 2e3 * (math.cos(t) * u + math.sin(t) * v)
        return dict(mates="hej/Body1", size=size, hole_d_mm=hole_d, clamp_mm=10.0, note="", length=20,
                    kind="thread-not-modeled", confidence="medium", axis=list(a),
                    head_m=list(face - 0.01 * a))

    ring = [site(20 + 30 * i, 50.0, "M6", 6.5) for i in range(12)]
    pin, extra = site(173, 52.0, "M3", 3.5), site(5, 70.0, "M6", 6.5)
    sites = ring + [pin, extra]
    inventory.apply_interfaces(sites)
    assert {(s["kind"], s["confidence"], s["length"]) for s in ring} == {("actuator-thread", "high", 20)}
    assert ring[0]["note"] == "HEJ 90 output: M6, 14 deep"
    assert (pin["kind"], pin["confidence"]) == ("dowel-pin", "excluded")
    assert (extra["kind"], extra["confidence"]) == ("no-thread", "excluded")


def test_std_length():
    assert inventory.std_length(9, 13, 11) == 12
    assert inventory.std_length(16.0001, 19) == 16
    assert inventory.std_length(17, 19) == 20  # nothing in range: next longer


@pytest.fixture
def script():
    _adsk_stub.install()
    try:
        yield _load("add_missing_screws", TOOLS / "AddMissingScrews" / "AddMissingScrews.py")
    finally:
        _adsk_stub.uninstall()


def test_screw_frame_puts_head_underside_on_bearing_point(script):
    axis = (0.0, 0.6, -0.8)
    o, x, y, z = script.screw_frame((1.0, 2.0, 3.0), axis, 0.4)
    assert z == pytest.approx(axis)
    for a, b in ((x, y), (y, z), (z, x)):
        assert script.dot(a, b) == pytest.approx(0, abs=1e-12)
    assert script.cross(x, y) == pytest.approx(z)  # right-handed
    bearing = tuple(oi + 0.4 * zi for oi, zi in zip(o, z))  # local (0, 0, k)
    assert bearing == pytest.approx((1.0, 2.0, 3.0))


def test_head_dims(script):
    assert script.head_dims(dict(size="M4", head="countersunk")) == (8.96, 2.48, 2.5, 1.5)
    assert script.head_dims(dict(size="M4", head="socket")) == (7.0, 4.0, 3.0, 2.0)
    assert script.head_dims(dict(size="M2", head="countersunk")) == (3.8, 2.0, 1.5, 1.0)  # no ISO 10642 M2


def test_select_sites(script):
    sites = [dict(size="M4", confidence="high"), dict(size="M4", confidence="low"),
             dict(size="M3.5", confidence="medium")]
    assert script.select_sites(sites) == sites[:1]
