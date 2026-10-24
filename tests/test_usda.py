from __future__ import annotations

import math

import pytest

from fusion2usd.core.math3d import Quat
from fusion2usd.usd.usda import UsdaStage, format_value, quote

pxr = pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd  # noqa: E402


def _open(stage: UsdaStage) -> Usd.Stage:
    layer = Sdf.Layer.CreateAnonymous(".usda")
    assert layer.ImportFromString(stage.to_string()), stage.to_string()
    return Usd.Stage.Open(layer)


def test_format_scalars():
    assert format_value("float", math.inf) == "inf"
    assert format_value("double", -math.inf) == "-inf"
    assert format_value("bool", True) == "true"
    assert format_value("quatf", Quat(1, 0, 0, 0)) == "(1, 0, 0, 0)"
    assert format_value("double3", (0.1, 2, 3)) == "(0.1, 2.0, 3.0)"
    assert format_value("point3f[]", [0, 1, 2, 3, 4, 5]) == "[(0, 1, 2), (3, 4, 5)]"
    assert format_value("int[]", []) == "[]"
    assert quote('a "b"\\\n') == '"a \\"b\\"\\\\\\n"'


def test_round_trip_edge_cases():
    s = UsdaStage(default_prim="Root", doc='doc with "quotes"\nand newline')
    root = s.define("Root", "Xform")
    root.metadata["kind"] = "component"
    root.apply("PhysicsRigidBodyAPI", "PhysicsMassAPI")
    root.custom_data.update({
        "name": "Grüße 日本 \"q\" \\ back",
        "n": 3, "big": 2 ** 40, "x": 1.5, "flag": False,
        "nested": {"deep": {"v": [1, 2]}, "empty": {}},
        "floats": [1, 2.5], "strings": ["a", 3, None], "empty_list": [],
        "weird key!": "ok", "none": None,
    })
    root.attr("lo", "float", -math.inf)
    root.attr("hi", "float", math.inf)
    root.attr("pts", "point3f[]", [])
    root.attr("idx", "int[]", [])
    root.attr("tok", "uniform token", "guide")
    root.attr("s", "string", 'x "y" \\ z ü')
    root.attr("asset", "asset", "./tex.png")
    root.attr("q", "quatd", Quat(0.5, 0.5, 0.5, 0.5))
    root.attr("mat", "matrix4d", ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (1, 2, 3, 1)))
    root.attr("n3", "normal3f[]", [0.0, 0.0, 1.0], metadata={"interpolation": "vertex"})
    root.attr("cust", "double", 0.1, custom=True)
    child = root.define("Child", "Scope")
    child.rel("r", [root.path])
    child.rel("many", ["/Root", "/Root/Child"])
    child.rel("none")

    stage = _open(s)
    assert stage.GetDefaultPrim().GetPath() == Sdf.Path("/Root")
    assert "quotes" in stage.GetRootLayer().documentation
    p = stage.GetPrimAtPath("/Root")
    assert p.GetMetadata("kind") == "component"
    assert list(p.GetPrimTypeInfo().GetAppliedAPISchemas()) == ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    cd = p.GetCustomData()
    assert cd["name"] == "Grüße 日本 \"q\" \\ back"
    assert cd["big"] == 2 ** 40 and cd["n"] == 3 and cd["flag"] is False
    assert list(cd["nested"]["deep"]["v"]) == [1, 2]
    assert list(cd["floats"]) == [1.0, 2.5] and list(cd["strings"]) == ["a", "3"]
    assert cd["weird key!"] == "ok" and "none" not in cd
    assert p.GetAttribute("lo").Get() == -math.inf and p.GetAttribute("hi").Get() == math.inf
    assert len(p.GetAttribute("pts").Get()) == 0
    assert p.GetAttribute("tok").GetVariability() == Sdf.VariabilityUniform
    assert p.GetAttribute("s").Get() == 'x "y" \\ z ü'
    assert p.GetAttribute("asset").Get().path == "./tex.png"
    assert p.GetAttribute("q").Get() == Gf.Quatd(0.5, 0.5, 0.5, 0.5)
    assert p.GetAttribute("mat").Get().ExtractTranslation() == Gf.Vec3d(1, 2, 3)
    assert p.GetAttribute("n3").GetMetadata("interpolation") == "vertex"
    assert p.GetAttribute("cust").IsCustom()
    c = stage.GetPrimAtPath("/Root/Child")
    assert c.GetRelationship("r").GetTargets() == [Sdf.Path("/Root")]
    assert len(c.GetRelationship("many").GetTargets()) == 2
    assert c.GetRelationship("none").GetTargets() == []


def test_large_array_precision():
    s = UsdaStage(default_prim="R")
    pts = [i * 0.001 for i in range(30000)]
    s.define("R", "Mesh").attr("points", "point3f[]", pts)
    stage = _open(s)
    got = stage.GetPrimAtPath("/R").GetAttribute("points").Get()
    assert len(got) == 10000
    assert got[9999] == Gf.Vec3f(*pts[-3:])


def test_save(tmp_path):
    s = UsdaStage(default_prim="R")
    s.define("R", "Xform")
    path = tmp_path / "x.usda"
    s.save(str(path))
    assert Usd.Stage.Open(str(path)).GetDefaultPrim().IsValid()
