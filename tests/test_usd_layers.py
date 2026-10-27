import pytest

from _mechanisms import gripper
from fusion2usd.exporter import Exporter, ModelFileSource
from fusion2usd.options import ExportOptions
from fusion2usd.serialization import ModelSerializer
from fusion2usd.usd import writer as usd_writer
from fusion2usd.usd.usda import UsdaStage
from fusion2usd.usd_support import pick_wheel


def test_split_off_moves_meshes_under_overs():
    stage = UsdaStage("R")
    link = stage.define("R", "Xform").define("link", "Xform")
    link.attr("physics:mass", "float", 1.0)
    link.define("visuals", "Xform").define("body", "Mesh").attr("points", "point3f[]", [0, 0, 0])
    stage.define("R2", "Xform")

    layer = stage.split_off("Mesh")
    assert stage.get("/R/link/visuals/body") is None and stage.get("/R/link/visuals") is not None
    assert [p.specifier for p in (layer.get("/R"), layer.get("/R/link"), layer.get("/R/link/visuals"))] == ["over"] * 3
    assert layer.get("/R/link/visuals/body").type_name == "Mesh" and layer.get("/R2") is None
    assert layer.get("/R/link").get("physics:mass") is None

    stage.sublayers.append("./r_meshes.usdc")
    assert "subLayers = [@./r_meshes.usdc@]" in stage.to_string()


def _export(tmp_path, name, **kw):
    tmp_path.mkdir(exist_ok=True)
    src = tmp_path / "g.model.json"
    ModelSerializer().save(gripper(), str(src))
    opts = ExportOptions(output_path=str(tmp_path / name), write_isaaclab_cfg=False, **kw)
    return Exporter(opts, actions=None).export(ModelFileSource(str(src)))


def _snapshot(path):
    from pxr import Usd

    stage = Usd.Stage.Open(str(path))
    return {str(p.GetPath()): (p.GetTypeName(), sorted(p.GetAppliedSchemas()),
                               {a.GetName(): str(a.Get()) for a in p.GetAttributes() if a.HasAuthoredValue()},
                               {r.GetName(): [str(t) for t in r.GetTargets()] for r in p.GetRelationships()})
            for p in stage.Traverse()}


def test_mesh_layer_composes_to_the_single_file_stage(tmp_path):
    pytest.importorskip("pxr")
    _export(tmp_path / "single", "robot.usda")
    report = _export(tmp_path / "layered", "robot.usda", mesh_layer=True)
    root, meshes = tmp_path / "layered" / "robot.usda", tmp_path / "layered" / "robot_meshes.usdc"
    assert report.outputs[:2] == [str(root), str(meshes)]
    assert "Mesh" not in root.read_text() and meshes.read_bytes().startswith(b"PXR-USDC")
    assert _snapshot(root) == _snapshot(tmp_path / "single" / "robot.usda")


def test_without_pxr_binary_output_falls_back_to_text(tmp_path, monkeypatch):
    monkeypatch.setattr(usd_writer, "has_pxr", lambda: False)
    report = _export(tmp_path, "robot.usdc", mesh_layer=True)
    assert (tmp_path / "robot.usda").exists() and (tmp_path / "robot_meshes.usda").exists()
    assert "subLayers = [@./robot_meshes.usda@]" in (tmp_path / "robot.usda").read_text()
    assert sum("pxr is not available" in w for w in report.warnings) == 2


FILES = [{"filename": f"usd_core-26.11-{py}-none-{tag}.whl", "packagetype": "bdist_wheel"} for py, tag in [
    ("cp313", "macosx_10_15_universal2"), ("cp314", "manylinux_2_27_aarch64.manylinux_2_28_aarch64"),
    ("cp314", "manylinux_2_27_x86_64.manylinux_2_28_x86_64"), ("cp314", "win_amd64"),
    ("cp314", "macosx_10_15_universal2"),
]]


@pytest.mark.parametrize("system, machine, tag", [
    ("Darwin", "arm64", "universal2"), ("Windows", "AMD64", "win_amd64"),
    ("Linux", "x86_64", "x86_64"), ("Linux", "aarch64", "aarch64"),
])
def test_pick_wheel(system, machine, tag):
    wheel = pick_wheel(FILES, "cp314", system, machine)
    assert "cp314" in wheel["filename"] and wheel["filename"].endswith(tag + ".whl")
    assert pick_wheel(FILES, "cp315", system, machine) is None


def test_collisions_reference_the_visual_meshes(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    model = gripper()
    for link in model.links.values():
        link.collisions = []
    src = tmp_path / "g.model.json"
    ModelSerializer().save(model, str(src))
    opts = ExportOptions(output_path=str(tmp_path / "r.usda"), write_isaaclab_cfg=False)
    Exporter(opts, actions=None).export(ModelFileSource(str(src)))

    text = (tmp_path / "r.usda").read_text()
    assert text.count("point3f[] points") == sum(len(link.visuals) for link in model.links.values())
    stage = Usd.Stage.Open(str(tmp_path / "r.usda"))
    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert colliders
    for prim in colliders:
        visual = stage.GetPrimAtPath(str(prim.GetPath()).replace("/collisions/", "/visuals/"))
        assert UsdGeom.Mesh(prim).GetPointsAttr().Get() == UsdGeom.Mesh(visual).GetPointsAttr().Get()
        assert UsdGeom.Imageable(prim).ComputePurpose() == UsdGeom.Tokens.guide
