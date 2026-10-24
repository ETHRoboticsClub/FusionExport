import os
from typing import List

import pytest

from _mechanisms import four_bar, gripper
from fusion2usd.actions import ExportArtifacts, PostExportAction
from fusion2usd.cli import main as cli_main
from fusion2usd.exporter import Exporter, ModelFileSource
from fusion2usd.options import ExportOptions
from fusion2usd.serialization import ModelSerializer


class SummaryAction(PostExportAction):
    name = "summary"

    def run(self, artifacts: ExportArtifacts) -> List[str]:
        path = artifacts.sibling("_summary.txt")
        with open(path, "w") as f:
            f.write(",".join(j.name for j in artifacts.model.joints))
        return [path]


class BrokenAction(PostExportAction):
    name = "broken"

    def run(self, artifacts):
        raise RuntimeError("boom")


def dump(model, tmp_path):
    path = tmp_path / f"{model.name}.model.json"
    ModelSerializer().save(model, str(path))
    return str(path)


def test_serialization_round_trip(tmp_path):
    m = gripper()
    back = ModelSerializer().load(dump(m, tmp_path))
    assert ModelSerializer().to_dict(back) == ModelSerializer().to_dict(m)


@pytest.mark.parametrize("factory", [four_bar, gripper])
def test_end_to_end_with_actions(tmp_path, factory):
    pxr = pytest.importorskip("pxr")
    from pxr import Usd, UsdPhysics

    opts = ExportOptions(output_path=str(tmp_path / "out" / "robot.usda"), write_urdf=True)
    report = Exporter(opts, actions=None).export(ModelFileSource(dump(factory(), tmp_path)))
    out = tmp_path / "out"
    assert (out / "robot.usda").exists() and (out / "robot_cfg.py").exists() and (out / "robot.urdf").exists()
    compile((out / "robot_cfg.py").read_text(), "robot_cfg.py", "exec")

    stage = Usd.Stage.Open(str(out / "robot.usda"))
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert len(roots) == 1
    # fixed base: on the robot Xform; floating base: on the base link itself
    assert roots[0].GetPath() == (stage.GetDefaultPrim().GetPath() if factory is four_bar
                                  else stage.GetDefaultPrim().GetPath().AppendChild("palm"))
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    assert joints
    if factory is four_bar:
        excluded = [p for p in joints if UsdPhysics.Joint(p).GetExcludeFromArticulationAttr().Get()]
        assert len(excluded) == 1
    assert report.outputs[0].endswith("robot.usda")


def test_custom_and_failing_actions(tmp_path):
    opts = ExportOptions(output_path=str(tmp_path / "r.usda"), write_isaaclab_cfg=False)
    report = Exporter(opts, actions=[SummaryAction(), BrokenAction()]).export(ModelFileSource(dump(gripper(), tmp_path)))
    assert (tmp_path / "r_summary.txt").exists()
    assert any("broken" in w and "boom" in w for w in report.warnings)


def test_action_by_module_path_and_unknown(tmp_path):
    opts = ExportOptions(output_path=str(tmp_path / "r.usda"), write_isaaclab_cfg=False,
                         post_actions=["test_exporter:SummaryAction", "does_not_exist"])
    report = Exporter(opts).export(ModelFileSource(dump(gripper(), tmp_path)))
    assert (tmp_path / "r_summary.txt").exists()
    assert any("does_not_exist" in w for w in report.warnings)


def test_cli(tmp_path, capsys):
    src = dump(four_bar(), tmp_path)
    assert cli_main([src, "-o", str(tmp_path / "cli" / "fb.usda"), "--zero-pose", "current", "--action", "urdf"]) == 0
    assert os.path.exists(tmp_path / "cli" / "fb.urdf")
    assert "Wrote:" in capsys.readouterr().out
