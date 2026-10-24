from __future__ import annotations

import ast
import os
import sys
import types

import pytest

from _usd_models import ALL_MODELS, arm_model, four_bar_model, mimic_model
from fusion2usd.core.report import ExportReport
from fusion2usd.isaaclab.cfg_writer import IsaacLabCfgWriter
from fusion2usd.options import DriveDefaults, ExportOptions


class _Cfg:
    """Records constructor kwargs in place of Isaac Lab config classes."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.fixture
def fake_isaaclab(monkeypatch):
    sim = types.ModuleType("isaaclab.sim")
    for name in ("UsdFileCfg", "RigidBodyPropertiesCfg", "ArticulationRootPropertiesCfg"):
        setattr(sim, name, type(name, (_Cfg,), {}))
    actuators = types.ModuleType("isaaclab.actuators")
    actuators.ImplicitActuatorCfg = type("ImplicitActuatorCfg", (_Cfg,), {})
    art = types.ModuleType("isaaclab.assets.articulation")
    art.ArticulationCfg = type("ArticulationCfg", (_Cfg,), {"InitialStateCfg": type("InitialStateCfg", (_Cfg,), {})})
    pkg = types.ModuleType("isaaclab")
    pkg.sim = sim
    for name, mod in {"isaaclab": pkg, "isaaclab.sim": sim, "isaaclab.actuators": actuators,
                      "isaaclab.assets": types.ModuleType("isaaclab.assets"),
                      "isaaclab.assets.articulation": art}.items():
        monkeypatch.setitem(sys.modules, name, mod)


def load(model, tmp_path, options=None, usd_name="usd/robot.usd"):
    cfg_path = tmp_path / "robot_cfg.py"
    IsaacLabCfgWriter(options or ExportOptions(), ExportReport()).write(model, str(tmp_path / usd_name), str(cfg_path))
    source = cfg_path.read_text()
    ast.parse(source)
    ns = {"__file__": str(cfg_path)}
    exec(compile(source, str(cfg_path), "exec"), ns)
    return ns, source


@pytest.mark.parametrize("make", ALL_MODELS)
def test_valid_python(make, tmp_path):
    source = IsaacLabCfgWriter(ExportOptions(), ExportReport()).render(make(), "a/b.usd", str(tmp_path / "c.py"))
    compile(source, "cfg.py", "exec")


def test_arm_cfg(fake_isaaclab, tmp_path):
    ns, _ = load(arm_model(), tmp_path, ExportOptions(drive=DriveDefaults(stiffness=20.0, damping=1.0)))
    cfg = ns["TEST_ARM_CFG"]
    assert os.path.normpath(cfg.spawn.usd_path) == str(tmp_path / "usd" / "robot.usd")
    assert cfg.spawn.articulation_props.enabled_self_collisions is False
    assert cfg.spawn.articulation_props.fix_root_link is None
    assert cfg.init_state.joint_pos == {"shoulder": 0.3, "elbow": -0.5}
    groups = list(cfg.actuators.values())
    assert len(groups) == 2
    sh = next(g for g in groups if g.joint_names_expr == ["shoulder"])
    assert (sh.effort_limit_sim, sh.velocity_limit_sim, sh.stiffness, sh.damping, sh.armature) == \
        (10.0, 3.0, 100.0, 5.0, 0.01)
    el = next(g for g in groups if g.joint_names_expr == ["elbow"])
    assert (el.stiffness, el.damping) == (20.0, 1.0)
    assert not hasattr(el, "effort_limit_sim") and not hasattr(el, "velocity_limit_sim")


def test_grouping_and_passive(fake_isaaclab, tmp_path):
    model = four_bar_model()
    ns, source = load(model, tmp_path)
    cfg = ns["FOUR_BAR_CFG"]
    assert set(cfg.init_state.joint_pos) == {"crank_joint", "coupler_joint", "rocker_joint"}
    assert [g.joint_names_expr for g in cfg.actuators.values()] == [["crank_joint"]]
    assert "#   coupler_joint" in source and "loop_joint" not in source

    ns, source = load(mimic_model(), tmp_path)
    cfg = ns["RACK_PINION_CFG"]
    assert cfg.init_state.joint_pos == {"wheel_joint": 0.4, "rack_joint": 0.02}
    assert [g.joint_names_expr for g in cfg.actuators.values()] == [["wheel_joint"]]
    assert "rack_joint (mimic)" in source


def test_identical_joints_share_group(fake_isaaclab, tmp_path):
    model = arm_model()
    model.joint("elbow").properties = model.joint("shoulder").properties
    ns, _ = load(model, tmp_path)
    groups = list(ns["TEST_ARM_CFG"].actuators.values())
    assert [g.joint_names_expr for g in groups] == [["shoulder", "elbow"]]
