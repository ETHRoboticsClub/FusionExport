"""Load a Fusion2USD export in Isaac Lab 3.x and check that it behaves.

    uv run --extra isaacsim python tools/isaaclab_smoke_test.py --cfg out/robot_cfg.py   # from an Isaac Lab 3 checkout

Checks: spawn/reset pose matches init_state, per-env placement of cloned robots (fixed bases stay at
their env origin), no NaNs while holding the default pose, loop-closure joints stay closed, and mimic
joints follow their leader with the authored gearing/offset. Exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cfg", required=True, help="the *_cfg.py written by the exporter")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--loop_tol", type=float, default=2e-3, help="max loop-closure gap (m)")
parser.add_argument("--mimic_tol", type=float, default=2e-3, help="max mimic error (rad or m)")
parser.add_argument("--mimic_offset_probe", type=float, default=10.0,
                    help="offset authored on angular mimic followers before playing; a wrong unit "
                         "(rad vs deg) then shows up as a large error. 0 disables")
parser.add_argument("--enable_cameras", action="store_true", help="render exported cameras (Isaac Lab 3 kwarg)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args, enable_cameras=args.enable_cameras).app

import torch  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402


class Checks:
    def __init__(self) -> None:
        self.failed = 0

    def check(self, ok: bool, label: str, detail: str = "") -> None:
        self.failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f": {detail}" if detail else ""), flush=True)


def load_exported_cfg(path: str):
    spec = importlib.util.spec_from_file_location("exported_cfg", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfgs = [v for k, v in vars(module).items() if k.endswith("_CFG") and isinstance(v, ArticulationCfg)]
    if len(cfgs) != 1:
        raise SystemExit(f"expected exactly one *_CFG in {path}, found {len(cfgs)}")
    return cfgs[0], getattr(module, "sensors", None)


def T(x):
    """Isaac Lab 3 data properties are ProxyArrays; get the torch view."""
    return x.torch if hasattr(x, "torch") else x


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v by (x, y, z, w) quaternion q (Isaac Lab 3 order)."""
    xyz, w = q[..., :3], q[..., 3:]
    v = v.expand_as(xyz)
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def main() -> int:
    robot_cfg, sensors_factory = load_exported_cfg(args.cfg)
    # Exports default to zero drive gains; the test needs leaders that actually move.
    for actuator in robot_cfg.actuators.values():
        if not actuator.stiffness:
            actuator.stiffness, actuator.damping = 200.0, 20.0

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
        light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
        # gives cameras a horizon to look at (an empty world renders a uniform image)
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                              init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.0)))

    scene_cfg = SceneCfg(num_envs=args.num_envs, env_spacing=3.0)
    sensor_cfgs = sensors_factory() if sensors_factory else {}
    if not args.enable_cameras:
        sensor_cfgs = {k: v for k, v in sensor_cfgs.items() if type(v).__name__ != "CameraCfg"}
    for name, cfg in sensor_cfgs.items():
        setattr(scene_cfg, name, cfg)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 200.0, device=args.device))
    scene = InteractiveScene(scene_cfg)
    if args.mimic_offset_probe:
        for prim in sim.stage.Traverse():
            for schema in prim.GetAppliedSchemas():
                # angular followers only (prismatic followers also use rot* tokens; an offset there is meters)
                if schema.startswith("PhysxMimicJointAPI:") and prim.IsA(UsdPhysics.RevoluteJoint):
                    prim.GetAttribute(f"physxMimicJoint:{schema.split(':', 1)[1]}:offset").Set(args.mimic_offset_probe)
    sim.reset()
    robot = scene["robot"]
    checks = Checks()
    print(f"bodies: {robot.body_names}\njoints: {robot.joint_names}", flush=True)
    checks.check(robot.num_joints > 0 or robot.num_bodies == 1, "articulation has joints", str(robot.num_joints))

    # -- spawn and reset pose
    init = robot_cfg.init_state
    origins = scene.env_origins
    expected_pos = origins + torch.tensor(init.pos, device=origins.device)
    expected_rot = torch.tensor(init.rot, device=origins.device).expand(args.num_envs, 4)

    def pose_error() -> tuple[float, float]:
        pos_err = (T(robot.data.root_pos_w) - expected_pos).norm(dim=-1).max().item()
        dot = (T(robot.data.root_quat_w) * expected_rot).sum(dim=-1).abs().clamp(max=1.0)
        return pos_err, (2.0 * torch.acos(dot)).max().item()

    scene.update(sim.get_physics_dt())
    p, r = pose_error()
    checks.check(p < 1e-3 and r < 1e-3, "spawn pose == init_state + env origin", f"pos {p:.2e} m, rot {r:.2e} rad")

    root_pose = T(robot.data.default_root_pose).clone()
    root_pose[:, :3] += origins
    robot.write_root_pose_to_sim_index(root_pose=root_pose)
    robot.write_root_velocity_to_sim_index(root_velocity=T(robot.data.default_root_vel).clone())
    robot.write_joint_position_to_sim_index(position=T(robot.data.default_joint_pos).clone())
    robot.write_joint_velocity_to_sim_index(velocity=T(robot.data.default_joint_vel).clone())
    scene.reset()
    default_q = T(robot.data.default_joint_pos).clone()

    # -- hold the default pose
    for _ in range(args.steps):
        robot.set_joint_position_target_index(target=default_q)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
    finite = torch.isfinite(T(robot.data.body_pos_w)).all().item() and torch.isfinite(T(robot.data.joint_pos)).all().item()
    checks.check(finite, "simulation stays finite")
    if robot.is_fixed_base:
        p, r = pose_error()
        checks.check(p < 1e-3 and r < 1e-3, "fixed base stays at its env origin (per clone)",
                     f"pos {p:.2e} m, rot {r:.2e} rad")
    spread = (T(robot.data.body_pos_w) - T(robot.data.root_pos_w).unsqueeze(1)).std(dim=0).max().item()
    checks.check(spread < 1e-3, "all clones behave identically", f"{spread:.2e} m")

    stage = sim.stage

    # -- loop closures: both joint frames must coincide in world space
    body_index = {name: i for i, name in enumerate(robot.body_names)}
    env0_prefix = "/World/envs/env_0/Robot"
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(env0_prefix) or not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        if not joint.GetExcludeFromArticulationAttr().Get():
            continue
        b0, b1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if not b0 or not b1:
            continue
        i0, i1 = body_index[b0[0].name], body_index[b1[0].name]
        lp0 = torch.tensor(list(joint.GetLocalPos0Attr().Get()), device=origins.device)
        lp1 = torch.tensor(list(joint.GetLocalPos1Attr().Get()), device=origins.device)
        pos, quat = T(robot.data.body_pos_w), T(robot.data.body_quat_w)
        w0 = pos[:, i0] + quat_rotate(quat[:, i0], lp0)
        w1 = pos[:, i1] + quat_rotate(quat[:, i1], lp1)
        gap = (w0 - w1).norm(dim=-1).max().item()
        checks.check(gap < args.loop_tol, f"loop closure {prim.GetName()} stays closed", f"gap {gap:.2e} m")

    # -- mimic joints: move the leader, the follower must obey q_f + gearing*q_l + offset = 0 (USD units)
    joint_index = {name: i for i, name in enumerate(robot.joint_names)}
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(env0_prefix):
            continue
        for schema in prim.GetAppliedSchemas():
            if not schema.startswith("PhysxMimicJointAPI:"):
                continue
            inst = schema.split(":", 1)[1]
            ns = f"physxMimicJoint:{inst}"
            leader = prim.GetRelationship(f"{ns}:referenceJoint").GetTargets()[0].name
            gearing = prim.GetAttribute(f"{ns}:gearing").Get()
            offset = prim.GetAttribute(f"{ns}:offset").Get()
            lf, ll = joint_index[prim.GetName()], joint_index[leader]
            leader_prim = prim.GetRelationship(f"{ns}:referenceJoint").GetTargets()[0]
            leader_revolute = stage.GetPrimAtPath(leader_prim).IsA(UsdPhysics.RevoluteJoint)
            # USD units: degrees for revolute joints, meters for prismatic ones (joint type, not instance name)
            f_scale = 180.0 / math.pi if prim.IsA(UsdPhysics.RevoluteJoint) else 1.0
            l_scale = 180.0 / math.pi if leader_revolute else 1.0
            lo, hi = T(robot.data.soft_joint_pos_limits)[0, ll].tolist()
            step = 0.3 if leader_revolute else 0.01
            target = default_q.clone()
            target[:, ll] = min(max(target[0, ll].item() + step, lo), hi)
            q_f_start = T(robot.data.joint_pos)[0, lf].item()
            for _ in range(args.steps):
                robot.set_joint_position_target_index(target=target)
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim.get_physics_dt())
            q_l = T(robot.data.joint_pos)[0, ll].item()
            q_f = T(robot.data.joint_pos)[0, lf].item()
            expected = -(gearing * q_l * l_scale + offset) / f_scale
            err = abs(q_f - expected)
            # an inactive mimic leaves the follower where it started, which can be close to "expected" by luck
            moved = abs(q_f - q_f_start) > 0.5 * abs(expected - q_f_start) or abs(expected - q_f_start) < args.mimic_tol
            checks.check(err < args.mimic_tol and moved, f"mimic {prim.GetName()} follows {leader}",
                         f"q_f={q_f:.4f} (start {q_f_start:.4f}), expected {expected:.4f} "
                         f"(gearing {gearing:g}, offset {offset:g})")

    # -- sensors from the exported cfg's sensors() factory
    for name, cfg in sensor_cfgs.items():
        sensor = scene[name]
        kind = type(cfg).__name__
        if kind == "CameraCfg":
            rgb = T(sensor.data.output["rgb"]).float()
            shape_ok = tuple(rgb.shape[:3]) == (args.num_envs, cfg.height, cfg.width)
            checks.check(shape_ok and rgb.std().item() > 0.0, f"camera {name} renders",
                         f"shape {tuple(rgb.shape)}, mean {rgb.mean().item():.1f}, std {rgb.std().item():.2f}")
            # Isaac Lab's ROS-convention forward (+Z) must match the exported USD camera's -Z axis
            prim = stage.GetPrimAtPath(f"{env0_prefix}/{cfg.prim_path.split('/Robot/', 1)[1]}")
            m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            usd_fwd = torch.tensor(list(m.TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized()), device=origins.device)
            ros_fwd = quat_rotate(T(sensor.data.quat_w_ros)[0], torch.tensor([0.0, 0.0, 1.0], device=origins.device))
            ang = math.degrees(math.acos(max(-1.0, min(1.0, torch.dot(usd_fwd.float(), ros_fwd.float()).item()))))
            checks.check(ang < 1.0, f"camera {name} optical axis matches Isaac Lab ROS convention", f"{ang:.2f} deg")
        else:
            values = [T(getattr(sensor.data, a)) for a in ("lin_acc_b", "ang_vel_b", "net_normal_forces_w")
                      if getattr(sensor.data, a, None) is not None]
            checks.check(bool(values) and all(torch.isfinite(v).all().item() for v in values),
                         f"{kind.replace('Cfg', '').lower()} sensor {name} reports finite data")

    print(f"\n{'ALL CHECKS PASSED' if checks.failed == 0 else f'{checks.failed} CHECK(S) FAILED'}", flush=True)
    return 1 if checks.failed else 0


if __name__ == "__main__":
    code = main()
    app.close(exit_code=code)  # Isaac Lab 3: close() exits the process with this status
    sys.exit(code)
