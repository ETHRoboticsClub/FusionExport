"""Load a Fusion2USD export (physics engine: Newton) in Newton and check that it behaves.

    pip install "newton[sim,importers]"
    python tools/newton_smoke_test.py out/robot.usda --solver mujoco kamino

Runs on CPU. Joints only: no contacts, gravity on. Checks per solver: the model imports and steps
without NaNs, loop-closure joints stay closed, and mimic followers obey q_f = coef0 + coef1 q_l after the
leader is displaced. Exits non-zero on failure.

Solver limits (Newton 1.7): Featherstone does not solve loop-closure joints and Kamino ignores mimic
constraints; those checks are skipped for them. MuJoCo anchors loop closures at joint zero, so export
mechanisms with loops with Joint zero = Current pose. MuJoCo's equality constraints are soft: mimic errors
settle over a few hundred steps.
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings

import numpy as np
import warp as wp

wp.config.log_level = logging.WARNING  # must precede Warp's initialization

import newton  # noqa: E402
from newton.usd import SchemaResolverNewton, SchemaResolverPhysx  # noqa: E402

SOLVERS = {
    "mujoco": newton.solvers.SolverMuJoCo,
    "kamino": newton.solvers.SolverKamino,
    "featherstone": newton.solvers.SolverFeatherstone,
    "xpbd": newton.solvers.SolverXPBD,
}
SOLVES_LOOPS = {"mujoco", "kamino", "xpbd"}
SOLVES_MIMICS = {"mujoco", "featherstone", "xpbd"}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("usd")
parser.add_argument("--solver", nargs="+", choices=sorted(SOLVERS), default=["mujoco", "kamino"])
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--dt", type=float, default=1e-3)
parser.add_argument("--loop_tol", type=float, default=2e-3, help="max loop-closure gap (m)")
parser.add_argument("--mimic_tol", type=float, default=2e-3, help="max mimic error (rad or m)")
parser.add_argument("--mimic_probe", type=float, default=0.2, help="leader displacement (rad or m); 0 disables")


class Checks:
    def __init__(self) -> None:
        self.failed = 0

    def check(self, ok: bool, label: str, detail: str = "") -> None:
        self.failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f": {detail}" if detail else ""), flush=True)


def build(path: str) -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    # Newton only reads JointStateAPI positions through the PhysX resolver; without it the robot starts at
    # joint zero, not at the exported pose. A Newton export carries no other physx* attribute.
    builder.add_usd(path, schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()])
    return builder


def anchor(body_q: np.ndarray, body: int, x: np.ndarray) -> np.ndarray:
    if body < 0:
        return x[:3]
    return body_q[body][:3] + np.array(wp.quat_rotate(wp.quat(*body_q[body][3:]), wp.vec3(*x[:3])))


def run(path: str, solver_name: str, args, checks: Checks) -> None:
    builder = build(path)
    labels = [str(label).rsplit("/", 1)[-1] for label in builder.joint_label]
    leaders = {lead for lead in builder.joint_mimic_joint if lead >= 0}
    q = list(builder.joint_q)
    for j in leaders:
        if args.mimic_probe and solver_name in SOLVES_MIMICS:
            q[builder.joint_q_start[j]] += args.mimic_probe
    builder.joint_q = q
    model = builder.finalize()

    state, next_state, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    try:
        solver = SOLVERS[solver_name](model)
    except Exception as e:  # unsupported model features are reported per solver
        checks.check(False, f"{solver_name}: model accepted", f"{type(e).__name__}: {e}")
        return
    for _ in range(args.steps):
        state.clear_forces()
        solver.step(state, next_state, control, None, args.dt)
        state, next_state = next_state, state

    body_q = state.body_q.numpy()
    checks.check(not np.isnan(body_q).any(), f"{solver_name}: no NaNs after {args.steps} steps")

    art, parent, child = model.joint_articulation.numpy(), model.joint_parent.numpy(), model.joint_child.numpy()
    X_p, X_c = model.joint_X_p.numpy(), model.joint_X_c.numpy()
    for j in np.flatnonzero(art < 0):
        if solver_name not in SOLVES_LOOPS:
            print(f"[SKIP] {solver_name}: loop closure {labels[j]} (solver ignores loop joints)")
            continue
        gap = float(np.linalg.norm(anchor(body_q, parent[j], X_p[j]) - anchor(body_q, child[j], X_c[j])))
        checks.check(gap < args.loop_tol, f"{solver_name}: loop closure {labels[j]} stays closed", f"gap {gap:.2e} m")

    joint_q, joint_qd = wp.zeros_like(model.joint_q), wp.zeros_like(model.joint_qd)
    newton.eval_ik(model, state, joint_q, joint_qd)
    joint_q, starts = joint_q.numpy(), model.joint_q_start.numpy()
    mimic_leader, coeffs = model.joint_mimic_joint.numpy(), model.joint_mimic_coeffs.numpy()
    for f in np.flatnonzero(mimic_leader >= 0):
        if solver_name not in SOLVES_MIMICS:
            print(f"[SKIP] {solver_name}: mimic {labels[f]} (solver ignores mimic constraints)")
            continue
        q_f, q_l = joint_q[starts[f]], joint_q[starts[mimic_leader[f]]]
        err = abs(q_f - (coeffs[f][0] + coeffs[f][1] * q_l))
        checks.check(err < args.mimic_tol, f"{solver_name}: mimic {labels[f]} follows {labels[mimic_leader[f]]}",
                     f"error {err:.2e}")


def main() -> int:
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    try:
        import coacd  # convex decomposition logs every hull it builds
        coacd.set_log_level("error")
    except ImportError:
        pass
    checks = Checks()
    for solver_name in args.solver:
        run(args.usd, solver_name, args, checks)
    print(f"{checks.failed} check(s) failed" if checks.failed else "all checks passed")
    return 1 if checks.failed else 0


if __name__ == "__main__":
    sys.exit(main())
