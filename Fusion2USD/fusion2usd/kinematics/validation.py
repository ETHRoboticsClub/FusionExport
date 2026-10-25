"""Final sanity checks and user-facing warnings before writing."""

from __future__ import annotations

import fnmatch
import os
from typing import List, Mapping, Optional

from ..model.robot import DofType, Joint, RobotModel
from ..options import PhysicsEngine
from . import roles
from .pipeline import ModelPass, PassContext

ALLOW_CONTINUOUS_ENV = "ALLOW_CONTINUOUS"
DISABLE_CONTINUOUS_WARN_ENV = "DISABLE_CONTINUOUS_WARN"

# Properties that silently fall back to export defaults when unset.
_DEFAULTED = ("effort", "velocity", "stiffness", "damping")


class MassSanitizePass(ModelPass):
    """Simulators reject zero/negative mass and inertia; clamp them and say so."""

    def run(self, model: RobotModel, ctx: PassContext) -> RobotModel:
        opts = ctx.options
        for link in model.links.values():
            if link.mass is None:
                if not link.is_virtual:
                    ctx.report.warn(f"Link {link.name!r} has no mass properties; the simulator derives them from its colliders")
                continue
            if link.mass.mass < opts.min_mass:
                ctx.report.warn(f"Link {link.name!r} mass {link.mass.mass:.3g} kg clamped to {opts.min_mass:g} kg")
            link.mass = link.mass.sanitized(opts.min_mass, opts.min_inertia)
        return model


class ValidationPass(ModelPass):
    def __init__(self, environ: Optional[Mapping[str, str]] = None) -> None:
        self.environ = os.environ if environ is None else environ

    def run(self, model: RobotModel, ctx: PassContext) -> RobotModel:
        allowed = self._allowed_continuous()
        warn_continuous = not self._flag(DISABLE_CONTINUOUS_WARN_ENV)
        follower_joints = {c.follower.joint for c in model.couplings} if ctx.options.export_mimic_joints else set()
        for joint in model.joints:
            # Newton writes ball joints as spherical joints, whose limits are not exported anyway
            newton_ball = joint.is_ball and ctx.options.physics_engine is PhysicsEngine.NEWTON
            if warn_continuous and not newton_ball:
                self._check_continuous(joint, allowed, ctx)
            if roles.is_driven(joint, follower_joints, ctx.options):
                self._check_defaults(joint, ctx)
        return model

    # -- continuous joints
    def _allowed_continuous(self) -> List[str]:
        raw = self.environ.get(ALLOW_CONTINUOUS_ENV, "")
        return [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]

    def _flag(self, name: str) -> bool:
        return self.environ.get(name, "").strip().lower() not in ("", "0", "false", "no", "off")

    @staticmethod
    def _matches(joint: Joint, patterns: List[str]) -> bool:
        names = {joint.name, joint.source, joint.metadata.get("fusion_name", ""), joint.metadata.get("decomposed_from", "")}
        return any(fnmatch.fnmatchcase(n, p) for p in patterns for n in names if n)

    def _check_continuous(self, joint: Joint, allowed: List[str], ctx: PassContext) -> None:
        open_dofs = [d for d in joint.dofs if d.type is DofType.ROTATION
                     and not (d.limit is not None and d.limit.lower is not None and d.limit.upper is not None)]
        if open_dofs and not self._matches(joint, allowed):
            one_sided = all(d.limit is not None and d.limit.is_bounded for d in open_dofs)
            kind = "only limited on one side" if one_sided else "continuous (no limits)"
            ctx.report.warn(
                f"Joint {joint.name!r} is {kind}. Set limits in Fusion, or add it to "
                f"${ALLOW_CONTINUOUS_ENV} (or set ${DISABLE_CONTINUOUS_WARN_ENV}=1) if intended"
            )

    # -- default values
    @staticmethod
    def _check_defaults(joint: Joint, ctx: PassContext) -> None:
        missing = [f for f in _DEFAULTED if getattr(joint.properties, f) is None]
        if missing:
            ctx.report.warn(f"Joint {joint.name!r} uses default values for: {', '.join(missing)}")
