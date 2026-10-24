"""Chooses which configuration is joint position zero in the USD."""

from __future__ import annotations

from ..model.robot import RobotModel
from ..options import ZeroPose
from .pipeline import ModelPass, PassContext


class ZeroPosePass(ModelPass):
    """In ``current`` mode the exported pose becomes q = 0: limits, targets and coupling offsets shift."""

    def run(self, model: RobotModel, ctx: PassContext) -> RobotModel:
        if ctx.options.zero_pose is not ZeroPose.CURRENT:
            return model
        shifts = {}
        for joint in model.joints:
            for i, dof in enumerate(joint.dofs):
                q0 = dof.value
                shifts[(joint.name, i)] = q0
                if dof.limit is not None:
                    dof.limit = dof.limit.shifted(-q0)
                dof.value = 0.0
            if joint.properties.target is not None and len(joint.dofs) == 1:
                joint.properties.target -= shifts[(joint.name, 0)]
        for c in model.couplings:
            # q_f' + s_f = r (q_l' + s_l) + c  =>  q_f' = r q_l' + (c + r s_l - s_f)
            s_l = shifts.get((c.leader.joint, c.leader.dof), 0.0)
            s_f = shifts.get((c.follower.joint, c.follower.dof), 0.0)
            c.offset = c.offset + c.ratio * s_l - s_f
        return model
