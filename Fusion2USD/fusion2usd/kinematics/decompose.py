"""Splits multi-DOF articulation joints into chains of single-DOF joints.

PhysX articulations (and Isaac Lab's joint indexing) handle revolute/prismatic joints best, so a
Fusion cylindrical, pin-slot, planar or ball joint becomes parent → helper → ... → child with one
DOF per joint. Helper links are tiny virtual bodies at the joint origin. Multi-DOF joints that were
excluded from the articulation (loop closures) stay whole; the USD writer emits them as D6 joints.

Newton solves spherical and D6 joints natively, so for that engine joints stay whole unless
``_newton_split_reason`` names a reason to split them.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional, Set

from ..core.inertia import MassProperties
from ..core.math3d import Transform, Vec3
from ..core.naming import NameRegistry
from ..model.robot import DofRef, DofType, Joint, JointProperties, Link, RobotModel
from ..options import PhysicsEngine
from .pipeline import ModelPass, PassContext


class JointDecompositionPass(ModelPass):
    def run(self, model: RobotModel, ctx: PassContext) -> RobotModel:
        newton = ctx.options.physics_engine is PhysicsEngine.NEWTON
        coupled = {r.joint for c in model.couplings for r in (c.leader, c.follower)} \
            if ctx.options.export_mimic_joints else set()
        joint_names = NameRegistry(reserved={j.name for j in model.joints} | {"root_joint"})
        link_names = NameRegistry(reserved=set(model.links) | {"joints", "Looks"})
        dof_map: Dict[DofRef, DofRef] = {}
        result = []
        for joint in model.joints:
            if len(joint.dofs) <= 1 or joint.exclude_from_articulation:
                result.append(joint)
                continue
            if newton:
                reason = self._newton_split_reason(joint, coupled)
                if reason is None:
                    result.append(joint)
                    continue
                ctx.report.info(f"Joint {joint.name!r} is split because {reason}")
            result.extend(self._split(model, joint, joint_names, link_names, dof_map, ctx))
        model.joints = result
        for c in model.couplings:
            c.leader = dof_map.get(c.leader, c.leader)
            c.follower = dof_map.get(c.follower, c.follower)
        return model

    @staticmethod
    def _newton_split_reason(joint: Joint, coupled: Set[str]) -> Optional[str]:
        if joint.name in coupled:
            return "it is part of a motion link (Newton's mimic only couples single-DOF joints)"
        if joint.properties.actuated:
            return "it is actuated (multi-DOF joints are written without drives)"
        rotations = [d.axis.normalized() for d in joint.dofs if d.type is DofType.ROTATION]
        slides = [d.axis.normalized() for d in joint.dofs if d.type is DofType.TRANSLATION]
        # A D6 translates in body0's frame, then rotates. Reversed, the slot sits on body1 and turns with it.
        if joint.metadata.get("reversed") and len(rotations) == 1 and any(
                abs(s.dot(rotations[0])) < 1.0 - 1e-6 for s in slides):
            return "it was reversed into the tree, and a D6 joint cannot slide along an axis that turns with it"
        return None

    @staticmethod
    def _split(model: RobotModel, joint: Joint, joint_names: NameRegistry, link_names: NameRegistry,
               dof_map: Dict[DofRef, DofRef], ctx: PassContext):
        opts = ctx.options
        # In Fusion's orientation translations come first (the slot/plane is fixed in the Fusion parent,
        # the pin rotates on top). A joint reversed by the tree pass is the inverse chain: rotations first.
        reversed_ = bool(joint.metadata.get("reversed"))
        order = sorted(range(len(joint.dofs)),
                       key=lambda k: (joint.dofs[k].type is DofType.TRANSLATION) == reversed_)
        pieces_props = JointDecompositionPass._piece_properties(joint, ctx)

        parent_tf = model.links[joint.parent].transform
        chain = [joint.parent]
        for k in order[:-1]:
            label = joint.dofs[k].label or f"dof{k}"
            helper = Link(
                name=link_names.claim(f"{joint.name}_{label}_link"),
                transform=Transform(parent_tf.rotation, joint.origin),
                mass=MassProperties(
                    opts.virtual_link_mass, Vec3(),
                    ((opts.virtual_link_inertia, 0.0, 0.0), (0.0, opts.virtual_link_inertia, 0.0),
                     (0.0, 0.0, opts.virtual_link_inertia)),
                ),
                is_virtual=True,
                source=joint.source,
                metadata={"generated": f"helper link of multi-DOF joint {joint.name}"},
            )
            model.add_link(helper)
            chain.append(helper.name)
        chain.append(joint.child)

        pieces = []
        for i, k in enumerate(order):
            dof = joint.dofs[k]
            piece = Joint(
                name=joint_names.claim(f"{joint.name}_{dof.label or f'dof{k}'}"),
                parent=chain[i],
                child=chain[i + 1],
                origin=joint.origin,
                dofs=[dof],
                properties=copy.copy(pieces_props[dof.type]),
                metadata=dict(joint.metadata, decomposed_from=joint.name),
                source=joint.source,
                source_type=joint.source_type,
                frame_hint=joint.frame_hint,
            )
            dof_map[DofRef(joint.name, k)] = DofRef(piece.name, 0)
            pieces.append(piece)
        ctx.report.info(
            f"Split {joint.source_type or 'multi-DOF'} joint {joint.name!r} into {', '.join(p.name for p in pieces)}"
        )
        return pieces

    @staticmethod
    def _piece_properties(joint: Joint, ctx: PassContext) -> Dict[DofType, JointProperties]:
        """Unit-bearing values (N·m vs N, rad/s vs m/s, ...) only apply to DOFs of the joint's primary type."""
        # Fusion lists the joint's primary motion first; reversal mirrored the DOF order.
        primary = joint.dofs[-1 if joint.metadata.get("reversed") else 0].type
        props = joint.properties
        if props.target is not None:
            ctx.report.warn(f"Joint {joint.name!r}: a drive target is ambiguous for a multi-DOF joint; ignored")
        base = copy.copy(props)
        base.target = None
        out = {primary: base}
        other = DofType.TRANSLATION if primary is DofType.ROTATION else DofType.ROTATION
        if any(d.type is other for d in joint.dofs):
            stripped = copy.copy(base)
            dropped = [f for f in _UNIT_FIELDS if getattr(stripped, f) is not None]
            for f in _UNIT_FIELDS:
                setattr(stripped, f, None)
            if dropped:
                ctx.report.warn(
                    f"Joint {joint.name!r}: {', '.join(dropped)} apply to its {primary.value} DOF only, "
                    f"not to the {other.value} DOF; export defaults are used there"
                )
            out[other] = stripped
        return out


_UNIT_FIELDS = ("effort", "velocity", "stiffness", "damping", "armature")
