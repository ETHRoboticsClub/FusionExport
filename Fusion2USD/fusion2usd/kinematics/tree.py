"""Builds the articulation tree: base selection, spanning tree, loop closure, orientation.

Reduced-coordinate articulations (PhysX, Newton) must be trees. Closed kinematic loops are kept by
excluding one joint per loop from the articulation (``physics:excludeFromArticulation``); the
simulator then solves that joint as a maximal-coordinate constraint.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Set, Tuple

from ..core.math3d import Transform
from ..core.naming import NameRegistry
from ..model.robot import Joint, Link, RobotModel
from ..options import DisconnectedPolicy
from .pipeline import ModelPass, PassContext


class KinematicTreePass(ModelPass):
    def run(self, model: RobotModel, ctx: PassContext) -> RobotModel:
        if not model.links:
            raise ValueError("the design contains no exportable bodies")
        self._drop_invalid_joints(model, ctx)
        model.base_link = self._select_base(model, ctx)
        model.fixed_base = self._resolve_fixed_base(model, ctx)

        unreachable = self._reachable_complement(model)
        if unreachable:
            self._handle_disconnected(model, ctx, unreachable)

        tree, excluded = self._spanning_tree(model)
        self._orient(model, tree)
        for joint in model.joints:
            joint.exclude_from_articulation = joint.name in excluded
        loops = {n for n in excluded if self._is_articulated(model, model.joint(n))}
        if loops:
            ctx.report.info(
                f"Closed kinematic loop(s): excluded {', '.join(sorted(loops))} from the articulation"
            )
        self._check_couplings(model, ctx)

        # Base first, then breadth-first order: easier to read in USD and in Isaac Lab body lists.
        order = self._bfs_order(model, tree_only=True)
        model.links = {name: model.links[name] for name in order + [n for n in model.links if n not in order]}
        return model

    # ------------------------------------------------------------------ steps

    @staticmethod
    def _drop_invalid_joints(model: RobotModel, ctx: PassContext) -> None:
        kept = []
        for j in model.joints:
            if not j.enabled:
                ctx.report.info(f"Joint {j.name!r} is disabled; skipped")
            elif j.parent == j.child:
                ctx.report.warn(f"Joint {j.name!r} connects {j.parent!r} to itself (rigidly merged parts); skipped")
            elif j.parent not in model.links or j.child not in model.links:
                ctx.report.warn(f"Joint {j.name!r} references a missing link; skipped")
            else:
                kept.append(j)
        dropped = {j.name for j in model.joints} - {j.name for j in kept}
        model.joints = kept
        model.couplings = [c for c in model.couplings
                           if c.leader.joint not in dropped and c.follower.joint not in dropped]

    @staticmethod
    def _select_base(model: RobotModel, ctx: PassContext) -> str:
        wanted = ctx.options.base_link
        if wanted:
            for link in model.links.values():
                if wanted in (link.name, link.source) or link.metadata.get("export_name") == wanted:
                    return link.name
            ctx.report.warn(f"Requested base link {wanted!r} not found; choosing automatically")
        if model.base_link in model.links:
            return model.base_link  # type: ignore[return-value]
        if model.grounded_links:
            return model.grounded_links[0]
        # Heaviest link is the most plausible chassis.
        return max(model.links.values(), key=link_mass).name

    @staticmethod
    def _resolve_fixed_base(model: RobotModel, ctx: PassContext) -> bool:
        if ctx.options.fixed_base is not None:
            return bool(ctx.options.fixed_base)
        return model.base_link in model.grounded_links

    def _reachable_complement(self, model: RobotModel) -> Set[str]:
        return set(model.links) - set(self._bfs_order(model))

    def _handle_disconnected(self, model: RobotModel, ctx: PassContext, unreachable: Set[str]) -> None:
        policy = ctx.options.disconnected
        groups = self._components(model, unreachable)
        names = ", ".join(sorted(unreachable))
        if policy is DisconnectedPolicy.DROP:
            ctx.report.warn(f"Dropped parts not connected to the base: {names}")
            for name in unreachable:
                del model.links[name]
            model.joints = [j for j in model.joints if j.parent in model.links and j.child in model.links]
            alive = {j.name for j in model.joints}
            model.couplings = [c for c in model.couplings if c.leader.joint in alive and c.follower.joint in alive]
            model.sensors = [s for s in model.sensors if s.link in model.links]
        elif policy is DisconnectedPolicy.FREE:
            ctx.report.warn(f"Parts not connected to the base are exported as free bodies: {names}")
            for name in unreachable:
                model.links[name].metadata["articulated"] = False
        else:
            ctx.report.warn(f"Parts not connected to the base were welded to it: {names}")
            base = model.links[model.base_link]  # type: ignore[index]
            names = NameRegistry(reserved={j.name for j in model.joints} | {"root_joint"})
            for group in groups:
                anchor = max(group, key=lambda n: link_mass(model.links[n]))
                child = model.links[anchor]
                model.add_joint(Joint(
                    name=names.claim(f"{anchor}_weld"), parent=base.name, child=anchor,
                    origin=child.transform.translation, frame_hint=Transform(child.transform.rotation),
                    source_type="weld", metadata={"generated": "disconnected part"},
                ))

    def _spanning_tree(self, model: RobotModel) -> Tuple[Dict[str, Joint], Set[str]]:
        """Prim's algorithm; cheap joints enter the tree first, so expensive ones get cut."""
        coupled = {r.joint for c in model.couplings for r in (c.leader, c.follower)}
        adjacency: Dict[str, List[Tuple[Tuple, Joint]]] = {n: [] for n in model.links}
        for idx, j in enumerate(model.joints):
            cost = (
                j.loop_closure_preferred,       # user asked to cut here
                j.name not in coupled,          # mimic joints must stay in the articulation
                j.properties.actuated is False, # keep actuated joints in reduced coordinates
                len(j.dofs) > 1,                # cut multi-DOF joints first: no helper links, fewer constraint rows
                idx,
            )
            adjacency[j.parent].append((cost, j))
            adjacency[j.child].append((cost, j))

        tree: Dict[str, Joint] = {}  # child link -> joint into it
        visited = {model.base_link}
        heap: List[Tuple[Tuple, int, str, Joint]] = []
        counter = 0
        for cost, j in adjacency[model.base_link]:  # type: ignore[index]
            heapq.heappush(heap, (cost, counter, model.base_link, j))
            counter += 1
        used: Set[str] = set()
        while heap:
            _, _, frm, j = heapq.heappop(heap)
            if j.name in used:
                continue
            to = j.other(frm)
            if to in visited:
                continue
            used.add(j.name)
            visited.add(to)
            tree[to] = j
            for cost, nxt in adjacency[to]:
                if nxt.name not in used:
                    heapq.heappush(heap, (cost, counter, to, nxt))
                    counter += 1

        # Joints outside the tree: loop closures, plus joints between free (non-articulated) bodies.
        return tree, {j.name for j in model.joints if j.name not in used}

    @staticmethod
    def _orient(model: RobotModel, tree: Dict[str, Joint]) -> None:
        for child, joint in tree.items():
            if joint.child != child:
                model.reverse_joint(joint.name)

    @staticmethod
    def _is_articulated(model: RobotModel, joint: Joint) -> bool:
        return all(model.links[n].metadata.get("articulated", True) for n in (joint.parent, joint.child))

    @staticmethod
    def _check_couplings(model: RobotModel, ctx: PassContext) -> None:
        kept = []
        followers: Set[Tuple[str, int]] = set()
        for c in model.couplings:
            leader, follower = model.find_joint(c.leader.joint), model.find_joint(c.follower.joint)
            if leader is None or follower is None:
                continue
            if leader.exclude_from_articulation or follower.exclude_from_articulation:
                ctx.report.warn(
                    f"Motion link {c.name!r}: joint is a loop-closure joint outside the articulation; "
                    "mimic constraint dropped"
                )
                continue
            key = (c.follower.joint, c.follower.dof)
            if key in followers:
                ctx.report.warn(f"Motion link {c.name!r}: {c.follower.joint!r} already follows another joint; dropped")
                continue
            followers.add(key)
            kept.append(c)
        model.couplings = kept

    # ------------------------------------------------------------------ graph helpers

    @staticmethod
    def _bfs_order(model: RobotModel, tree_only: bool = False) -> List[str]:
        if model.base_link not in model.links:
            return []
        order, seen = [model.base_link], {model.base_link}
        i = 0
        while i < len(order):
            cur = order[i]
            i += 1
            for j in model.joints_of(cur):
                if tree_only and j.exclude_from_articulation:
                    continue
                nxt = j.other(cur)
                if nxt not in seen:
                    seen.add(nxt)
                    order.append(nxt)
        return order  # type: ignore[return-value]

    @staticmethod
    def _components(model: RobotModel, nodes: Set[str]) -> List[List[str]]:
        remaining, groups = set(nodes), []
        while remaining:
            stack, group = [remaining.pop()], []
            while stack:
                cur = stack.pop()
                group.append(cur)
                for j in model.joints_of(cur):
                    nxt = j.other(cur)
                    if nxt in remaining:
                        remaining.remove(nxt)
                        stack.append(nxt)
            groups.append(sorted(group))
        return groups


def link_mass(link: Link) -> float:
    return link.mass.mass if link.mass else 0.0

