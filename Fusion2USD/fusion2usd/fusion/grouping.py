"""Group Fusion occurrences into rigid links (pure logic, no ``adsk``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set

ROOT = "<root>"  # synthetic node: root-component bodies and joints whose occurrence is null


@dataclass
class OccurrenceRecord:
    key: str                      # unique occurrence id (full path name)
    parent: Optional[str] = None  # key of the parent occurrence, None for top-level occurrences
    depth: int = 0
    has_bodies: bool = False      # has exported bodies of its own (not counting children)
    grounded: bool = False
    ground_to_parent: bool = False


@dataclass
class JointEdge:
    key: str
    one: Optional[str]  # occurrence key; None means ROOT
    two: Optional[str]
    rigid: bool


@dataclass
class Group:
    anchor: str
    members: List[str]
    grounded: bool = False

    @property
    def contains_root(self) -> bool:
        return ROOT in self.members


@dataclass
class GroupingResult:
    groups: List[Group] = field(default_factory=list)
    group_of: Dict[str, int] = field(default_factory=dict)  # node key -> index into ``groups``
    internal_joints: List[str] = field(default_factory=list)  # movable joints inside one group
    grounded_group: Optional[int] = None

    def group_index(self, occurrence: Optional[str]) -> Optional[int]:
        return self.group_of.get(ROOT if occurrence is None else occurrence)


class _UnionFind:
    def __init__(self, keys: Iterable[str]) -> None:
        self._parent = {k: k for k in keys}

    def find(self, k: str) -> str:
        root = k
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[k] != root:
            self._parent[k], k = root, self._parent[k]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


class PartGrouper:
    """Merges occurrences that cannot move relative to each other.

    Nodes are occurrences that have bodies, are referenced by a joint or rigid group, or are
    (ground-to-parent) grounded, plus ``ROOT`` when the root component has bodies or a joint
    references a null occurrence. Merged: rigid joints, rigid groups, ground-to-parent with the
    nearest ancestor node (ROOT for top-level occurrences), rigid clusters without any movable-joint
    reference with the nearest ancestor node of their shallowest member, and all grounded nodes together with ROOT.
    """

    def __init__(self, occurrences: Sequence[OccurrenceRecord], joints: Sequence[JointEdge],
                 rigid_groups: Sequence[Sequence[str]] = (), root_has_bodies: bool = False) -> None:
        self._occ = {o.key: o for o in occurrences}
        self._order = {o.key: i for i, o in enumerate(occurrences)}
        self._joints = list(joints)
        self._rigid_groups = [list(g) for g in rigid_groups]
        self._root_has_bodies = root_has_bodies

    def run(self) -> GroupingResult:
        nodes = self._nodes()
        uf = _UnionFind(nodes)
        movable_refs = self._ref_counts(movable=True)
        all_refs = self._ref_counts(movable=None)

        for j in self._joints:
            if j.rigid:
                uf.union(self._node(j.one), self._node(j.two))
        for g in self._rigid_groups:
            members = [k for k in g if k in nodes]
            for k in members[1:]:
                uf.union(members[0], k)
        for key in nodes:
            if key != ROOT and self._occ[key].ground_to_parent:
                uf.union(self._nearest_ancestor_node(self._occ[key], nodes) or ROOT, key)
        # A rigid cluster that no movable joint touches moves with the sub-assembly containing it.
        components: Dict[str, List[str]] = {}
        for key in nodes:
            components.setdefault(uf.find(key), []).append(key)
        for members in components.values():
            if ROOT in members or any(movable_refs.get(k, 0) for k in members):
                continue
            shallowest = min(members, key=self._sort_key)
            ancestor = self._nearest_ancestor_node(self._occ[shallowest], nodes)
            if ancestor is not None and uf.find(ancestor) != uf.find(shallowest):
                uf.union(ancestor, shallowest)
        grounded = [k for k in nodes if k == ROOT or self._occ[k].grounded]
        for k in grounded[1:]:
            uf.union(grounded[0], k)

        result = GroupingResult()
        by_root: Dict[str, List[str]] = {}
        for key in sorted(nodes, key=self._sort_key):
            by_root.setdefault(uf.find(key), []).append(key)
        grounded_root = uf.find(grounded[0]) if grounded else None
        for root, members in by_root.items():
            idx = len(result.groups)
            anchor = ROOT if ROOT in members else min(
                members, key=lambda k: (-movable_refs.get(k, 0), -all_refs.get(k, 0), self._sort_key(k)))
            result.groups.append(Group(anchor, members, grounded=root == grounded_root))
            for k in members:
                result.group_of[k] = idx
            if root == grounded_root:
                result.grounded_group = idx
        for j in self._joints:
            if not j.rigid and result.group_of[self._node(j.one)] == result.group_of[self._node(j.two)]:
                result.internal_joints.append(j.key)
        return result

    # -- helpers
    @staticmethod
    def _node(key: Optional[str]) -> str:
        return ROOT if key is None else key

    def _sort_key(self, key: str):
        if key == ROOT:
            return (-1, -1)
        return (self._occ[key].depth, self._order[key])

    def _nodes(self) -> Set[str]:
        nodes: Set[str] = set()
        if self._root_has_bodies:
            nodes.add(ROOT)
        for o in self._occ.values():
            if o.has_bodies or o.grounded or o.ground_to_parent:
                nodes.add(o.key)
        for j in self._joints:
            for k in (j.one, j.two):
                if k is None or k in self._occ:
                    nodes.add(self._node(k))
                else:
                    raise KeyError(f"joint {j.key!r} references unknown occurrence {k!r}")
        for g in self._rigid_groups:
            nodes.update(k for k in g if k in self._occ)
        # Ground-to-parent at the top level means grounded to the root component.
        if any(self._occ[k].ground_to_parent and self._nearest_ancestor_node(self._occ[k], nodes) is None
               for k in nodes if k != ROOT):
            nodes.add(ROOT)
        return nodes

    def _nearest_ancestor_node(self, occ: OccurrenceRecord, nodes: Set[str]) -> Optional[str]:
        parent = occ.parent
        while parent is not None:
            if parent in nodes:
                return parent
            parent = self._occ[parent].parent
        return None

    def _ref_counts(self, movable: Optional[bool]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for j in self._joints:
            if movable is not None and j.rigid == movable:
                continue
            for k in (j.one, j.two):
                n = self._node(k)
                counts[n] = counts.get(n, 0) + 1
        return counts
