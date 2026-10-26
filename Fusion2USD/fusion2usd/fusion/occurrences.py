"""Walks the Fusion occurrence tree into :class:`OccurrenceRecord` inputs for the grouper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .grouping import OccurrenceRecord
from .units import items


@dataclass
class OccurrenceInfo:
    entity: Any                                     # root-context occurrence proxy
    record: OccurrenceRecord
    bodies: List[Any] = field(default_factory=list)  # exported bodies of this occurrence only
    hidden: bool = False                            # has bodies, but all of them are hidden (and skipped)


class OccurrenceScanner:
    def __init__(self, root_component: Any, include_hidden: bool) -> None:
        self._root = root_component
        self._include_hidden = include_hidden

    def scan(self) -> List[OccurrenceInfo]:
        """All occurrences, parents before children."""
        out: List[OccurrenceInfo] = []
        stack = [(occ, None, 0) for occ in reversed(list(items(self._root.occurrences)))]
        while stack:
            occ, parent, depth = stack.pop()
            key = occ.fullPathName
            all_bodies = list(items(occ.bRepBodies))
            bodies = self.bodies(all_bodies)
            out.append(OccurrenceInfo(occ, OccurrenceRecord(
                key=key, parent=parent, depth=depth, has_bodies=bool(bodies),
                grounded=bool(occ.isGrounded), ground_to_parent=bool(occ.isGroundToParent),
            ), bodies, hidden=bool(all_bodies) and not bodies))
            children = list(items(occ.childOccurrences))
            stack.extend((child, key, depth + 1) for child in reversed(children))
        return out

    def root_bodies(self) -> List[Any]:
        return self.bodies(self._root.bRepBodies)

    def bodies(self, collection: Any) -> List[Any]:
        # BRepBody.isVisible already accounts for hidden parent occurrences.
        return [b for b in items(collection) if self._include_hidden or b.isVisible]

    @staticmethod
    def by_key(infos: List[OccurrenceInfo]) -> Dict[str, OccurrenceInfo]:
        return {i.record.key: i for i in infos}


def resolve_key(key: Optional[str], context: str, known: Dict[str, Any]) -> Optional[str]:
    """Root-context path of ``key``, or None if it cannot be placed.

    Joints and rigid groups inside an externally referenced design report occurrence paths relative to
    that design ("I-R:1+I-PP:1") although their assembly context is a full path ("orca_right_joints:1+I-R:1").
    The missing prefix is one of the context's ancestors."""
    if key is None or key in known:
        return key
    parts = context.split("+") if context else []
    for n in range(len(parts), 0, -1):
        candidate = "+".join(parts[:n]) + "+" + key
        if candidate in known:
            return candidate
    return None


def find(infos: Dict[str, OccurrenceInfo], key: Optional[str]) -> Optional[Any]:
    info = infos.get(key) if key is not None else None
    return info.entity if info is not None else None
