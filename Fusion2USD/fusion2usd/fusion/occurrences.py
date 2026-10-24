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
            bodies = self.bodies(occ.bRepBodies)
            out.append(OccurrenceInfo(occ, OccurrenceRecord(
                key=key, parent=parent, depth=depth, has_bodies=bool(bodies),
                grounded=bool(occ.isGrounded), ground_to_parent=bool(occ.isGroundToParent),
            ), bodies))
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


def find(infos: Dict[str, OccurrenceInfo], key: Optional[str]) -> Optional[Any]:
    info = infos.get(key) if key is not None else None
    return info.entity if info is not None else None
