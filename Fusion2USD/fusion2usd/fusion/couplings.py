"""Fusion motion links -> :class:`Coupling`."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from ..core.report import ExportReport
from ..model.robot import Coupling, DofRef, Joint
from .joints import MOTION_DOF_INDEX, FusionJoint, context_path, to_si
from .units import items


class MotionLinkReader:
    """Collects motion links from the root and every occurrence's component (as proxies)."""

    def __init__(self, root_component: Any, occurrences: List[Any], report: ExportReport) -> None:
        self._root = root_component
        self._occurrences = occurrences
        self._report = report

    def read(self, fusion_joints: List[FusionJoint], joints: Dict[str, Joint]) -> List[Coupling]:
        """``joints`` maps the unique joint name to the model joint (only exported joints)."""
        by_identity = {fj.identity: fj for fj in fusion_joints}
        out: List[Coupling] = []
        for link, ctx in self._links():
            coupling = self._coupling(link, ctx, by_identity, joints)
            if coupling is not None:
                out.append(coupling)
        return out

    def _links(self) -> Iterator[Tuple[Any, str]]:
        seen: Set[Tuple[str, str]] = set()
        sources: List[Tuple[Any, Optional[Any]]] = [(self._root, None)]
        sources += [(occ.component, occ) for occ in self._occurrences]
        for component, occ in sources:
            ctx = occ.fullPathName if occ is not None else ""
            for link in items(component.motionLinks):
                if occ is not None:
                    link = link.createForAssemblyContext(occ)
                if link is None or link.isSuppressed:
                    continue
                identity = (ctx, link.name)
                if identity in seen:
                    continue
                seen.add(identity)
                yield link, ctx

    def _coupling(self, link: Any, ctx: str, by_identity: Dict[Tuple[str, str], FusionJoint],
                  joints: Dict[str, Joint]) -> Optional[Coupling]:
        name = f"{ctx}/{link.name}" if ctx else link.name
        one = self._resolve(link.jointOne, ctx, by_identity)
        two = self._resolve(link.jointTwo, ctx, by_identity) if link.jointTwo is not None else one
        if one is None or two is None:
            self._report.warn(f"motion link {name!r}: cannot resolve its joints; skipped")
            return None
        if one.name not in joints or two.name not in joints:
            self._report.warn(f"motion link {name!r}: a linked joint is not exported; skipped")
            return None
        i1, i2 = MOTION_DOF_INDEX.get(link.motionOne), MOTION_DOF_INDEX.get(link.motionTwo)
        if i1 is None or i2 is None or i1 >= len(one.specs) or i2 >= len(two.specs):
            self._report.warn(f"motion link {name!r}: unsupported motion types; skipped")
            return None
        v1 = to_si(one.specs[i1], link.valueOne.value)
        v2 = to_si(two.specs[i2], link.valueTwo.value)
        if abs(v1) < 1e-15:
            self._report.warn(f"motion link {name!r}: zero leader value; skipped")
            return None
        ratio = v2 / v1 * (-1.0 if link.isReversed else 1.0)
        q1 = joints[one.name].dofs[i1].value
        q2 = joints[two.name].dofs[i2].value
        return Coupling(name=name, leader=DofRef(one.name, i1), follower=DofRef(two.name, i2),
                        ratio=ratio, offset=q2 - ratio * q1, source=name)

    @staticmethod
    def _resolve(joint: Any, ctx: str, by_identity: Dict[Tuple[str, str], FusionJoint]) -> Optional[FusionJoint]:
        if joint is None:
            return None
        # Joints of a link proxy may come back native (no context): fall back to the link's context.
        own = context_path(joint)
        return by_identity.get((own or ctx, joint.name)) or by_identity.get((ctx, joint.name))
