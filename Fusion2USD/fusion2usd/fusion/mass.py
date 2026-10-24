"""Mass properties of a set of Fusion bodies, in world coordinates (SI)."""

from __future__ import annotations

from typing import Any, List, Optional

import adsk.core
import adsk.fusion

from ..core.inertia import MassProperties
from ..core.report import ExportReport
from .units import KG_CM2, outputs, point_m, vec


def _accuracy() -> Any:
    acc = adsk.fusion.CalculationAccuracy
    return getattr(acc, "HighCalculationAccuracy", acc.MediumCalculationAccuracy)


def _from_properties(props: Any) -> Optional[MassProperties]:
    if props is None or props.mass <= 0.0:
        return None
    moments = outputs(props.getPrincipalMomentsOfInertia(), 3)
    axes = outputs(props.getPrincipalAxes(), 3)
    return MassProperties.from_principal(
        props.mass, point_m(props.centerOfMass),
        tuple(m * KG_CM2 for m in moments), tuple(vec(a) for a in axes),  # type: ignore[arg-type]
    )


class MassCalculator:
    def __init__(self, design: Any, report: ExportReport) -> None:
        self._design = design
        self._report = report

    def compute(self, bodies: List[Any], label: str) -> Optional[MassProperties]:
        if not bodies:
            return None
        try:
            coll = adsk.core.ObjectCollection.create()
            for b in bodies:
                coll.add(b)
            return _from_properties(self._design.physicalProperties(coll, _accuracy()))
        except Exception as exc:
            self._report.info(f"{label}: combined mass calculation failed ({exc}); summing bodies")
        parts = []
        for b in bodies:
            try:
                mp = _from_properties(b.getPhysicalProperties(_accuracy()))
            except Exception as exc:
                self._report.warn(f"{label}: mass of body {b.name!r} unavailable ({exc})")
                continue
            if mp is not None:
                parts.append(mp)
        return MassProperties.combine(parts) if parts else None
