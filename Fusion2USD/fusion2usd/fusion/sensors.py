"""Tagged Fusion joint origins -> raw :class:`Sensor` records (values stay unparsed strings)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set

from ..core.report import ExportReport
from ..model.robot import Sensor, SensorKind
from .joints import context_path, occurrence_key
from .metadata import AttributeReader
from .units import items, point_m, transform_m

KIND_KEY = "sensor"
ORIGIN_TOLERANCE = 1e-3  # m, transform origin vs. joint geometry origin

LinkResolver = Callable[[Optional[str]], Optional[str]]  # occurrence key (None = root) -> link name


class SensorReader:
    """A sensor is a joint origin carrying a ``fusion2usd.sensor`` attribute (camera|imu|contact|frame)."""

    def __init__(self, root_component: Any, report: ExportReport, link_of: LinkResolver,
                 attributes: Optional[AttributeReader] = None) -> None:
        self._root = root_component
        self._report = report
        self._link_of = link_of
        self._attributes = attributes or AttributeReader()
        self._names: Set[str] = set()

    def read(self) -> List[Sensor]:
        out = []
        for origin in items(self._root.allJointOrigins):  # root-context proxies
            sensor = self._sensor(origin)
            if sensor is not None:
                out.append(sensor)
        return out

    def _sensor(self, origin: Any) -> Optional[Sensor]:
        name = origin.name
        attrs = self._read_attributes(origin, name)
        tag = attrs.get(KIND_KEY)
        if not tag:
            return None
        try:
            kind = SensorKind(str(tag).strip().lower())
        except ValueError:
            self._report.warn(f"joint origin {name!r}: unknown sensor kind {tag!r}; skipped")
            return None
        link = self._link_of(occurrence_key(origin.assemblyContext))
        if link is None:
            self._report.warn(f"sensor {name!r}: no exported link for its occurrence; skipped")
            return None
        try:
            transform = transform_m(origin.transform)
        except Exception as exc:  # broken references make the getter throw
            self._report.warn(f"sensor {name!r}: cannot read its frame ({exc}); skipped")
            return None
        self._check_origin(origin, name, transform.translation)
        context = context_path(origin)
        return Sensor(name=self._unique(name), kind=kind, link=link, transform=transform,
                      metadata={"attributes": attrs, "fusion_name": name},
                      source=f"{context}/{name}" if context else name)

    def _read_attributes(self, origin: Any, name: str) -> Dict[str, Any]:
        try:
            return self._attributes.read(origin)
        except ValueError as exc:
            self._report.warn(f"joint origin {name!r}: invalid fusion2usd attributes ({exc})")
            return dict(self._attributes.read_raw(origin))

    def _check_origin(self, origin: Any, name: str, translation: Any) -> None:
        """Sanity check that the proxy transform is in world space, like the joint geometry."""
        try:
            geometry = origin.geometry
            point = geometry.origin if geometry is not None else None
        except Exception:
            return
        if point is None:
            return
        error = (point_m(point) - translation).norm()
        if error > ORIGIN_TOLERANCE:
            self._report.warn(f"sensor {name!r}: transform origin is {error * 1000:.1f} mm off its joint "
                              f"geometry origin; using the transform")

    def _unique(self, name: str) -> str:
        candidate, n = name, 1
        while candidate in self._names:
            n += 1
            candidate = f"{name}_{n}"
        self._names.add(candidate)
        return candidate
