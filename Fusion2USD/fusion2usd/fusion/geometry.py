"""Triangle meshes and appearance materials from Fusion B-Rep bodies."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import adsk.core
import adsk.fusion

from ..core.math3d import Transform, Vec3
from ..core.report import ExportReport
from ..model.robot import Material, Mesh
from ..options import MeshQuality
from .units import CM, items, transform_m, vec

_QUALITY = {
    MeshQuality.LOW: adsk.fusion.TriangleMeshQualityOptions.LowQualityTriangleMesh,
    MeshQuality.NORMAL: adsk.fusion.TriangleMeshQualityOptions.NormalQualityTriangleMesh,
    MeshQuality.HIGH: adsk.fusion.TriangleMeshQualityOptions.HighQualityTriangleMesh,
    MeshQuality.VERY_HIGH: adsk.fusion.TriangleMeshQualityOptions.VeryHighQualityTriangleMesh,
}

Bounds = Tuple[Vec3, Vec3]


def _bounds(points: List[float]) -> Bounds:
    xs, ys, zs = points[0::3], points[1::3], points[2::3]
    return Vec3(min(xs), min(ys), min(zs)), Vec3(max(xs), max(ys), max(zs))


def _bounds_error(a: Bounds, b: Bounds) -> float:
    return (a[0] - b[0]).norm() + (a[1] - b[1]).norm()


def _transformed_bounds(b: Bounds, t: Transform) -> Bounds:
    corners = [t.apply_point(Vec3(x, y, z)) for x in (b[0].x, b[1].x) for y in (b[0].y, b[1].y)
               for z in (b[0].z, b[1].z)]
    flat = [c for p in corners for c in p]
    return _bounds(flat)


def _apply(points: List[float], t: Transform, scale: float = 1.0) -> List[float]:
    """``t · (scale · p)`` for a flat xyz array."""
    r = t.rotation.to_matrix()
    tx, ty, tz = t.translation
    out = [0.0] * len(points)
    for i in range(0, len(points), 3):
        x, y, z = points[i] * scale, points[i + 1] * scale, points[i + 2] * scale
        out[i] = r[0][0] * x + r[0][1] * y + r[0][2] * z + tx
        out[i + 1] = r[1][0] * x + r[1][1] * y + r[1][2] * z + ty
        out[i + 2] = r[2][0] * x + r[2][1] * y + r[2][2] * z + tz
    return out


def srgb_to_linear(c: float) -> float:
    """Standard sRGB transfer function, inverted (Fusion colors are sRGB)."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


class AppearanceReader:
    """Builds one :class:`Material` per Fusion appearance (keyed by appearance id)."""

    def __init__(self) -> None:
        self._by_id: Dict[str, Material] = {}
        self._names: Dict[str, str] = {}  # material name -> appearance id

    @property
    def materials(self) -> List[Material]:
        return list(self._by_id.values())

    def material_for(self, appearance: Any) -> Optional[Material]:
        if appearance is None:
            return None
        key = appearance.id or appearance.name
        if key not in self._by_id:
            self._by_id[key] = self._build(appearance, self._unique_name(appearance.name or "material", key))
        return self._by_id[key]

    def _unique_name(self, name: str, key: str) -> str:
        candidate, n = name, 1
        while self._names.get(candidate, key) != key:
            n += 1
            candidate = f"{name}_{n}"
        self._names[candidate] = key
        return candidate

    @staticmethod
    def _build(appearance: Any, name: str) -> Material:
        color = None
        transparency = None
        props = appearance.appearanceProperties
        for prop in items(props):
            cp = adsk.core.ColorProperty.cast(prop)
            if cp is not None and cp.value is not None and (color is None or cp.id == "generic_diffuse"):
                color = cp.value
            fp = adsk.core.FloatProperty.cast(prop)
            if fp is not None and fp.id == "generic_transparency":
                transparency = fp.value
        if color is None:
            return Material(name)
        opacity = color.opacity / 255.0
        if transparency is not None:
            opacity = min(opacity, 1.0 - transparency)
        rgb = (srgb_to_linear(color.red / 255.0), srgb_to_linear(color.green / 255.0),
               srgb_to_linear(color.blue / 255.0))
        return Material(name, rgb, opacity)


class MeshExtractor:
    """Tessellates bodies and expresses the result in a link frame (meters)."""

    def __init__(self, report: ExportReport, appearances: Optional[AppearanceReader] = None) -> None:
        self._report = report
        self.appearances = appearances or AppearanceReader()

    def mesh(self, body: Any, quality: MeshQuality, link_transform: Transform,
             with_material: bool) -> Optional[Mesh]:
        try:
            calc = body.meshManager.createMeshCalculator()
            calc.setQuality(_QUALITY[quality])
            tri = calc.calculate()
        except Exception as exc:
            self._report.warn(f"body {body.name!r}: tessellation failed ({exc})")
            return None
        if tri is None or not tri.nodeCount:
            self._report.warn(f"body {body.name!r}: empty tessellation")
            return None
        points_cm = list(tri.nodeCoordinatesAsDouble)
        to_link = link_transform.inverse() * self._mesh_to_world(body, points_cm)
        normals = list(tri.normalVectorsAsDouble) or None
        if normals is not None and len(normals) != len(points_cm):
            normals = None
        if normals is not None:
            normals = _apply(normals, Transform(to_link.rotation))
        material = None
        if with_material:
            appearance = body.appearance
            if appearance is None and body.assemblyContext is not None:
                appearance = body.assemblyContext.appearance
            mat = self.appearances.material_for(appearance)
            material = mat.name if mat is not None else None
        return Mesh(body.name, _apply(points_cm, to_link, CM), list(tri.nodeIndices), normals, material)

    def _mesh_to_world(self, body: Any, points_cm: List[float]) -> Transform:
        """Transform taking tessellation coordinates (scaled to m) to world.

        A proxy body's ``boundingBox`` is in world coordinates; tessellation output is expected to
        be too, but if it matches only after applying the assembly context transform, use that.
        """
        ctx = getattr(body, "assemblyContext", None)
        if ctx is None:
            return Transform()
        mesh_b = _bounds([p * CM for p in points_cm])
        try:
            bb = body.boundingBox
            body_b = (vec(bb.minPoint, CM), vec(bb.maxPoint, CM))
            ctx_t = transform_m(ctx.transform2)
        except Exception:
            return Transform()
        if _bounds_error(_transformed_bounds(mesh_b, ctx_t), body_b) < 0.5 * _bounds_error(mesh_b, body_b):
            return ctx_t
        return Transform()
