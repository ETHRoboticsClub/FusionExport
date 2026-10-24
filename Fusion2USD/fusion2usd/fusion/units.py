"""Fusion API value conversion (internal units are cm, kg and rad) and collection helpers."""

from __future__ import annotations

from typing import Any, Iterator, Optional, Sequence

from ..core.math3d import Transform, Vec3

CM = 0.01            # cm -> m
KG_CM2 = 1e-4        # kg·cm² -> kg·m²


def items(collection: Any) -> Iterator[Any]:
    """Iterate a Fusion collection (``count``/``item``), a SWIG vector (e.g. the ``JointVector``
    returned by ``Component.allJoints``) or a plain Python sequence."""
    if collection is None:
        return iter(())
    if hasattr(collection, "count") and hasattr(collection, "item"):
        return (collection.item(i) for i in range(collection.count))
    return iter(collection)


def outputs(result: Any, n: int) -> Sequence[Any]:
    """The last ``n`` values of a SWIG call with output arguments (a leading bool status is dropped)."""
    if not isinstance(result, (list, tuple)):
        raise TypeError(f"expected a tuple of output values, got {type(result).__name__}")
    return tuple(result)[-n:]


def vec(v: Any, scale: float = 1.0) -> Vec3:
    """Vector3D / Point3D -> Vec3, optionally scaled (use ``CM`` for points)."""
    return Vec3(v.x * scale, v.y * scale, v.z * scale)


def point_m(p: Any) -> Vec3:
    return vec(p, CM)


def optional_vec(v: Any) -> Optional[Vec3]:
    if v is None:
        return None
    out = vec(v)
    return out if out.norm() > 1e-12 else None


def transform_m(matrix: Any) -> Transform:
    """Matrix3D (cm) -> orthonormal Transform (m)."""
    origin, x_axis, y_axis, _ = outputs(matrix.getAsCoordinateSystem(), 4)
    return Transform.from_axes(point_m(origin), vec(x_axis), vec(y_axis))


def optional_transform_m(matrix: Any) -> Optional[Transform]:
    return None if matrix is None else transform_m(matrix)
