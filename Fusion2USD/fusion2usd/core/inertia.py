"""Rigid-body mass properties: combination, frame changes and principal decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from .math3d import (
    IDENTITY3,
    Mat3,
    Quat,
    Transform,
    Vec3,
    mat_add,
    mat_from_columns,
    mat_mul,
    mat_scale,
    mat_transpose,
    symmetric_eigen,
)


def _point_mass_tensor(mass: float, d: Vec3) -> Mat3:
    """Parallel-axis term ``m (|d|² E − d dᵀ)``."""
    dd = d.dot(d)
    v = (d.x, d.y, d.z)
    return tuple(  # type: ignore[return-value]
        tuple(mass * ((dd if i == j else 0.0) - v[i] * v[j]) for j in range(3)) for i in range(3)
    )


@dataclass(frozen=True)
class MassProperties:
    """Mass (kg), center of mass (m) and inertia tensor about the COM (kg·m²), all in one frame."""

    mass: float
    com: Vec3
    inertia: Mat3 = IDENTITY3

    @staticmethod
    def from_principal(mass: float, com: Vec3, moments: Tuple[float, float, float],
                       axes: Tuple[Vec3, Vec3, Vec3]) -> "MassProperties":
        r = mat_from_columns(*(a.normalized() for a in axes))
        d = ((moments[0], 0.0, 0.0), (0.0, moments[1], 0.0), (0.0, 0.0, moments[2]))
        return MassProperties(mass, com, mat_mul(mat_mul(r, d), mat_transpose(r)))

    @staticmethod
    def combine(parts: Iterable["MassProperties"]) -> "MassProperties":
        parts = [p for p in parts if p.mass > 0.0]
        if not parts:
            raise ValueError("no parts with positive mass to combine")
        total = sum(p.mass for p in parts)
        com = Vec3()
        for p in parts:
            com = com + p.com * (p.mass / total)
        inertia: Mat3 = ((0.0,) * 3,) * 3  # type: ignore[assignment]
        for p in parts:
            inertia = mat_add(inertia, mat_add(p.inertia, _point_mass_tensor(p.mass, p.com - com)))
        return MassProperties(total, com, inertia)

    def transformed(self, t: Transform) -> "MassProperties":
        """Re-express in another frame, where ``t`` maps current-frame coordinates to the new frame."""
        r = t.rotation.to_matrix()
        return MassProperties(self.mass, t.apply_point(self.com), mat_mul(mat_mul(r, self.inertia), mat_transpose(r)))

    def principal(self) -> Tuple[Tuple[float, float, float], Quat]:
        """Diagonal inertia and the rotation of the principal axes, as USD's MassAPI wants them."""
        values, vecs = symmetric_eigen(self.inertia)
        return values, Quat.from_matrix(vecs)

    def sanitized(self, min_mass: float, min_inertia: float) -> "MassProperties":
        """Clamp tiny/negative values that PhysX rejects (thin sheets, numerical noise)."""
        values, rot = self.principal()
        values = tuple(max(v, min_inertia) for v in values)
        r = rot.to_matrix()
        d = ((values[0], 0.0, 0.0), (0.0, values[1], 0.0), (0.0, 0.0, values[2]))
        return MassProperties(max(self.mass, min_mass), self.com, mat_mul(mat_mul(r, d), mat_transpose(r)))

    def scaled_mass(self, new_mass: float) -> "MassProperties":
        """Same shape and density distribution, different total mass."""
        if self.mass <= 0:
            return MassProperties(new_mass, self.com, self.inertia)
        return MassProperties(new_mass, self.com, mat_scale(self.inertia, new_mass / self.mass))
