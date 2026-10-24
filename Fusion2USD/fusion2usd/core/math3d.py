"""Minimal, dependency-free 3D math (Fusion's bundled Python has no guaranteed numpy)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

Mat3 = Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]

EPS = 1e-12


@dataclass(frozen=True)
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @staticmethod
    def of(values: Iterable[float]) -> "Vec3":
        x, y, z = values
        return Vec3(float(x), float(y), float(z))

    def __add__(self, o: "Vec3") -> "Vec3":
        return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o: "Vec3") -> "Vec3":
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, s: float) -> "Vec3":
        return Vec3(self.x * s, self.y * s, self.z * s)

    __rmul__ = __mul__

    def __truediv__(self, s: float) -> "Vec3":
        return Vec3(self.x / s, self.y / s, self.z / s)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def dot(self, o: "Vec3") -> float:
        return self.x * o.x + self.y * o.y + self.z * o.z

    def cross(self, o: "Vec3") -> "Vec3":
        return Vec3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )

    def norm(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self) -> "Vec3":
        n = self.norm()
        if n < EPS:
            raise ValueError("cannot normalize a zero-length vector")
        return self / n

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def is_close(self, o: "Vec3", tol: float = 1e-9) -> bool:
        return (self - o).norm() <= tol

    def any_perpendicular(self) -> "Vec3":
        """A unit vector perpendicular to this (non-zero) vector."""
        a = self.normalized()
        helper = Vec3(1.0, 0.0, 0.0) if abs(a.x) < 0.9 else Vec3(0.0, 1.0, 0.0)
        return a.cross(helper).normalized()


# ---------------------------------------------------------------- Mat3 helpers

IDENTITY3: Mat3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def mat_from_columns(c0: Vec3, c1: Vec3, c2: Vec3) -> Mat3:
    return ((c0.x, c1.x, c2.x), (c0.y, c1.y, c2.y), (c0.z, c1.z, c2.z))


def mat_column(m: Mat3, i: int) -> Vec3:
    return Vec3(m[0][i], m[1][i], m[2][i])


def mat_transpose(m: Mat3) -> Mat3:
    return tuple(tuple(m[j][i] for j in range(3)) for i in range(3))  # type: ignore[return-value]


def mat_mul(a: Mat3, b: Mat3) -> Mat3:
    return tuple(  # type: ignore[return-value]
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3)
    )


def mat_vec(m: Mat3, v: Vec3) -> Vec3:
    return Vec3(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    )


def mat_add(a: Mat3, b: Mat3) -> Mat3:
    return tuple(tuple(a[i][j] + b[i][j] for j in range(3)) for i in range(3))  # type: ignore[return-value]


def mat_scale(a: Mat3, s: float) -> Mat3:
    return tuple(tuple(a[i][j] * s for j in range(3)) for i in range(3))  # type: ignore[return-value]


def mat_det(m: Mat3) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def orthonormalize(x: Vec3, y: Vec3) -> Mat3:
    """Right-handed rotation matrix whose first column follows ``x`` and second lies in the x/y plane."""
    ex = x.normalized()
    ey = (y - ex * ex.dot(y)).normalized()
    ez = ex.cross(ey)
    return mat_from_columns(ex, ey, ez)


# ---------------------------------------------------------------- Quaternion


@dataclass(frozen=True)
class Quat:
    """Unit quaternion, stored (w, x, y, z) like USD's GfQuat."""

    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @staticmethod
    def identity() -> "Quat":
        return Quat()

    @staticmethod
    def from_axis_angle(axis: Vec3, angle: float) -> "Quat":
        a = axis.normalized()
        s = math.sin(angle / 2.0)
        return Quat(math.cos(angle / 2.0), a.x * s, a.y * s, a.z * s)

    @staticmethod
    def from_matrix(m: Mat3) -> "Quat":
        tr = m[0][0] + m[1][1] + m[2][2]
        if tr > 0.0:
            s = math.sqrt(tr + 1.0) * 2.0
            q = Quat(0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
        elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
            s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
            q = Quat((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
        elif m[1][1] > m[2][2]:
            s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
            q = Quat((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s)
        else:
            s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
            q = Quat((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s)
        return q.normalized().canonical()

    def to_matrix(self) -> Mat3:
        w, x, y, z = self.w, self.x, self.y, self.z
        return (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )

    def __mul__(self, o: "Quat") -> "Quat":
        return Quat(
            self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
            self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
        )

    def conjugate(self) -> "Quat":
        return Quat(self.w, -self.x, -self.y, -self.z)

    def normalized(self) -> "Quat":
        n = math.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)
        if n < EPS:
            raise ValueError("zero quaternion")
        return Quat(self.w / n, self.x / n, self.y / n, self.z / n)

    def canonical(self) -> "Quat":
        """Same rotation with w >= 0 (q and -q are equivalent)."""
        return self if self.w >= 0 else Quat(-self.w, -self.x, -self.y, -self.z)

    def rotate(self, v: Vec3) -> Vec3:
        return mat_vec(self.to_matrix(), v)

    def angle_to(self, o: "Quat") -> float:
        d = abs(self.w * o.w + self.x * o.x + self.y * o.y + self.z * o.z)
        return 2.0 * math.acos(min(1.0, d))

    def to_axis_angle(self) -> Tuple[Vec3, float]:
        q = self.normalized().canonical()
        angle = 2.0 * math.acos(min(1.0, q.w))
        s = math.sqrt(max(0.0, 1.0 - q.w * q.w))
        if s < 1e-12:
            return Vec3(1.0, 0.0, 0.0), 0.0
        return Vec3(q.x / s, q.y / s, q.z / s), angle

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.w, self.x, self.y, self.z)


# ---------------------------------------------------------------- Transform


@dataclass(frozen=True)
class Transform:
    """Rigid transform: ``p_parent = rotation * p_local + translation``."""

    rotation: Quat = Quat()
    translation: Vec3 = Vec3()

    @staticmethod
    def identity() -> "Transform":
        return Transform()

    @staticmethod
    def from_axes(origin: Vec3, x_axis: Vec3, y_axis: Vec3) -> "Transform":
        """Frame from an origin and (possibly slightly non-orthogonal) x/y axes; z = x × y."""
        return Transform(Quat.from_matrix(orthonormalize(x_axis, y_axis)), origin)

    @staticmethod
    def with_x_axis(origin: Vec3, x_axis: Vec3, y_hint: Vec3 | None = None) -> "Transform":
        """Frame at ``origin`` whose +X follows ``x_axis``; ``y_hint`` steers the remaining roll."""
        x = x_axis.normalized()
        if y_hint is None or x.cross(y_hint).norm() < 1e-6:
            y_hint = x.any_perpendicular()
        return Transform.from_axes(origin, x, y_hint)

    @staticmethod
    def from_matrix4(rows: Sequence[Sequence[float]]) -> "Transform":
        """From a row-major 4x4 (column-vector convention, translation in the last column)."""
        m: Mat3 = tuple(tuple(float(rows[i][j]) for j in range(3)) for i in range(3))  # type: ignore[assignment]
        # Strip any scale/shear that slipped into CAD matrices.
        rot = orthonormalize(mat_column(m, 0), mat_column(m, 1))
        return Transform(Quat.from_matrix(rot), Vec3(rows[0][3], rows[1][3], rows[2][3]))

    def to_matrix4(self) -> Tuple[Tuple[float, ...], ...]:
        r = self.rotation.to_matrix()
        t = self.translation
        return (
            (r[0][0], r[0][1], r[0][2], t.x),
            (r[1][0], r[1][1], r[1][2], t.y),
            (r[2][0], r[2][1], r[2][2], t.z),
            (0.0, 0.0, 0.0, 1.0),
        )

    def __mul__(self, o: "Transform") -> "Transform":
        return Transform(
            (self.rotation * o.rotation).normalized(),
            self.rotation.rotate(o.translation) + self.translation,
        )

    def inverse(self) -> "Transform":
        inv = self.rotation.conjugate()
        return Transform(inv, -inv.rotate(self.translation))

    def apply_point(self, p: Vec3) -> Vec3:
        return self.rotation.rotate(p) + self.translation

    def apply_vector(self, v: Vec3) -> Vec3:
        return self.rotation.rotate(v)

    def axis(self, i: int) -> Vec3:
        return mat_column(self.rotation.to_matrix(), i)

    def is_close(self, o: "Transform", tol: float = 1e-6) -> bool:
        return self.translation.is_close(o.translation, tol) and self.rotation.angle_to(o.rotation) <= tol


def angle_between(a: Vec3, b: Vec3) -> float:
    c = a.normalized().dot(b.normalized())
    return math.acos(max(-1.0, min(1.0, c)))


def symmetric_eigen(m: Mat3, sweeps: int = 50) -> Tuple[Tuple[float, float, float], Mat3]:
    """Jacobi eigen-decomposition of a symmetric 3x3: returns (eigenvalues, right-handed eigenvector columns)."""
    a = [list(row) for row in m]
    v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    for _ in range(sweeps):
        off = abs(a[0][1]) + abs(a[0][2]) + abs(a[1][2])
        if off < 1e-18:
            break
        for p, q in ((0, 1), (0, 2), (1, 2)):
            if abs(a[p][q]) < 1e-30:
                continue
            theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
            t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
            c = 1.0 / math.sqrt(t * t + 1.0)
            s = t * c
            for k in range(3):
                akp, akq = a[k][p], a[k][q]
                a[k][p] = c * akp - s * akq
                a[k][q] = s * akp + c * akq
            for k in range(3):
                apk, aqk = a[p][k], a[q][k]
                a[p][k] = c * apk - s * aqk
                a[q][k] = s * apk + c * aqk
            for k in range(3):
                vkp, vkq = v[k][p], v[k][q]
                v[k][p] = c * vkp - s * vkq
                v[k][q] = s * vkp + c * vkq
    values = (a[0][0], a[1][1], a[2][2])
    vecs: Mat3 = tuple(tuple(row) for row in v)  # type: ignore[assignment]
    if mat_det(vecs) < 0:
        vecs = tuple((row[0], row[1], -row[2]) for row in vecs)  # type: ignore[assignment]
    return values, vecs
