import math
import random

import pytest

from fusion2usd.core.inertia import MassProperties
from fusion2usd.core.math3d import Quat, Transform, Vec3, mat_mul, mat_transpose, symmetric_eigen
from fusion2usd.core.naming import NameRegistry, sanitize


def rand_quat(rng):
    return Quat(*(rng.uniform(-1, 1) for _ in range(4))).normalized()


def test_quat_matrix_round_trip():
    rng = random.Random(0)
    for _ in range(200):
        q = rand_quat(rng).canonical()
        assert Quat.from_matrix(q.to_matrix()).angle_to(q) < 1e-6


def test_transform_compose_inverse():
    rng = random.Random(1)
    a = Transform(rand_quat(rng), Vec3(1, 2, 3))
    b = Transform(rand_quat(rng), Vec3(-2, 0.5, 4))
    p = Vec3(0.3, -0.7, 2.0)
    assert (a * b).apply_point(p).is_close(a.apply_point(b.apply_point(p)))
    assert (a * a.inverse()).is_close(Transform())


def test_with_x_axis():
    t = Transform.with_x_axis(Vec3(1, 1, 1), Vec3(0, 0, 2))
    assert t.axis(0).is_close(Vec3(0, 0, 1))
    assert abs(t.axis(1).dot(t.axis(0))) < 1e-12


def test_symmetric_eigen_reconstructs():
    rng = random.Random(2)
    r = rand_quat(rng).to_matrix()
    d = ((3.0, 0, 0), (0, 1.0, 0), (0, 0, 2.0))
    m = mat_mul(mat_mul(r, d), mat_transpose(r))
    values, vecs = symmetric_eigen(m)
    assert sorted(values) == pytest.approx([1.0, 2.0, 3.0])
    back = mat_mul(mat_mul(vecs, tuple(tuple(values[i] if i == j else 0.0 for j in range(3)) for i in range(3))),
                   mat_transpose(vecs))
    for i in range(3):
        assert back[i] == pytest.approx(m[i], abs=1e-9)


def test_parallel_axis_combination():
    # Two 1 kg point-ish masses at ±1 on x: Izz = Iyy = 2, Ixx = 0 (+ their own tiny inertia).
    eps = 1e-6
    tiny = ((eps, 0, 0), (0, eps, 0), (0, 0, eps))
    total = MassProperties.combine([MassProperties(1, Vec3(1, 0, 0), tiny), MassProperties(1, Vec3(-1, 0, 0), tiny)])
    assert total.mass == 2
    assert total.com.is_close(Vec3())
    assert total.inertia[1][1] == pytest.approx(2 + 2 * eps)
    assert total.inertia[0][0] == pytest.approx(2 * eps)


def test_mass_transform_and_principal():
    mp = MassProperties.from_principal(2.0, Vec3(0, 0, 1), (1.0, 2.0, 3.0),
                                       (Vec3(0, 1, 0), Vec3(-1, 0, 0), Vec3(0, 0, 1)))
    t = Transform(Quat.from_axis_angle(Vec3(0, 0, 1), 0.3), Vec3(1, 0, 0))
    moved = mp.transformed(t)
    values, rot = moved.principal()
    assert sorted(values) == pytest.approx([1.0, 2.0, 3.0])
    assert moved.com.is_close(Vec3(1, 0, 1))


def test_naming():
    assert sanitize("Revolute 1") == "Revolute_1"
    assert sanitize("3rd:part") == "_3rd_part"
    reg = NameRegistry({"joints"})
    assert reg.claim("joints") == "joints_2"
    assert reg.claim("a b") == "a_b"
    assert reg.claim("a-b") == "a_b_2"
