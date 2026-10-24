"""The live self-test script imports against the adsk stub and its pure helpers behave."""

import importlib.util
import math
import os
import sys

import pytest

import _adsk_stub

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "Fusion2USD", "selftest", "SelfTest.py")


@pytest.fixture
def selftest():
    _adsk_stub.install()
    spec = importlib.util.spec_from_file_location("SelfTest", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["SelfTest"] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    yield module
    del sys.modules["SelfTest"]
    _adsk_stub.uninstall()


def test_bar_corners_enclose_segment(selftest):
    corners = selftest.bar_corners((0.0, 0.0), (0.0, 5.0), 1.0)
    xs, ys = sorted(c[0] for c in corners), sorted(c[1] for c in corners)
    assert xs[0] == pytest.approx(-0.5) and xs[-1] == pytest.approx(0.5)
    assert ys[0] == pytest.approx(-0.5) and ys[-1] == pytest.approx(5.5)
    # Consecutive corners form a rectangle (right angles).
    for i in range(4):
        a, b, c = corners[i], corners[(i + 1) % 4], corners[(i + 2) % 4]
        dot = (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1])
        assert dot == pytest.approx(0.0, abs=1e-12)


def test_four_bar_is_grashof_crank_rocker(selftest):
    d = lambda p, q: math.hypot(q[0] - p[0], q[1] - p[1])  # noqa: E731
    ground, crank = d(selftest.A, selftest.D), d(selftest.A, selftest.B)
    coupler, rocker = d(selftest.B, selftest.C), d(selftest.D, selftest.C)
    lengths = sorted([ground, crank, coupler, rocker])
    assert lengths[0] == crank and lengths[0] + lengths[3] < lengths[1] + lengths[2]


def test_values_restored_check(selftest):
    checker = selftest.ResultChecker(None, None, None, "", {}, {"a": [0.1]}, {"a": [0.1 + 1e-6]}, "")
    (check,) = checker.values_restored()
    assert not check.passed and check.line().startswith("[FAIL]")
