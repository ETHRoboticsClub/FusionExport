"""Collision geometry and filtering that need no third-party packages (Fusion ships none).

``link_hull``: one convex hull over all of a link's collision meshes. Its vertices are the meshes' extreme points
in a fixed, evenly spread set of directions, so the hull is the exact hull's inscribed polytope through those
points: every vertex lies on the exact hull, and it has at most as many vertices as directions. Newton and PhysX
cap a hull at 64 vertices by default, so the export's default of 64 directions is what they simulate; it reaches
within about 2.5 % of the body's radius of the exact hull.

``joint_neighbours``: pairs of links at most ``distance`` joints apart, counting every joint the export writes,
loop-closing ones included.
"""

from __future__ import annotations

import math
from typing import Dict, FrozenSet, Iterable, List, Sequence, Set, Tuple

Point = Tuple[float, float, float]


def fibonacci_directions(count: int) -> List[Point]:
    """``count`` unit vectors spread evenly over the sphere (golden-angle spiral), deterministic."""
    golden = math.pi * (3.0 - math.sqrt(5.0))
    out = []
    for i in range(count):
        z = 1.0 - 2.0 * (i + 0.5) / count
        r = math.sqrt(max(0.0, 1.0 - z * z))
        a = golden * i
        out.append((r * math.cos(a), r * math.sin(a), z))
    return out


def support_points(points: Sequence[float], directions: Sequence[Point]) -> List[Point]:
    """The extreme point of a flat ``[x0, y0, z0, x1, ...]`` cloud in each direction, without duplicates, in the
    order the directions first reach them."""
    n = len(points) // 3
    best = [(-math.inf, -1)] * len(directions)
    for i in range(n):
        x, y, z = points[3 * i], points[3 * i + 1], points[3 * i + 2]
        for k, (dx, dy, dz) in enumerate(directions):
            d = x * dx + y * dy + z * dz
            if d > best[k][0]:
                best[k] = (d, i)
    seen: Dict[int, None] = {}
    for _, i in best:
        if i >= 0:
            seen.setdefault(i, None)
    return [(points[3 * i], points[3 * i + 1], points[3 * i + 2]) for i in seen]


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Point, b: Point) -> Point:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def convex_hull(points: Sequence[Point]) -> Tuple[List[Point], List[Tuple[int, int, int]]]:
    """Vertices and outward triangles of the convex hull of a few hundred points (incremental, O(n^2)).
    Raises ``ValueError`` for fewer than four points not all in one plane."""
    pts = list(dict.fromkeys(points))
    if len(pts) < 4:
        raise ValueError("a convex hull needs four points not in one plane")
    scale = max(max(abs(c) for c in p) for p in pts) or 1.0
    eps = 1e-9 * scale
    # Initial tetrahedron: two farthest-apart extremes, the farthest from their line, the farthest from that plane.
    a = min(range(len(pts)), key=lambda i: pts[i])
    b = max(range(len(pts)), key=lambda i: _dot(_sub(pts[i], pts[a]), _sub(pts[i], pts[a])))
    ab = _sub(pts[b], pts[a])
    c = max(range(len(pts)), key=lambda i: _dot(_cross(ab, _sub(pts[i], pts[a])), _cross(ab, _sub(pts[i], pts[a]))))
    normal = _cross(ab, _sub(pts[c], pts[a]))
    d = max(range(len(pts)), key=lambda i: abs(_dot(normal, _sub(pts[i], pts[a]))))
    if abs(_dot(normal, _sub(pts[d], pts[a]))) <= eps * math.sqrt(_dot(normal, normal)):
        raise ValueError("the points lie in one plane")
    centre = tuple(sum(pts[i][k] for i in (a, b, c, d)) / 4.0 for k in range(3))

    def oriented(i: int, j: int, k: int) -> Tuple[int, int, int]:
        n = _cross(_sub(pts[j], pts[i]), _sub(pts[k], pts[i]))
        return (i, k, j) if _dot(n, _sub(pts[i], centre)) < 0 else (i, j, k)

    faces: Set[Tuple[int, int, int]] = {oriented(a, b, c), oriented(a, b, d), oriented(a, c, d), oriented(b, c, d)}
    for p in range(len(pts)):
        if p in (a, b, c, d):
            continue
        visible = []
        for f in faces:
            i, j, k = f
            n = _cross(_sub(pts[j], pts[i]), _sub(pts[k], pts[i]))
            if _dot(n, _sub(pts[p], pts[i])) > eps * math.sqrt(_dot(n, n)):
                visible.append(f)
        if not visible:
            continue
        edges: Dict[Tuple[int, int], int] = {}
        for i, j, k in visible:
            for e in ((i, j), (j, k), (k, i)):
                edges[e] = edges.get(e, 0) + 1
        for f in visible:
            faces.discard(f)
        for (i, j) in edges:
            if (j, i) not in edges:  # a horizon edge: its other face stays
                faces.add((i, j, p))
    used = sorted({v for f in faces for v in f})
    index = {v: n for n, v in enumerate(used)}
    return [pts[v] for v in used], sorted(tuple(index[v] for v in f) for f in faces)


def link_hull(meshes: Iterable[Sequence[float]],
              directions: int = 64) -> Tuple[List[Point], List[Tuple[int, int, int]]]:
    """One hull over flat point arrays (each ``[x0, y0, z0, ...]``, one frame)."""
    dirs = fibonacci_directions(directions)
    extremes: List[Point] = []
    for points in meshes:
        extremes += support_points(points, dirs)
    flat = [c for p in extremes for c in p]
    return convex_hull(support_points(flat, dirs))


def joint_neighbours(edges: Iterable[Tuple[str, str]], distance: int) -> Set[FrozenSet[str]]:
    """Pairs of nodes at most ``distance`` edges apart (breadth-first from each node)."""
    graph: Dict[str, Set[str]] = {}
    for a, b in edges:
        if a != b:
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)
    pairs: Set[FrozenSet[str]] = set()
    for start in graph:
        frontier, seen = {start}, {start}
        for _ in range(distance):
            frontier = {n for f in frontier for n in graph[f]} - seen
            seen |= frontier
        pairs |= {frozenset((start, n)) for n in seen if n != start}
    return pairs


__all__ = ["fibonacci_directions", "support_points", "convex_hull", "link_hull", "joint_neighbours"]
