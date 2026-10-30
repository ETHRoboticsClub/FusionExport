"""Conservative, dependency-free collision reduction in link coordinates.

Only closed, consistently wound convex meshes provide containment certificates.
No vertex welding/rounding, sampling-based containment, or cross-link merging.
Uncertifiable geometry is retained. Primitive fits are INSIDE the input polytope.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import itertools
import math

from .core.math3d import Quat, Vec3, mat_from_columns

# Numerical roundoff allowance, not the user-facing fitting tolerance (SI metres).
EPS = 1e-10
MAX_PLANE_TESTS = 2_000_000


def vertices(mesh):
    return [Vec3(*mesh.points[i:i + 3]) for i in range(0, len(mesh.points), 3)]


def triangles(mesh):
    pts = [p.as_tuple() for p in vertices(mesh)]
    return [tuple(pts[i] for i in mesh.indices[j:j + 3]) for j in range(0, len(mesh.indices), 3)]


def signature(mesh):
    return tuple(sorted(tuple(sorted(t)) for t in triangles(mesh)))


def valid_mesh(mesh):
    return (len(mesh.points) % 3 == 0 and len(mesh.indices) % 3 == 0
            and all(math.isfinite(x) for x in mesh.points)
            and all(0 <= i < mesh.vertex_count for i in mesh.indices))


def convex_planes(mesh):
    """Return supporting planes only for a closed, connected convex solid."""
    if not valid_mesh(mesh) or not mesh.indices:
        return None
    ts = triangles(mesh)
    pts = list({p for t in ts for p in t})
    if len(pts) < 4 or len(pts) * len(ts) > MAX_PLANE_TESTS:
        return None
    edges = Counter((t[i], t[(i + 1) % 3]) for t in ts for i in range(3))
    if any(a == b or count != 1 or edges[b, a] != 1 for (a, b), count in edges.items()):
        return None
    # A manifold may have multiple shells (including an internal cavity).
    adjacency = {p: set() for p in pts}
    for a, b in edges:
        adjacency[a].add(b)
    seen, pending = set(), [pts[0]]
    while pending:
        p = pending.pop()
        if p not in seen:
            seen.add(p)
            pending.extend(adjacency[p] - seen)
    if len(seen) != len(pts):
        return None
    points = [Vec3(*p) for p in pts]
    center = sum(points, Vec3()) / len(points)
    planes = []
    orientation = None
    for t in ts:
        a, b, c = (Vec3(*p) for p in t)
        n = (b - a).cross(c - a)
        length = n.norm()
        if length <= 1e-20:
            return None
        n = n / length
        interior = n.dot(center - a)
        if abs(interior) <= EPS:
            return None
        sign = 1 if interior < 0 else -1
        if orientation is not None and sign != orientation:
            return None
        orientation = sign
        n = n * sign
        # Relative coordinates avoid cancellation for off-origin links.
        if any(n.dot(p - a) > EPS for p in points):
            return None
        planes.append((n, a))
    return planes


def contains(planes, mesh):
    return all(n.dot(p - a) <= EPS for p in vertices(mesh) for n, a in planes)


def merge_convex(one, two, p1=None, p2=None):
    """Remove matching planar interfaces, then certify the union is convex.

    Handles different diagonals on a shared face; unmatched subdivisions or
    overlapping solids are deliberately left alone.
    """
    p1 = convex_planes(one) if p1 is None else p1
    p2 = convex_planes(two) if p2 is None else p2
    if p1 is None or p2 is None:
        return None
    ts1, ts2 = triangles(one), triangles(two)

    def faces(ts, planes):
        groups = []
        for i, (n, a) in enumerate(planes):
            for normal, anchor, indices in groups:
                if n.dot(normal) > 1 - 1e-12 and abs(normal.dot(a - anchor)) <= EPS:
                    indices.append(i)
                    break
            else:
                groups.append((n, a, [i]))
        return [(n, frozenset(p for i in ids for p in ts[i]), ids) for n, _, ids in groups]

    remove1, remove2 = set(), set()
    for n, points, ids in faces(ts1, p1):
        for m, other, js in faces(ts2, p2):
            if points == other and n.dot(m) < -1 + 1e-12:
                remove1.update(ids)
                remove2.update(js)
    if not remove1:
        return None
    # Orient both shells outward before joining (input can be inward wound).
    remaining = []
    for ts, planes, removed in ((ts1, p1, remove1), (ts2, p2, remove2)):
        for i, t in enumerate(ts):
            if i not in removed:
                a, b, c = (Vec3(*p) for p in t)
                remaining.append(t if (b - a).cross(c - a).dot(planes[i][0]) > 0 else t[::-1])
    coords = list(dict.fromkeys(p for t in remaining for p in t))
    lookup = {p: i for i, p in enumerate(coords)}
    merged = replace(one, points=[v for p in coords for v in p],
                     indices=[lookup[p] for t in remaining for p in t], normals=None, primitive=None)
    return merged if convex_planes(merged) is not None else None


def fit_primitive(mesh, planes, tolerance):
    """Certify an inscribed box/sphere/capsule and a maximum inward distance.

    Distance to a convex primitive is convex, so testing all polytope vertices
    bounds the error over its entire solid, not just the sampled surface.
    """
    pts = vertices(mesh)
    lo, hi = mesh.bounds()
    center = (lo + hi) * 0.5
    identity = [1.0, 0.0, 0.0, 0.0]
    # A box needs three orthogonal face directions and all eight corners inside.
    axes = []
    for n, _ in planes:
        if all(abs(n.dot(a)) < 1e-10 for a in axes):
            axes.append(n)
        if len(axes) == 2:
            axes.append(axes[0].cross(axes[1]).normalized())
            break
    if len(axes) == 3:
        bounds = [(min(p.dot(a) for p in pts), max(p.dot(a) for p in pts)) for a in axes]
        c = sum((a * ((low + high) / 2) for a, (low, high) in zip(axes, bounds)), Vec3())
        half = [(high - low) / 2 for low, high in bounds]
        corners = [c + sum((a * (s * r) for a, s, r in zip(axes, signs, half)), Vec3())
                   for signs in itertools.product((-1, 1), repeat=3)]
        if all(n.dot(p - anchor) <= EPS for n, anchor in planes for p in corners):
            return dict(kind="box", center=list(c), rotation=list(Quat.from_matrix(mat_from_columns(*axes)).as_tuple()),
                        size=[2 * r for r in half], error=0.0)
    radius = min(n.dot(a - center) for n, a in planes) - EPS
    if radius > 0:
        error = max((p - center).norm() for p in pts) - radius
        if error <= tolerance:
            return dict(kind="sphere", center=list(center), rotation=identity, radius=radius, error=error)
    # Candidate axes are link axes plus long vertex directions; certification,
    # rather than the heuristic, decides whether a candidate may be used.
    directions = [Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)]
    farthest = sorted(pts, key=lambda p: (p - center).norm(), reverse=True)[:6]
    directions += [(p - center).normalized() for p in farthest if (p - center).norm() > EPS]
    for axis in directions:
        axial = max(abs((p - center).dot(axis)) for p in pts)
        radial = max(((p - center) - axis * (p - center).dot(axis)).norm() for p in pts)
        half = axial - radial
        if half <= EPS:
            continue
        radius = min(n.dot(a - center) - half * abs(n.dot(axis)) for n, a in planes) - EPS
        if radius <= 0:
            continue
        error = max(((p - center) - axis * max(-half, min(half, (p - center).dot(axis)))).norm()
                    for p in pts) - radius
        if error <= tolerance:
            x = axis.any_perpendicular()
            q = Quat.from_matrix(mat_from_columns(x, axis.cross(x), axis))
            return dict(kind="capsule", center=list(center), rotation=list(q.as_tuple()),
                        radius=radius, height=2 * half, error=error)
    return None


def optimize_collisions(model, options, report):
    if not options.export_collisions or not options.optimize_collisions:
        return model
    tolerance = options.collision_primitive_tolerance
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("collision_primitive_tolerance must be finite and nonnegative (metres)")
    for link in model.links.values():
        approximation = link.collision_approximation or options.collision_approximation
        approximation = getattr(approximation, "value", approximation)
        # Explicit approximation choices retain their authored semantics.
        if approximation not in ("convexDecomposition", "convexHull"):
            continue
        original = link.collisions or link.visuals
        if not original:
            continue
        meshes, seen = [], set()
        for mesh in original:
            # Keep invalid meshes for existing writer diagnostics, without probing.
            if not valid_mesh(mesh):
                meshes.append(mesh)
                continue
            key = signature(mesh)
            if key not in seen:
                meshes.append(mesh)
                seen.add(key)
        duplicates = len(original) - len(meshes)
        contained = merged_count = 0
        certificates = {id(m): convex_planes(m) for m in meshes}
        valid = {id(m): valid_mesh(m) for m in meshes}
        bounds = {id(m): m.bounds() for m in meshes if m.vertex_count}

        def covered_by(mesh, other):
            if certificates[id(other)] is None:
                return False
            lo, hi = bounds[id(mesh)]
            outer_lo, outer_hi = bounds[id(other)]
            return (all(a >= b - EPS and c <= d + EPS for a, b, c, d in zip(lo, outer_lo, hi, outer_hi))
                    and contains(certificates[id(other)], mesh))

        i = 0
        while i < len(meshes):
            mesh = meshes[i]
            if (mesh.vertex_count and valid[id(mesh)] and any(other is not mesh and covered_by(mesh, other) for other in meshes)):
                meshes.pop(i)
                contained += 1
            else:
                i += 1
        changed = True
        while changed:
            changed = False
            for i in range(len(meshes)):
                for j in range(i + 1, len(meshes)):
                    if certificates[id(meshes[i])] is None or certificates[id(meshes[j])] is None:
                        continue
                    # Broad phase only; never used as a containment certificate.
                    lo1, hi1 = meshes[i].bounds()
                    lo2, hi2 = meshes[j].bounds()
                    if any(a > b or c > d for a, b, c, d in zip(lo1, hi2, lo2, hi1)):
                        continue
                    merged = merge_convex(meshes[i], meshes[j], certificates[id(meshes[i])], certificates[id(meshes[j])])
                    if merged is not None:
                        meshes[i] = merged
                        meshes.pop(j)
                        certificates[id(merged)] = convex_planes(merged)
                        merged_count += 1
                        changed = True
                        break
                if changed:
                    break
        out, primitives, uncertified = [], Counter(), 0
        maximum_error = 0.0
        for mesh in meshes:
            planes = certificates[id(mesh)]
            if planes is None:
                uncertified += 1
                out.append(mesh)
                continue
            primitive = fit_primitive(mesh, planes, tolerance)
            out.append(replace(mesh, primitive=primitive) if primitive else mesh)
            if primitive:
                primitives[primitive["kind"]] += 1
                maximum_error = max(maximum_error, primitive["error"])
        link.collisions = out
        report.info(f"Collision {link.name}: {len(original)} -> {len(out)} shapes before cooking; "
                    f"{duplicates} duplicates, {contained} contained, {merged_count} convex merges; "
                    f"primitives {dict(primitives)}, maximum primitive inward error {maximum_error:.6g} m")
        if uncertified:
            report.warn(f"Collision {link.name}: {uncertified} meshes not certified convex (concave, invalid, "
                        "or verification budget exceeded), retained without fitting/merging. Simulator "
                        "approximation may fill holes/socket clearances; inspect cooked colliders.")
    return model
