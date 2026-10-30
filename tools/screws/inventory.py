"""Inventory the screws an exported robot is missing, from the hole geometry of its visual meshes.

    python tools/screws/inventory.py humanoid_V1.usda -o tools/screws/AddMissingScrews

Pipeline: find cylindrical holes per body -> group coaxial touching holes into stacks -> cast a ray
out of each stack's mouths to see what they open onto -> pick size/length/head side per stack.
Writes ``screws.json`` (the AddMissingScrews Fusion script compares its own B-Rep findings against it) and ``screws.csv``.

Coordinates are stage world coordinates in meters. For fusion2usd exports whose base link holds
the root component (cad_frame_rotation identity), that is Fusion's root frame at the exported pose.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from pxr import Usd, UsdGeom

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "AddMissingScrews"))
import screwlogic  # noqa: E402
from screwlogic import apply_interfaces, clearance_size, std_length, tap_size  # noqa: E402,F401

SHARP_COS = np.cos(np.radians(50))  # edges sharper than this bound faces (hole rim, flat bottom)
CSK_SHARP_COS = np.cos(np.radians(40))
R_MIN, R_MAX = 0.0006, 0.009
RAY_MAX = 0.06


# --- meshes -----------------------------------------------------------------------------------

def visual_meshes(stage):
    """Yields (link, body, world points, triangles) for every visual mesh."""
    cache = UsdGeom.XformCache()
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or "/visuals/" not in str(prim.GetPath()):
            continue
        mesh = UsdGeom.Mesh(prim)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        if not (counts == 3).all():
            raise ValueError(f"{prim.GetPath()}: expected triangles")
        pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        m = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        tris = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
            tris = tris[:, ::-1]
        path = str(prim.GetPath()).split("/")
        yield path[2], path[-1], pts @ m[:3, :3] + m[3, :3], tris


def weld(pts, tris, tol=1e-6):
    _, first, inv = np.unique(np.round(pts / tol).astype(np.int64), axis=0,
                              return_index=True, return_inverse=True)
    return pts[first], inv.reshape(-1)[tris]


# --- hole detection ---------------------------------------------------------------------------

def fit_circle(p):
    c0 = p.mean(0)
    _, _, vt = np.linalg.svd(p - c0)
    u, v, n = vt
    x, y = (p - c0) @ u, (p - c0) @ v
    (a, b, c), *_ = np.linalg.lstsq(np.c_[2 * x, 2 * y, np.ones_like(x)], x * x + y * y, rcond=None)
    r = np.sqrt(c + a * a + b * b)
    center = c0 + a * u + b * v
    rel = p - center
    res = np.abs(np.linalg.norm(rel - np.outer(rel @ n, n), axis=1) - r).max()
    return center, n, r, res, np.abs((p - c0) @ n).max()


def sharp_loops(tris, fn, sharp_cos=None):
    """Closed loops of sharp edges, with the two faces on each edge."""
    sharp_cos = SHARP_COS if sharp_cos is None else sharp_cos
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    ef = np.tile(np.arange(len(tris)), 3)
    es = np.sort(e, axis=1)
    order = np.lexsort((es[:, 1], es[:, 0]))
    es, ef = es[order], ef[order]
    i = np.nonzero((es[1:] == es[:-1]).all(1))[0]
    edges, fa, fb = es[i], ef[i], ef[i + 1]
    sharp = (fn[fa] * fn[fb]).sum(1) < sharp_cos

    neighbors = defaultdict(list)
    for a, b in zip(fa, fb):
        neighbors[a].append(b)
        neighbors[b].append(a)

    sedges, sfaces = edges[sharp], np.c_[fa[sharp], fb[sharp]]
    adj = defaultdict(list)
    for k, (a, b) in enumerate(sedges):
        adj[a].append((b, k))
        adj[b].append((a, k))
    seen, loops = set(), []
    for start in list(adj):
        if start in seen or len(adj[start]) != 2:
            continue
        verts, stack, simple, eds = [], [start], True, set()
        seen.add(start)
        while stack:
            x = stack.pop()
            verts.append(x)
            simple &= len(adj[x]) == 2
            for y, k in adj[x]:
                eds.add(k)
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        if simple and 6 <= len(verts) <= 400:
            loops.append((verts, [f for k in eds for f in sfaces[k]], len(eds)))
    return loops, neighbors


def detect_holes(pts, tris):
    """Cylindrical holes of one welded body: list of dicts with p0, axis, d, depth (meters)."""
    v0, v1, v2 = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(fn, axis=1)
    fn[norm > 1e-14] /= norm[norm > 1e-14, None]
    cent = (v0 + v1 + v2) / 3

    # A countersink's rims are 45 deg edges: the second pass finds the bore below one. It only adds
    # cylinders the first pass missed (on coarse walls a lower threshold breaks the rim loops).
    loops, neighbors = sharp_loops(tris, fn)
    loops += sharp_loops(tris, fn, CSK_SHARP_COS)[0]
    holes, used = [], set()
    for verts, faces, n_edges in loops:
        center, n, r, res, planar = fit_circle(pts[verts])
        if not (R_MIN <= r <= R_MAX) or res > 0.04 * r + 2e-5 or planar > 0.05 * r + 2e-5:
            continue
        wall = [f for f in faces if abs(fn[f] @ n) < 0.3]
        if len(wall) < 0.8 * n_edges:
            continue
        radial = cent[wall] - center
        radial -= np.outer(radial @ n, n)
        if (np.einsum("ij,ij->i", radial, fn[wall]) < 0).mean() < 0.8:
            continue  # wall normals point away from the axis: a boss, not a hole
        region, stack = set(wall), list(wall)
        while stack:
            for g in neighbors[stack.pop()]:
                if g in region or abs(fn[g] @ n) > 0.3:
                    continue
                d = cent[g] - center
                if abs(np.linalg.norm(d - (d @ n) * n) - r) > 0.15 * r + 5e-5:
                    continue
                region.add(g)
                stack.append(g)
        if used & region:
            continue  # the other rim of a cylinder we already have
        used |= region
        t = (pts[np.unique(tris[list(region)])] - center) @ n
        hole = dict(p0=center + t.min() * n, axis=n, d=2 * r, depth=float(t.max() - t.min()))
        hole.update(countersink(pts, tris, fn, cent, neighbors, region, center, n, r, t.min(), t.max()))
        holes.append(hole)
    return holes


def countersink(pts, tris, fn, cent, neighbors, region, center, n, r, t0, t1):
    """{'csk_end': 'lo'|'hi', 'csk_d': Ø} if a cone opens out from one end of the bore."""
    seeds = {g for f in region for g in neighbors[f] if g not in region and 0.5 < abs(fn[g] @ n) < 0.87}
    for end, t_end in (("lo", t0), ("hi", t1)):
        cone = {g for g in seeds if abs((cent[g] - center) @ n - t_end) < r}
        stack = list(cone)
        while stack:
            for g in neighbors[stack.pop()]:
                if g not in cone and g not in region and 0.5 < abs(fn[g] @ n) < 0.87 \
                        and abs((cent[g] - center) @ n - t_end) < 3 * r:
                    cone.add(g)
                    stack.append(g)
        if len(cone) >= 6:
            rel = pts[np.unique(tris[list(cone)])] - center
            d = 2 * np.linalg.norm(rel - np.outer(rel @ n, n), axis=1).max()
            if d >= 1.6 * 2 * r:  # smaller: an edge chamfer, not a countersink
                return dict(csk_end=end, csk_d=float(d))
    return {}


# --- ray casting ------------------------------------------------------------------------------

class Scene:
    def __init__(self, meshes):
        v0, e1, e2, owner, self.names = [], [], [], [], []
        for link, body, pts, tris in meshes:
            a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
            v0.append(a)
            e1.append(b - a)
            e2.append(c - a)
            owner.append(np.full(len(tris), len(self.names)))
            self.names.append((link, body))
        self.v0, self.e1, self.e2, self.owner = map(np.concatenate, (v0, e1, e2, owner))
        self.lo = np.minimum(self.v0, np.minimum(self.v0 + self.e1, self.v0 + self.e2))
        self.hi = np.maximum(self.v0, np.maximum(self.v0 + self.e1, self.v0 + self.e2))

    def cast(self, o, d):
        """First hit along the ray as (distance, (link, body)), or None."""
        end = o + d * RAY_MAX
        m = np.nonzero(((self.hi >= np.minimum(o, end) - 1e-4) &
                        (self.lo <= np.maximum(o, end) + 1e-4)).all(1))[0]
        if not len(m):
            return None
        v0, e1, e2 = self.v0[m], self.e1[m], self.e2[m]
        p = np.cross(d, e2)
        det = (e1 * p).sum(1)
        ok = np.abs(det) > 1e-18
        inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
        s = o - v0
        u = (s * p).sum(1) * inv
        q = np.cross(s, e1)
        v = (q @ d) * inv
        t = (e2 * q).sum(1) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 2e-6) & (t < RAY_MAX)
        if not hit.any():
            return None
        k = np.argmin(np.where(hit, t, np.inf))
        return float(t[k]), self.names[self.owner[m[k]]]


def mirror_mismatches(sites):
    """Left/right link pairs whose screw counts differ, as {(left, right): {(size, length): (nl, nr)}}."""
    counts = defaultdict(Counter)
    for s in sites:
        counts[s["link"]][(s["size"], s["length"])] += 1
    out = {}
    for link in sorted(counts):
        m = re.search(r"_l_(Mirror_)?", link)
        if not m or m.group(1):
            continue
        for other in (link[:m.start()] + "_r_" + link[m.end():], link[:m.start()] + "_l_Mirror_" + link[m.end():]):
            if other in counts:
                a, b = counts[link], counts[other]
                diff = {k: (a[k], b[k]) for k in sorted(set(a) | set(b)) if a[k] != b[k]}
                if diff:
                    out[(link, other)] = diff
    return out


# --- main -------------------------------------------------------------------------------------

def run(stage_path):
    meshes = list(visual_meshes(Usd.Stage.Open(stage_path)))
    holes = []
    for link, body, pts, tris in meshes:
        for h in detect_holes(*weld(pts, tris)):
            h.update(p0=tuple(map(float, h["p0"])), axis=tuple(map(float, h["axis"])), link=link, body=body)
            holes.append(h)
    scene = Scene(meshes)
    stacks, sites = screwlogic.run(holes, lambda o, d: scene.cast(np.array(o), np.array(d)))
    return holes, stacks, sites


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage")
    ap.add_argument("-o", "--out", default=".")
    args = ap.parse_args()

    holes, stacks, sites = run(args.stage)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "screws.json"), "w") as f:
        json.dump(dict(source=os.path.abspath(args.stage), units="m", sites=sites), f, indent=1)
    cols = ["id", "link", "bodies", "mates", "head", "size", "length", "kind", "confidence", "clamp_mm", "hole_d_mm",
            "pattern", "note", "head_m", "axis"]
    with open(os.path.join(args.out, "screws.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sites)

    print(f"{len(holes)} holes, {len(stacks)} stacks -> {len(sites)} screw sites "
          f"{dict(Counter(s['confidence'] for s in sites))}")
    bom = Counter((s["size"], s["length"]) for s in sites if s["confidence"] != "excluded")
    for (size, length), n in sorted(bom.items(), key=lambda kv: (float(kv[0][0][1:]), kv[0][1])):
        print(f"  {size}x{length:<3d} x{n}")
    for (left, right), diff in mirror_mismatches(sites).items():
        print(f"  L/R mismatch {left} vs {right}: " +
              ", ".join(f"{k[0]}x{k[1]} {a}/{b}" for k, (a, b) in diff.items()))


if __name__ == "__main__":
    main()
