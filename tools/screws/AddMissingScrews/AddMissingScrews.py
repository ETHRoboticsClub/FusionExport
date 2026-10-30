"""Finds the holes in the active design that should have a screw, and adds the screws.

Holes come from the B-Rep: every cylindrical face of every visible solid, with countersinks from
coaxial cones. screwlogic.py (shared with tools/screws/inventory.py) stacks coaxial holes, reads
what each stack's mouths open onto and decides size, length and head side. Screws are ISO 4762
socket head or ISO 10642 countersunk, rigidly grouped with the part whose hole the head sits in,
and collected in a "Fasteners (auto)" component; running the script again offers to remove them.

A dry run writes the reports only: placement_report.csv (every site) and compare_report.csv
(against screws.json from the USD inventory, when present).
"""

import csv
import importlib
import json
import math
import os
import sys
import traceback

import adsk.core
import adsk.fusion

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import screwlogic  # noqa: E402

importlib.reload(screwlogic)  # Fusion keeps modules loaded between runs

R_MIN_CM, R_MAX_CM = 0.06, 0.9  # hole radii considered
RAY_MAX_CM = 6.0
CONTAINER = "Fasteners (auto)"  # used by earlier versions; removed with the screws
ATTR_GROUP, ATTR_SITE = "AddMissingScrews", "site"
RIGID_PREFIX = "screws: "
CONFIDENCE = ("high", "medium")  # add "low" to also place the uncertain sites
MATERIALS = ("Steel", "Steel, Alloy", "Stainless Steel")

# ISO 4762: head diameter dk, head height k, socket width s, socket depth t (mm)
ISO4762 = {2: (3.8, 2.0, 1.5, 1.0), 2.5: (4.5, 2.5, 2.0, 1.1), 3: (5.5, 3.0, 2.5, 1.3),
           4: (7.0, 4.0, 3.0, 2.0), 5: (8.5, 5.0, 4.0, 2.5), 6: (10.0, 6.0, 5.0, 3.0),
           8: (13.0, 8.0, 6.0, 4.0)}
# ISO 10642 countersunk: head diameter dk, head height k, socket width s, socket depth t (mm).
# Its nominal length is overall, head included.
ISO10642 = {3: (6.72, 1.86, 2.0, 1.1), 4: (8.96, 2.48, 2.5, 1.5), 5: (11.2, 3.1, 3.0, 1.9),
            6: (13.44, 3.72, 4.0, 2.2), 8: (17.92, 4.96, 5.0, 3.0)}


def head_dims(site):
    """(dk, k, s, t) in mm for the site's head type."""
    d = nominal(site["size"])
    if site.get("head") == "countersunk" and d in ISO10642:
        return ISO10642[d]
    return ISO4762[d]


def nominal(size):
    return float(size.lstrip("M"))


def select_sites(sites, confidence=CONFIDENCE):
    return [s for s in sites if s["confidence"] in confidence and nominal(s["size"]) in ISO4762]


def perpendicular(a):
    """A unit vector perpendicular to unit vector ``a``."""
    ref = (1.0, 0.0, 0.0) if abs(a[0]) < 0.9 else (0.0, 1.0, 0.0)
    x = cross(ref, a)
    return scale(x, 1.0 / norm(x))


def screw_frame(bearing_cm, axis, head_height_cm):
    """Placement (origin, x, y, z) for a screw modeled with its head top at the origin, tip along +Z.

    ``bearing_cm`` is where the underside of the head sits, ``axis`` points from head to tip.
    """
    z = scale(axis, 1.0 / norm(axis))
    x = perpendicular(z)
    y = cross(z, x)
    origin = tuple(b - head_height_cm * zi for b, zi in zip(bearing_cm, z))
    return origin, x, y, z


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def scale(a, k):
    return tuple(x * k for x in a)


def norm(a):
    return math.sqrt(dot(a, a))


# --- Fusion -----------------------------------------------------------------------------------

def is_screw(occ):
    """Occurrences this script added: never holes, never obstacles."""
    return occ is not None and (occ.fullPathName.startswith(CONTAINER)
                                or occ.component.name.startswith(("ISO 4762", "ISO 10642")))


def body_key(body):
    occ = body.assemblyContext
    return f"{occ.fullPathName if occ else '(root)'}/{body.name}"


def solid_bodies(root):
    """{key: body} for every visible solid body (root context), screws excluded."""
    out = {}
    for b in root.bRepBodies:
        if b.isVisible and b.isSolid:
            out[body_key(b)] = b
    for occ in root.allOccurrences:
        if is_screw(occ):
            continue
        for b in occ.bRepBodies:
            if b.isVisible and b.isSolid:
                out[body_key(b)] = b
    return out


def edge_points(face):
    """Points along the face's edges (ends and middles), cm."""
    pts = []
    for e in face.edges:
        if e.isDegenerate:
            continue
        ev = e.evaluator
        ok, p0, p1 = ev.getEndPoints()
        ok2, t0, t1 = ev.getParameterExtents()
        if not (ok and ok2):
            continue
        ok, pm = ev.getPointAtParameter((t0 + t1) / 2)
        pts += [p0.asArray(), p1.asArray()] + ([pm.asArray()] if ok else [])
    return pts


def unit(v):
    return scale(v, 1.0 / norm(v))


def coaxial(o1, a1, o2, a2, tol=1e-3):
    rel = tuple(p - q for p, q in zip(o2, o1))
    return abs(dot(a1, a2)) > 0.9999 and norm(tuple(r - dot(rel, a1) * c for r, c in zip(rel, a1))) < tol


def interval(pts, o, a):
    ts = [dot(tuple(p - q for p, q in zip(pt, o)), a) for pt in pts]
    return min(ts), max(ts)


def is_hole(body, face, o, a, r):
    """True if the cylinder face bounds a void (material outside the radius), not a boss."""
    p = face.pointOnFace.asArray()
    rel = tuple(x - y for x, y in zip(p, o))
    toward = unit(tuple(x - dot(rel, a) * c for x, c in zip(rel, a)))
    q = tuple(x - min(0.01, 0.2 * r) * u for x, u in zip(p, toward))
    return body.pointContainment(adsk.core.Point3D.create(*q)) == \
        adsk.fusion.PointContainment.PointOutsidePointContainment


def collect_holes(bodies, progress=None):
    """screwlogic holes (meters) from the cylinder and cone faces of the bodies."""
    holes = []
    for n, (key, body) in enumerate(bodies.items(), 1):
        bores, cones = [], []
        for face in body.faces:
            st = face.geometry.surfaceType
            if st == adsk.core.SurfaceTypes.CylinderSurfaceType:
                cyl = adsk.core.Cylinder.cast(face.geometry)
                r = cyl.radius
                if not (R_MIN_CM <= r <= R_MAX_CM):
                    continue
                o, a = cyl.origin.asArray(), unit(cyl.axis.asArray())
                if not is_hole(body, face, o, a, r):
                    continue
                pts = edge_points(face)
                if not pts:
                    continue
                # split faces of one bore merge; separate bores on one axis (two walls) do not
                for g in bores:
                    if abs(g["r"] - r) < 1e-4 and coaxial(g["o"], g["a"], o, a):
                        g0, g1 = interval(g["pts"], g["o"], g["a"])
                        n0, n1 = interval(pts, g["o"], g["a"])
                        if n0 <= g1 + 1e-3 and n1 >= g0 - 1e-3:
                            g["pts"] += pts
                            break
                else:
                    bores.append(dict(o=o, a=a, r=r, pts=pts))
            elif st == adsk.core.SurfaceTypes.ConeSurfaceType:
                cone = adsk.core.Cone.cast(face.geometry)
                radii = [e.geometry.radius for e in face.edges
                         if e.geometry is not None  # degenerate edges (a cone's apex) have none
                         and e.geometry.curveType in (adsk.core.Curve3DTypes.Circle3DCurveType,
                                                      adsk.core.Curve3DTypes.Arc3DCurveType)]
                if radii:
                    cones.append(dict(o=cone.origin.asArray(), a=unit(cone.axis.asArray()),
                                      rmax=max(radii), pts=edge_points(face)))
        for g in bores:
            lo, hi = interval(g["pts"], g["o"], g["a"])
            hole = dict(p0=tuple(0.01 * (x + lo * c) for x, c in zip(g["o"], g["a"])), axis=g["a"],
                        d=0.02 * g["r"], depth=0.01 * (hi - lo), link="", body=key)
            for c in cones:
                if c["rmax"] < 1.6 * g["r"] or not coaxial(g["o"], g["a"], c["o"], c["a"]):
                    continue
                c0, c1 = interval(c["pts"], g["o"], g["a"])
                for end, t in (("lo", lo), ("hi", hi)):
                    if min(abs(c0 - t), abs(c1 - t)) < 0.02:
                        hole.update(csk_end=end, csk_d=0.02 * c["rmax"])
            holes.append(hole)
        if progress:
            progress(n, len(bodies))
    return holes


def make_cast(root):
    """screwlogic ray probe: first face hit (not a screw, not the bore's own wall)."""
    face_type = adsk.fusion.BRepEntityTypes.BRepFaceEntityType

    def cast(origin_m, direction):
        o = tuple(100.0 * v for v in origin_m)
        hits = adsk.core.ObjectCollection.create()
        ents = root.findBRepUsingRay(adsk.core.Point3D.create(*o), adsk.core.Vector3D.create(*direction),
                                     face_type, -1, True, hits)
        best = None
        for i in range(ents.count):
            face = ents.item(i)
            if is_screw(face.body.assemblyContext):
                continue
            dist = norm(tuple(p - q for p, q in zip(hits.item(i).asArray(), o)))
            if dist < 1e-4 or dist > RAY_MAX_CM or (best and dist >= best[0]):
                continue
            g = face.geometry
            if g.surfaceType in (adsk.core.SurfaceTypes.CylinderSurfaceType, adsk.core.SurfaceTypes.ConeSurfaceType):
                ax = unit(g.axis.asArray())
                if coaxial(o, unit(direction), g.origin.asArray(), ax):
                    continue
            best = (dist, body_key(face.body))
        return None if best is None else (best[0] / 100.0, ("", best[1]))

    return cast


def build_screw(root, site, material, transform):
    """New screw component (head top at origin, shank along +Z) and its first occurrence."""
    size, length = site["size"], site["length"]
    d = nominal(size)
    countersunk = site.get("head") == "countersunk" and d in ISO10642
    dk, k, s, t = (v / 10.0 for v in head_dims(site))
    occ = root.occurrences.addNewComponent(transform)
    comp = occ.component
    comp.name = f"ISO {'10642' if countersunk else '4762'} {size}x{length:g}"
    ops = adsk.fusion.FeatureOperations
    origin = adsk.core.Point3D.create(0, 0, 0)

    def extrude(sketch, distance, op):
        ext = comp.features.extrudeFeatures.createInput(sketch.profiles.item(0), op)
        ext.setDistanceExtent(False, adsk.core.ValueInput.createByReal(distance))
        return comp.features.extrudeFeatures.add(ext)

    plane_in = comp.constructionPlanes.createInput()
    plane_in.setByOffset(comp.xYConstructionPlane, adsk.core.ValueInput.createByReal(k))
    under_head = comp.constructionPlanes.add(plane_in)
    head = comp.sketches.add(comp.xYConstructionPlane)
    head.sketchCurves.sketchCircles.addByCenterRadius(origin, dk / 2)
    if countersunk:  # cone from the head diameter at the top down to the shank
        tip = comp.sketches.add(under_head)
        tip.sketchCurves.sketchCircles.addByCenterRadius(origin, d / 20.0)
        loft = comp.features.loftFeatures.createInput(ops.NewBodyFeatureOperation)
        loft.loftSections.add(head.profiles.item(0))
        loft.loftSections.add(tip.profiles.item(0))
        comp.features.loftFeatures.add(loft)
    else:
        extrude(head, k, ops.NewBodyFeatureOperation)

    shank = comp.sketches.add(under_head)
    shank.sketchCurves.sketchCircles.addByCenterRadius(origin, d / 20.0)
    extrude(shank, (length / 10.0 - k) if countersunk else length / 10.0, ops.JoinFeatureOperation)

    socket = comp.sketches.add(comp.xYConstructionPlane)
    lines = socket.sketchCurves.sketchLines
    rc = s / 2 / math.cos(math.pi / 6)
    pts = [adsk.core.Point3D.create(rc * math.cos(i * math.pi / 3), rc * math.sin(i * math.pi / 3), 0)
           for i in range(6)]
    for i in range(6):
        lines.addByTwoPoints(pts[i], pts[(i + 1) % 6])
    extrude(socket, t, ops.CutFeatureOperation)

    if material is not None:
        comp.material = material
    return comp, occ


def find_material(app):
    for lib in app.materialLibraries:
        for name in MATERIALS:
            m = lib.materials.itemByName(name)
            if m is not None:
                return m
    return None


def place(design, matches, ui, app):
    # Screws are created at root level, where their transform is their own: a transform set on an
    # occurrence inside a sub-component is a proxy override that reverts on recompute (that sent
    # copies back to the origin). They are then moved into one container, keeping their position.
    root = design.rootComponent
    material = find_material(app)
    comps = {}
    placed = []
    progress = ui.progressBar
    progress.show("Adding screws %v / %m", 0, len(matches), False)
    for n, (site, body, bearing) in enumerate(matches, 1):
        key = (site["size"], site["length"], site.get("head"))
        k_cm = head_dims(site)[1] / 10.0
        o, x, y, z = screw_frame(bearing, site["axis"], k_cm)
        m = adsk.core.Matrix3D.create()
        m.setWithCoordinateSystem(adsk.core.Point3D.create(*o), adsk.core.Vector3D.create(*x),
                                  adsk.core.Vector3D.create(*y), adsk.core.Vector3D.create(*z))
        if key not in comps:
            comps[key], occ = build_screw(root, site, material, m)
        else:
            occ = root.occurrences.addExistingComponent(comps[key], m)
            occ.transform2 = m  # addExistingComponent ignores its transform
        occ.attributes.add(ATTR_GROUP, ATTR_SITE, str(site["id"]))
        host = body.assemblyContext
        site["host"] = host.fullPathName if host is not None else "(root)"
        site["bearing_cm"] = " ".join(f"{v:.3f}" for v in bearing)
        box = body.boundingBox
        site["in_host_box"] = all(lo - 0.1 <= v <= hi + 0.1 for v, lo, hi in
                                  zip(bearing, box.minPoint.asArray(), box.maxPoint.asArray()))
        site["status"] = "placed"
        placed.append([site, occ, o, host])
        progress.progressValue = n
        if n % 20 == 0:
            adsk.doEvents()

    holder = root.occurrences.addNewComponent(adsk.core.Matrix3D.create())
    holder.component.name = CONTAINER
    for entry in placed:
        moved = entry[1].moveToComponent(holder)
        entry[1] = moved if moved.assemblyContext is not None else moved.createForAssemblyContext(holder)

    parametric = design.designType == adsk.fusion.DesignTypes.ParametricDesignType
    if parametric and design.snapshots.hasPendingSnapshot:
        design.snapshots.add()
    by_host = {}
    for _, occ, _, host in placed:
        if host is None:
            occ.isGrounded = True
        else:
            by_host.setdefault(host.fullPathName, [host]).append(occ)
    for host_name, occs in by_host.items():
        group = adsk.core.ObjectCollection.create()
        for o in occs:
            group.add(o)
        root.rigidGroups.add(group, False).name = (RIGID_PREFIX + host_name)[:120]
    progress.hide()
    adsk.doEvents()
    # checked after the rigid groups, so a recompute that moves screws shows up here
    for site, occ, o, _ in placed:
        now = occ.transform2.translation.asArray()
        site["placed_err_mm"] = round(norm(tuple(p - q for p, q in zip(now, o))) * 10, 3)


def remove_previous(root):
    """Deletes screws and rigid groups added by earlier runs; returns how many screws there were."""
    groups = [g for g in root.rigidGroups if g.name.startswith(RIGID_PREFIX)]
    for g in groups:
        g.deleteMe()
    occs = [o for o in root.occurrences if o.attributes.itemByName(ATTR_GROUP, ATTR_SITE) is not None]
    legacy = root.occurrences.itemByName(CONTAINER + ":1")
    count = len(occs) + (legacy.childOccurrences.count if legacy else 0)
    for o in occs:
        o.deleteMe()
    if legacy:
        legacy.deleteMe()
    return count


def has_previous(root):
    return (root.occurrences.itemByName(CONTAINER + ":1") is not None
            or any(o.attributes.itemByName(ATTR_GROUP, ATTR_SITE) is not None for o in root.occurrences))


def write_report(sites):
    path = os.path.join(HERE, "placement_report.csv")
    cols = ["id", "status", "host", "head", "size", "length", "kind", "confidence", "placed_err_mm", "in_host_box",
            "pattern", "note", "head_body", "mates", "bearing_cm", "head_m", "axis"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sites)
    return path


def compare_with_usd(sites):
    """Matches against screws.json (USD inventory): (counts, rows). Same size, heads within 1 mm."""
    path = os.path.join(HERE, "screws.json")
    if not os.path.exists(path):
        return None, []
    with open(path) as f:
        usd = select_sites(json.load(f)["sites"])
    ours = select_sites(sites)
    rows, used = [], set()
    for u in usd:
        best = None
        for i, s in enumerate(ours):
            d = norm(tuple(p - q for p, q in zip(s["head_m"], u["head_m"])))
            if s["size"] == u["size"] and d < 1e-3 and i not in used and (best is None or d < best[1]):
                best = (i, d)
        if best:
            used.add(best[0])
            s = ours[best[0]]
            if s["length"] != u["length"] or s["head"] != u.get("head", "socket"):
                rows.append(dict(match="both, differs", fusion_id=s["id"], usd_id=u["id"], size=s["size"],
                                 fusion=f"{s['head']} x{s['length']:g}", usd=f"{u.get('head')} x{u['length']:g}",
                                 where=s["head_body"], head_m=s["head_m"]))
        else:
            rows.append(dict(match="usd only", usd_id=u["id"], size=u["size"], usd=f"x{u['length']:g}",
                             where=u["link"], head_m=u["head_m"], note=u["note"]))
    for i, s in enumerate(ours):
        if i not in used:
            rows.append(dict(match="fusion only", fusion_id=s["id"], size=s["size"], fusion=f"x{s['length']:g}",
                             where=s["head_body"], head_m=s["head_m"], note=s["note"]))
    with open(os.path.join(HERE, "compare_report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, ["match", "fusion_id", "usd_id", "size", "fusion", "usd", "where", "head_m", "note"])
        w.writeheader()
        w.writerows(rows)
    counts = dict(both=len(used), usd_only=sum(r["match"] == "usd only" for r in rows),
                  fusion_only=sum(r["match"] == "fusion only" for r in rows))
    return counts, rows


def run(context):
    app = adsk.core.Application.get()
    ui = app.userInterface
    try:
        design = adsk.fusion.Design.cast(app.activeProduct)
        if design is None:
            ui.messageBox("Open the robot design first.")
            return
        root = design.rootComponent
        answer = ui.messageBox("Yes: find holes and add screws\nNo: dry run, reports only\nCancel: stop",
                               "Add missing screws", adsk.core.MessageBoxButtonTypes.YesNoCancelButtonType,
                               adsk.core.MessageBoxIconTypes.QuestionIconType)
        if answer == adsk.core.DialogResults.DialogCancel:
            return
        dry = answer == adsk.core.DialogResults.DialogNo
        if not dry and has_previous(root):
            answer = ui.messageBox("Screws from an earlier run exist. Remove them first?\n\n"
                                   "Yes: remove them, then add screws again.\nNo: stop.",
                                   "Add missing screws", adsk.core.MessageBoxButtonTypes.YesNoButtonType,
                                   adsk.core.MessageBoxIconTypes.QuestionIconType)
            if answer != adsk.core.DialogResults.DialogYes:
                return
            remove_previous(root)

        progress = ui.progressBar

        def step(label):
            def cb(i, total):
                if i == 1:
                    progress.show(label + " %v / %m", 0, total, False)
                progress.progressValue = i
                if i % 25 == 0:
                    adsk.doEvents()
            return cb

        bodies = solid_bodies(root)
        holes = collect_holes(bodies, step("Scanning bodies"))
        _, sites = screwlogic.run(holes, make_cast(root), step("Probing holes"))
        progress.hide()

        wanted = select_sites(sites)
        for s in sites:
            s["status"] = "skipped: confidence or size"
        for s in wanted:
            s["status"] = "would place" if dry else "to place"
        counts, _ = compare_with_usd(sites)
        found = (f"{len(holes)} holes, {len(sites)} screw sites, {len(wanted)} with confidence "
                 f"{'/'.join(CONFIDENCE)}.")
        if counts:
            found += (f"\nAgainst the USD inventory: {counts['both']} in both, {counts['fusion_only']} only here, "
                      f"{counts['usd_only']} only in the USD (compare_report.csv).")
        if dry:
            ui.messageBox(f"Dry run. {found}\nReport: {write_report(sites)}")
            return
        answer = ui.messageBox(f"{found}\n\nAdd {len(wanted)} screws?", "Add missing screws",
                               adsk.core.MessageBoxButtonTypes.YesNoButtonType,
                               adsk.core.MessageBoxIconTypes.QuestionIconType)
        if answer != adsk.core.DialogResults.DialogYes:
            ui.messageBox(f"Nothing added. Report: {write_report(sites)}")
            return
        matches = [(s, bodies[s["head_body"]], tuple(100.0 * v for v in s["head_m"])) for s in wanted]
        try:
            place(design, matches, ui, app)
        finally:
            report = write_report(sites)
        off = sum(1 for s, _, _ in matches if s.get("placed_err_mm", 0) > 0.01 or not s.get("in_host_box", True))
        ui.messageBox(f"Added {len(matches)} screws ({off} not where intended).\n"
                      f"Run the script again to remove them.\nReport: {report}")
    except Exception:
        ui.messageBox("Failed:\n" + traceback.format_exc())
