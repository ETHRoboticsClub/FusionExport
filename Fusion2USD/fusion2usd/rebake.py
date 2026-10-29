"""Re-apply the collision stage to an existing fusion2usd export, without Fusion.

``python -m fusion2usd.rebake robot.usda --overrides overrides.json --filter-joint-distance 2``

Rewrites what the export options ``collision_geometry: link_hull``, ``filter_joint_distance`` and the overrides'
``collision``/``collision_separate`` link keys and ``armature``/``friction`` joint keys would have written, and
nothing else: one convex hull per link (``fusion2usd.collision.link_hull``, in the link frame) in place of its
per-mesh colliders, filtered pairs for links up to that many joints apart, and the ball joints' dynamics. The same
functions run inside a Fusion export, so a re-export with those options gives the same colliders and filters.
Needs ``usd-core`` (pxr). Edits are made in the layer that already holds each spec (meshes in a layered export's
mesh layer, physics in the root layer); the files are saved in place.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional, Set

from .collision import joint_neighbours, link_hull
from .kinematics.properties import OverrideSet, _layered


def _source(prim) -> str:
    return str(prim.GetCustomDataByKey("source") or "")


def rebake(path: str, overrides: Optional[str], filter_joint_distance: int, hull_directions: int,
           self_collisions: Optional[bool]) -> Dict[str, List[str]]:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(path)
    rules = OverrideSet.load(overrides)
    root_layer = stage.GetRootLayer()
    report: Dict[str, List[str]] = {"hulls": [], "no_collision": [], "separate": [], "filtered": [], "joints": []}
    links = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]

    def spec_layer(prim_path: Sdf.Path) -> Sdf.Layer:
        for layer in stage.GetLayerStack():
            if layer.GetPrimAtPath(prim_path):
                return layer
        return root_layer

    for link in links:
        defaults, globbed, exact = rules.for_link([_source(link), link.GetName()])
        raw = _layered(defaults, globbed, exact)
        mode = str(raw.get("collision", raw.get("collision_approximation", "")) or "").replace("_", "").lower()
        separate = raw.get("collision_separate") or []
        separate = set(separate if isinstance(separate, (list, tuple)) else str(separate).split(","))
        scope = stage.GetPrimAtPath(link.GetPath().AppendChild("collisions"))
        if not scope:
            continue
        colliders = [p for p in scope.GetChildren() if p.HasAPI(UsdPhysics.CollisionAPI)]
        if not colliders:
            continue
        layer = spec_layer(colliders[0].GetPath())
        stage.SetEditTarget(Usd.EditTarget(layer))
        if mode == "none":
            for p in colliders:
                stage.RemovePrim(p.GetPath())
            report["no_collision"].append(link.GetName())
            continue
        frame = UsdGeom.Xformable(link).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
        clouds = []
        for p in colliders:
            if p.GetName() in separate:
                UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexHull")
                report["separate"].append(f"{link.GetName()}/{p.GetName()}")
                continue
            to_link = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * frame
            clouds.append([c for v in UsdGeom.Mesh(p).GetPointsAttr().Get() for c in to_link.Transform(Gf.Vec3d(v))])
            stage.RemovePrim(p.GetPath())
        if not clouds:
            continue
        vertices, faces = link_hull(clouds, hull_directions)
        hull = UsdGeom.Mesh.Define(stage, scope.GetPath().AppendChild("hull"))
        hull.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices])
        hull.CreateFaceVertexCountsAttr([3] * len(faces))
        hull.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
        lo = [min(v[k] for v in vertices) for k in range(3)]
        hi = [max(v[k] for v in vertices) for k in range(3)]
        hull.CreateExtentAttr([Gf.Vec3f(*lo), Gf.Vec3f(*hi)])
        hull.CreateSubdivisionSchemeAttr("none")
        UsdPhysics.CollisionAPI.Apply(hull.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(hull.GetPrim()).CreateApproximationAttr("convexHull")
        report["hulls"].append(f"{link.GetName()}: {len(vertices)} vertices from {len(clouds)} meshes")

    stage.SetEditTarget(Usd.EditTarget(root_layer))
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    edges = []
    for j in joints:
        targets = [UsdPhysics.Joint(j).GetBody0Rel().GetTargets(), UsdPhysics.Joint(j).GetBody1Rel().GetTargets()]
        if all(len(t) == 1 for t in targets):
            edges.append((str(targets[0][0]), str(targets[1][0])))
    if filter_joint_distance > 1:
        existing: Dict[str, Set[str]] = {}
        for link in links:
            if link.HasAPI(UsdPhysics.FilteredPairsAPI):
                existing[str(link.GetPath())] = {str(t) for t in
                                                 UsdPhysics.FilteredPairsAPI(link).GetFilteredPairsRel().GetTargets()}
        for a, b in sorted(tuple(sorted(p)) for p in joint_neighbours(edges, filter_joint_distance)):
            if a in existing.get(b, set()) or b in existing.get(a, set()):
                continue
            prim = stage.GetPrimAtPath(b)  # as the export does: on the later link, targeting the other
            UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().AddTarget(Sdf.Path(a))
            existing.setdefault(b, set()).add(a)
            report["filtered"].append(f"{Sdf.Path(a).name} | {Sdf.Path(b).name}")
    for j in joints:
        if not j.IsA(UsdPhysics.SphericalJoint):
            continue
        defaults, globbed, exact = rules.for_joint([_source(j), j.GetName()])
        raw = _layered(defaults, globbed, exact)
        values = {"newton:armature": raw.get("armature"), "newton:friction": raw.get("friction")}
        values = {k: float(v) for k, v in values.items() if v is not None}
        if values:
            j.AddAppliedSchema("NewtonJointAPI")  # by name: Newton's schemas need not be registered
            for name, value in values.items():
                j.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(value)
            report["joints"].append(f"{j.GetName()}: {values}")
    if self_collisions is not None:
        for p in stage.Traverse():
            if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                p.CreateAttribute("newton:selfCollisionEnabled", Sdf.ValueTypeNames.Bool).Set(self_collisions)
    for layer in stage.GetLayerStack():
        if layer.dirty and not layer.anonymous:
            layer.Save()
    return report


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="fusion2usd.rebake", description=__doc__.splitlines()[0])
    p.add_argument("usd", help="a fusion2usd export (its root .usda/.usd)")
    p.add_argument("--overrides", help="the overrides JSON of the export (link collision keys, joint dynamics)")
    p.add_argument("--filter-joint-distance", type=int, default=1)
    p.add_argument("--hull-directions", type=int, default=64)
    sc = p.add_mutually_exclusive_group()
    sc.add_argument("--self-collisions", dest="self_collisions", action="store_true", default=None)
    sc.add_argument("--no-self-collisions", dest="self_collisions", action="store_false")
    args = p.parse_args(argv)
    report = rebake(args.usd, args.overrides, args.filter_joint_distance, args.hull_directions, args.self_collisions)
    for key, rows in report.items():
        print(f"{key}: {len(rows)}")
        for row in rows:
            print(f"  {row}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
