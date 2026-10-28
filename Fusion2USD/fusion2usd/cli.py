"""Offline re-export of a dumped model: ``python -m fusion2usd.cli robot.model.json -o robot.usda``."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from .actions import registered_actions
from .exporter import Exporter, ModelFileSource
from .options import CollisionApproximation, ExportOptions, PhysicsEngine, ZeroPose


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fusion2usd", description=__doc__)
    p.add_argument("model", help="raw model JSON written by the Fusion add-in (write_model_json)")
    p.add_argument("-o", "--output", required=True, help="output .usda/.usd/.usdc")
    p.add_argument("--options", help="ExportOptions JSON (as saved by the add-in)")
    p.add_argument("--mesh-layer", action="store_true", help="write meshes to a sublayer (.usdc when pxr is available)")
    p.add_argument("--name", help="robot name")
    p.add_argument("--engine", choices=[e.value for e in PhysicsEngine], help="physics engine the USD is for")
    p.add_argument("--zero-pose", choices=[z.value for z in ZeroPose])
    base = p.add_mutually_exclusive_group()
    base.add_argument("--fixed-base", action="store_true", default=None)
    base.add_argument("--floating-base", dest="fixed_base", action="store_false")
    p.add_argument("--base-link")
    p.add_argument("--collision", choices=[c.value for c in CollisionApproximation])
    p.add_argument("--overrides", help="per-joint/link overrides JSON")
    p.add_argument("--action", action="append", default=[],
                   help=f"post-export action (repeatable): {', '.join(registered_actions())} or module:Class")
    p.add_argument("--urdf-mesh-uri", help="the URDF folder as meshes reference it, e.g. package://pkg/urdf")
    p.add_argument("--no-isaaclab-cfg", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def options_from_args(args: argparse.Namespace) -> ExportOptions:
    if args.options:
        with open(args.options, "r", encoding="utf-8") as f:
            opts = ExportOptions.from_json(f.read())
    else:
        opts = ExportOptions()
    opts.output_path = args.output
    if args.mesh_layer:
        opts.mesh_layer = True
    if args.name:
        opts.robot_name = args.name
    if args.engine:
        opts.physics_engine = PhysicsEngine(args.engine)
    if args.zero_pose:
        opts.zero_pose = ZeroPose(args.zero_pose)
    if args.fixed_base is not None:
        opts.fixed_base = args.fixed_base
    if args.base_link:
        opts.base_link = args.base_link
    if args.collision:
        opts.collision_approximation = CollisionApproximation(args.collision)
    if args.overrides:
        opts.overrides_path = args.overrides
    if args.urdf_mesh_uri is not None:
        opts.urdf_mesh_uri = args.urdf_mesh_uri
    if args.no_isaaclab_cfg:
        opts.write_isaaclab_cfg = False
    if "urdf" in args.action:
        opts.write_urdf = True
    opts.post_actions += [a for a in args.action if a != "urdf"]
    opts.write_model_json = False
    return opts


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    report = Exporter(options_from_args(args)).export(ModelFileSource(args.model))
    print(report.summary())
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
