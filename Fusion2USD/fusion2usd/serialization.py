"""JSON (de)serialization of RobotModel, so a Fusion extraction can be re-processed offline."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .core.inertia import MassProperties
from .core.math3d import Quat, Transform, Vec3
from .model.robot import (
    Coupling, Dof, DofRef, DofType, Joint, JointProperties, Limit, Link, Material, Mesh, RobotModel, Sensor,
    SensorKind,
)

FORMAT_VERSION = 1


def _vec(v: Vec3) -> list:
    return [v.x, v.y, v.z]


def _tf(t: Optional[Transform]) -> Optional[dict]:
    if t is None:
        return None
    return {"rotation": list(t.rotation.as_tuple()), "translation": _vec(t.translation)}


def _untf(d: Optional[dict]) -> Optional[Transform]:
    if d is None:
        return None
    return Transform(Quat(*d["rotation"]), Vec3.of(d["translation"]))


class ModelSerializer:
    def to_dict(self, model: RobotModel) -> Dict[str, Any]:
        return {
            "format": "fusion2usd-model",
            "version": FORMAT_VERSION,
            "name": model.name,
            "base_link": model.base_link,
            "fixed_base": model.fixed_base,
            "grounded_links": list(model.grounded_links),
            "root_pose": _tf(model.root_pose),
            "metadata": model.metadata,
            "materials": [vars(m).copy() for m in model.materials.values()],
            "links": [self._link(l) for l in model.links.values()],
            "joints": [self._joint(j) for j in model.joints],
            "sensors": [
                {"name": s.name, "kind": s.kind.value, "link": s.link, "transform": _tf(s.transform),
                 "params": s.params, "metadata": s.metadata, "source": s.source}
                for s in model.sensors
            ],
            "couplings": [
                {"name": c.name, "leader": [c.leader.joint, c.leader.dof], "follower": [c.follower.joint, c.follower.dof],
                 "ratio": c.ratio, "offset": c.offset, "source": c.source}
                for c in model.couplings
            ],
        }

    def from_dict(self, d: Dict[str, Any]) -> RobotModel:
        if d.get("format") != "fusion2usd-model":
            raise ValueError("not a fusion2usd model file")
        model = RobotModel(
            name=d["name"], base_link=d.get("base_link"), fixed_base=d.get("fixed_base", False),
            grounded_links=list(d.get("grounded_links", [])), metadata=d.get("metadata", {}),
            root_pose=_untf(d.get("root_pose")) or Transform(),
        )
        for m in d.get("materials", []):
            m = dict(m)
            m["color"] = tuple(m["color"])
            model.add_material(Material(**m))
        for l in d.get("links", []):
            model.add_link(self._unlink(l))
        for j in d.get("joints", []):
            model.add_joint(self._unjoint(j))
        for s in d.get("sensors", []):
            params = dict(s.get("params", {}))
            if "clipping_range" in params:
                params["clipping_range"] = tuple(params["clipping_range"])
            model.sensors.append(Sensor(s["name"], SensorKind(s["kind"]), s["link"], _untf(s["transform"]) or Transform(),
                                        params, s.get("metadata", {}), s.get("source", "")))
        for c in d.get("couplings", []):
            model.couplings.append(Coupling(c["name"], DofRef(*c["leader"]), DofRef(*c["follower"]),
                                            c["ratio"], c.get("offset", 0.0), c.get("source", "")))
        return model

    def save(self, model: RobotModel, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(model), f)

    def load(self, path: str) -> RobotModel:
        with open(path, "r", encoding="utf-8") as f:
            return self.from_dict(json.load(f))

    # -- links
    @staticmethod
    def _mesh(m: Mesh) -> dict:
        return {"name": m.name, "points": m.points, "indices": m.indices, "normals": m.normals, "material": m.material}

    def _link(self, l: Link) -> dict:
        mass = None
        if l.mass is not None:
            mass = {"mass": l.mass.mass, "com": _vec(l.mass.com), "inertia": [list(r) for r in l.mass.inertia]}
        return {
            "name": l.name, "transform": _tf(l.transform), "mass": mass,
            "visuals": [self._mesh(m) for m in l.visuals], "collisions": [self._mesh(m) for m in l.collisions],
            "collision_approximation": l.collision_approximation, "collision_separate": l.collision_separate,
            "metadata": l.metadata, "source": l.source, "is_virtual": l.is_virtual,
        }

    @staticmethod
    def _unlink(d: dict) -> Link:
        mass = None
        if d.get("mass"):
            m = d["mass"]
            mass = MassProperties(m["mass"], Vec3.of(m["com"]), tuple(tuple(r) for r in m["inertia"]))  # type: ignore[arg-type]
        meshes = lambda key: [Mesh(**m) for m in d.get(key, [])]  # noqa: E731
        return Link(
            name=d["name"], transform=_untf(d["transform"]) or Transform(), mass=mass,
            visuals=meshes("visuals"), collisions=meshes("collisions"),
            collision_approximation=d.get("collision_approximation"),
            collision_separate=list(d.get("collision_separate", [])), metadata=d.get("metadata", {}),
            source=d.get("source", ""), is_virtual=d.get("is_virtual", False),
        )

    # -- joints
    @staticmethod
    def _joint(j: Joint) -> dict:
        return {
            "name": j.name, "parent": j.parent, "child": j.child, "origin": _vec(j.origin),
            "dofs": [
                {"type": d.type.value, "axis": _vec(d.axis), "value": d.value, "label": d.label,
                 "limit": None if d.limit is None else [d.limit.lower, d.limit.upper]}
                for d in j.dofs
            ],
            "properties": j.properties.to_dict(), "metadata": j.metadata, "source": j.source,
            "source_type": j.source_type, "frame_hint": _tf(j.frame_hint),
            "exclude_from_articulation": j.exclude_from_articulation,
            "loop_closure_preferred": j.loop_closure_preferred, "enabled": j.enabled,
        }

    @staticmethod
    def _unjoint(d: dict) -> Joint:
        dofs = [
            Dof(DofType(x["type"]), Vec3.of(x["axis"]), x.get("value", 0.0),
                None if x.get("limit") is None else Limit(*x["limit"]), x.get("label", ""))
            for x in d.get("dofs", [])
        ]
        return Joint(
            name=d["name"], parent=d["parent"], child=d["child"], origin=Vec3.of(d["origin"]), dofs=dofs,
            properties=JointProperties(**d.get("properties", {})), metadata=d.get("metadata", {}),
            source=d.get("source", ""), source_type=d.get("source_type", ""), frame_hint=_untf(d.get("frame_hint")),
            exclude_from_articulation=d.get("exclude_from_articulation", False),
            loop_closure_preferred=d.get("loop_closure_preferred", False), enabled=d.get("enabled", True),
        )
