"""Reads a Fusion design into the raw :class:`RobotModel` (before the kinematic passes)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import adsk.core

from ..core.math3d import Transform
from ..core.report import ExportReport
from ..model.robot import Joint, Link, Mesh, RobotModel
from ..options import ExportOptions
from .couplings import MotionLinkReader
from .geometry import MeshExtractor
from .grouping import ROOT, Group, GroupingResult, JointEdge, PartGrouper
from .joints import FusionJoint, JointBuilder, JointCollector, context_path
from .mass import MassCalculator
from .metadata import AttributeReader
from .occurrences import OccurrenceInfo, OccurrenceScanner, resolve_key
from .probe import JointProbe
from .sensors import SensorReader
from .units import items, transform_m


def _unique(name: str, used: Set[str]) -> str:
    candidate, n = name, 1
    while candidate in used:
        n += 1
        candidate = f"{name}_{n}"
    used.add(candidate)
    return candidate


class LinkBuilder:
    """Builds one :class:`Link` per occurrence group."""

    def __init__(self, design: Any, options: ExportOptions, report: ExportReport,
                 occurrences: Dict[str, OccurrenceInfo], root_bodies: List[Any]) -> None:
        self._root = design.rootComponent
        self._options = options
        self._report = report
        self._occ = occurrences
        self._root_bodies = root_bodies
        self._mass = MassCalculator(design, report)
        self.meshes = MeshExtractor(report)
        self._attributes = AttributeReader()
        self._names: Set[str] = set()

    def bodies(self, group: Group) -> List[Any]:
        out: List[Any] = []
        for key in group.members:
            out.extend(self._root_bodies if key == ROOT else self._occ[key].bodies)
        return out

    def _root_group_member(self, group: Group):
        """The root frame anchors the grounded group, but a member occurrence identifies it better than
        the root component (which carries the document name, e.g. "(Unsaved)")."""
        return next((self._occ[k].entity for k in group.members if k != ROOT), None)

    def build(self, group: Group) -> Link:
        anchor = None if group.anchor == ROOT else self._occ[group.anchor].entity
        identity = anchor or self._root_group_member(group)  # names the link; None: root bodies only
        name = _unique(identity.name if identity is not None else "base", self._names)
        transform = Transform() if group.contains_root else transform_m(anchor.transform2)
        link = Link(name=name, transform=transform,
                    source=identity.fullPathName if identity is not None else (self._root.name or "base"),
                    metadata=self._metadata(group, anchor))
        bodies = self.bodies(group)
        mass = self._mass.compute(bodies, f"link {name!r}")
        if mass is None or mass.mass <= 0.0:
            self._report.warn(f"link {name!r} has no mass (no solid bodies or no physical material)")
        else:
            link.mass = mass.transformed(transform.inverse())
        if self._options.export_visuals:
            link.visuals = self._meshes(bodies, link, visual=True)
        if self._options.export_collisions:
            link.collisions = self._meshes(bodies, link, visual=False)
        return link

    def _meshes(self, bodies: List[Any], link: Link, visual: bool) -> List[Mesh]:
        quality = self._options.visual_quality if visual else self._options.collision_quality
        used: Set[str] = set()
        out = []
        for body in bodies:
            mesh = self.meshes.mesh(body, quality, link.transform, with_material=visual)
            if mesh is not None:
                mesh.name = _unique(mesh.name or "body", used)
                out.append(mesh)
        return out

    def _metadata(self, group: Group, anchor: Any) -> Dict[str, Any]:
        component = self._root if anchor is None else anchor.component
        attributes: Dict[str, Any] = {}
        entities = [self._root if k == ROOT else self._occ[k].entity for k in group.members]
        anchor_entity = self._root if anchor is None else anchor
        for entity in sorted(entities, key=lambda e: e is anchor_entity):  # anchor last: it wins
            try:
                attributes.update(self._attributes.read(entity))
            except ValueError as exc:
                self._report.warn(f"{entity.name!r}: invalid fusion2usd attributes ({exc})")
        material = component.material
        return {
            "fusion_occurrences": [self._root.name if k == ROOT else k for k in group.members],
            "component": component.name,
            "part_number": component.partNumber,
            "description": component.description,
            "material": material.name if material is not None else None,
            "attributes": attributes,
        }


class FusionDesignReader:
    def __init__(self, design: Any, options: ExportOptions, report: ExportReport) -> None:
        self._design = design
        self._options = options
        self._report = report

    def read(self) -> RobotModel:
        root = self._design.rootComponent
        scanner = OccurrenceScanner(root, self._options.include_hidden)
        infos = scanner.scan()
        by_key = OccurrenceScanner.by_key(infos)
        root_bodies = scanner.root_bodies()
        fusion_joints = self._known_joints(JointCollector(root, self._report).collect(), by_key)

        grouping = PartGrouper(
            [i.record for i in infos],
            [JointEdge(fj.name, fj.key_one, fj.key_two, fj.is_rigid) for fj in fusion_joints],
            self._rigid_groups(root, by_key),
            root_has_bodies=bool(root_bodies),
        ).run()
        internal = set(grouping.internal_joints)
        for name in grouping.internal_joints:
            self._report.warn(f"joint {name!r} connects parts that are rigidly connected otherwise; dropped")
        movable = [fj for fj in fusion_joints if not fj.is_rigid and fj.name not in internal]

        model = RobotModel(name=self._options.robot_name or self._document_name())
        model.metadata.update({"source": "Autodesk Fusion", "document": self._document_name(),
                               "source_up_axis": self._up_axis()})
        builder = LinkBuilder(self._design, self._options, self._report, by_key, root_bodies)
        link_names = self._build_links(model, builder, grouping, movable)
        # Before probing, so sensors are read at the same pose as the links.
        link_of = self._link_resolver(grouping, link_names, by_key, model)
        model.sensors = SensorReader(root, self._report, link_of).read()
        for material in builder.meshes.appearances.materials:
            model.add_material(material)

        joint_builder = JointBuilder(self._report)
        entries = []
        for fj in movable:
            parent = link_names[grouping.group_index(fj.key_two)]
            child = link_names[grouping.group_index(fj.key_one)]
            entries.append((fj, model.add_joint(joint_builder.build(fj, parent, child))))
        if self._options.probe_joints and entries:
            JointProbe(self._report).run(entries)

        joints_by_name: Dict[str, Joint] = {j.name: j for j in model.joints}
        model.couplings = MotionLinkReader(root, [i.entity for i in infos], self._report).read(
            fusion_joints, joints_by_name)
        self._report.info(f"Read {len(model.links)} link(s), {len(model.joints)} joint(s), "
                          f"{len(model.couplings)} motion link(s), {len(model.sensors)} sensor(s) from Fusion")
        return model

    # -- steps
    def _build_links(self, model: RobotModel, builder: LinkBuilder, grouping: GroupingResult,
                     movable: List[FusionJoint]) -> Dict[int, str]:
        referenced = {grouping.group_index(k) for fj in movable for k in (fj.key_one, fj.key_two)}
        names: Dict[int, str] = {}
        for idx, group in enumerate(grouping.groups):
            if idx not in referenced and not builder.bodies(group):
                continue  # empty helper occurrences (e.g. grounded sub-assembly shells)
            link = model.add_link(builder.build(group))
            names[idx] = link.name
        g = grouping.grounded_group
        if g is not None and g in names:
            model.grounded_links = [names[g]]
            model.base_link = names[g]
        return names

    @staticmethod
    def _link_resolver(grouping: GroupingResult, names: Dict[int, str], by_key: Dict[str, OccurrenceInfo],
                       model: RobotModel):
        """Occurrence key -> link name: the nearest ancestor occurrence that is in an exported link,
        else the root group, else the base link."""
        def link_of(key: Optional[str]) -> Optional[str]:
            while key is not None:
                idx = grouping.group_index(key)
                if idx in names:
                    return names[idx]
                info = by_key.get(key)
                key = info.record.parent if info is not None else None
            return names.get(grouping.group_index(None), model.base_link)
        return link_of

    def _known_joints(self, joints: List[FusionJoint], by_key: Dict[str, OccurrenceInfo]) -> List[FusionJoint]:
        out = []
        for fj in joints:
            one = resolve_key(fj.key_one, fj.context, by_key)
            two = resolve_key(fj.key_two, fj.context, by_key)
            unknown = [k for k, r in ((fj.key_one, one), (fj.key_two, two)) if k is not None and r is None]
            if unknown:
                self._report.warn(f"joint {fj.name!r} references unknown occurrence(s) {unknown}; skipped")
                continue
            # Swap in the root-context proxies so keys, transforms and probing all live in the root frame.
            if one != fj.key_one:
                fj.occurrence_one = by_key[one].entity
            if two != fj.key_two:
                fj.occurrence_two = by_key[two].entity
            out.append(fj)
        return out

    def _rigid_groups(self, root: Any, by_key: Dict[str, OccurrenceInfo]) -> List[List[str]]:
        groups = []
        for rg in items(root.allRigidGroups):
            if rg.isSuppressed:
                continue
            context = context_path(rg)
            members = []
            for o in items(rg.occurrences):
                key = resolve_key(o.fullPathName, context, by_key)
                if key is None:
                    self._report.warn(f"rigid group {rg.name!r}: unknown occurrence {o.fullPathName!r}; ignored")
                else:
                    members.append(key)
            groups.append(members)
        return groups

    def _document_name(self) -> str:
        doc = self._design.parentDocument
        return doc.name if doc is not None else self._design.rootComponent.name

    def _up_axis(self) -> str:
        axis = (self._options.source_up_axis or "auto").upper()
        if axis in ("Y", "Z"):
            return axis
        try:
            prefs = adsk.core.Application.get().preferences.generalPreferences
            orientation = prefs.defaultModelingOrientation
        except Exception as exc:
            self._report.warn(f"cannot read the modeling orientation preference ({exc}); assuming Y up")
            return "Y"
        return "Z" if orientation == adsk.core.DefaultModelingOrientations.ZUpModelingOrientation else "Y"
