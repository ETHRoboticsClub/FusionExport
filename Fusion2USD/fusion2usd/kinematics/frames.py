"""World-frame conversion (Fusion designs may be Y-up; Isaac Lab is Z-up)."""

from __future__ import annotations

import math

from ..core.math3d import Quat, Transform, Vec3
from ..model.robot import RobotModel
from .pipeline import ModelPass, PassContext

Y_UP_TO_Z_UP = Transform(Quat.from_axis_angle(Vec3(1.0, 0.0, 0.0), math.pi / 2.0))


class WorldTransformPass(ModelPass):
    """Re-expresses every world-space quantity as ``world_new = transform * world_old``."""

    def __init__(self, transform: Transform) -> None:
        self.transform = transform

    def run(self, model: RobotModel, ctx: PassContext) -> RobotModel:
        t = self.transform
        for link in model.links.values():
            link.transform = t * link.transform  # meshes and mass live in the link frame: untouched
        for joint in model.joints:
            joint.origin = t.apply_point(joint.origin)
            for dof in joint.dofs:
                dof.axis = t.apply_vector(dof.axis)
            if joint.frame_hint is not None:
                joint.frame_hint = t * joint.frame_hint
        for sensor in model.sensors:
            sensor.transform = t * sensor.transform
        return model


class UpAxisPass(WorldTransformPass):
    """Rotates a Y-up design into Isaac Lab's Z-up convention."""

    def __init__(self) -> None:
        super().__init__(Transform())

    def run(self, model: RobotModel, ctx: PassContext) -> RobotModel:
        up = ctx.options.source_up_axis
        if up == "auto":
            up = model.metadata.get("source_up_axis", "Z")
        up = str(up).upper()
        model.metadata["source_up_axis"] = up
        ctx.report.info(f"Source up axis: {up}" + (" (converted to Z-up)" if up == "Y" else ""))
        if up != "Y":
            return model
        self.transform = Y_UP_TO_Z_UP
        return super().run(model, ctx)


class BaseFramePass(WorldTransformPass):
    """Moves the asset origin onto the base link (Isaac Lab / URDF-importer convention).

    The base link frame is made axis-aligned (geometry and mass are re-expressed), then the whole
    model is translated so the base sits at the origin. ``model.root_pose`` keeps the original
    placement for ``init_state`` in the Isaac Lab cfg.
    """

    def __init__(self) -> None:
        super().__init__(Transform())

    def run(self, model: RobotModel, ctx: PassContext) -> RobotModel:
        base = model.links[model.base_link]
        upright = Transform(translation=base.transform.translation)
        to_new = upright.inverse() * base.transform  # old base-frame coords -> new base-frame coords
        base.visuals = [m.transformed(to_new) for m in base.visuals]
        base.collisions = [m.transformed(to_new) for m in base.collisions]
        if base.mass is not None:
            base.mass = base.mass.transformed(to_new)
        # Sensor/IMU offsets defined in the CAD component frame need this rotation (w, x, y, z).
        base.metadata["cad_frame_rotation"] = list(base.transform.rotation.as_tuple())
        base.transform = upright

        model.root_pose = upright
        self.transform = upright.inverse()
        return super().run(model, ctx)
