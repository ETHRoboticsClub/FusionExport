"""Engine-specific USD schemas. Everything else the builder writes is plain UsdPhysics.

Each engine gets exactly the schemas it reads, nothing for the other one.
"""

from __future__ import annotations

from typing import Optional

from ..model.robot import Coupling
from ..options import CollisionApproximation, PhysicsEngine
from .usda import Prim


class EngineSchemas:
    #: revolute joints in a mimic coupling must have finite limits
    mimic_needs_finite_limits = False
    #: free D6 axes only become DOFs when they carry a PhysicsLimitAPI (also an infinite one)
    d6_free_axes_need_limits = False
    #: D6 rotation limits are exported
    d6_rotation_limits = True

    def articulation_root(self, prim: Prim, self_collisions: bool) -> None:
        raise NotImplementedError

    def joint_dynamics(self, prim: Prim, armature: Optional[float], friction: Optional[float],
                       velocity: Optional[float]) -> None:
        """``velocity`` is in USD units (deg/s or m/s)."""
        raise NotImplementedError

    def mimic(self, prim: Prim, leader_path: str, c: Coupling, f_scale: float, l_scale: float) -> None:
        """``q_f = ratio q_l + offset`` in SI; ``*_scale`` convert follower / leader values to USD units."""
        raise NotImplementedError

    def collision(self, mesh: Prim, approx: str) -> None:
        mesh.apply("PhysicsCollisionAPI", "PhysicsMeshCollisionAPI")
        mesh.attr("physics:approximation", "uniform token", approx)

    def contact_sensor(self, link: Prim, threshold: float) -> None:
        raise NotImplementedError


class PhysxSchemas(EngineSchemas):
    # Isaac Sim 6.2 disables mimic joints on revolute joints without finite limits.
    mimic_needs_finite_limits = True
    # Unverified which Fusion ranges PhysX D6 twist limits accept; its D6 joints are loop closures only.
    d6_rotation_limits = False

    def articulation_root(self, prim: Prim, self_collisions: bool) -> None:
        prim.apply("PhysxArticulationAPI")
        prim.attr("physxArticulation:enabledSelfCollisions", "bool", self_collisions)

    def joint_dynamics(self, prim, armature, friction, velocity) -> None:
        values = {"physxJoint:maxJointVelocity": velocity, "physxJoint:armature": armature,
                  "physxJoint:jointFriction": friction}
        values = {k: v for k, v in values.items() if v is not None}
        if values:
            prim.apply("PhysxJointAPI")
            for name, value in values.items():
                prim.attr(name, "float", value)

    def mimic(self, prim, leader_path, c, f_scale, l_scale) -> None:
        # PhysX (Isaac Sim 6.2) only parses rot* instance/axis tokens; for a prismatic joint the token still
        # selects its single DOF and the values stay in meters (verified in simulation).
        inst = "rotX"
        ns = f"physxMimicJoint:{inst}"
        prim.apply(f"PhysxMimicJointAPI:{inst}")
        prim.rel(f"{ns}:referenceJoint", [leader_path])
        prim.attr(f"{ns}:referenceJointAxis", "uniform token", inst)
        # q_f' + gearing q_l' + offset = 0  in USD units (deg / m)
        prim.attr(f"{ns}:gearing", "float", -c.ratio * f_scale / l_scale)
        prim.attr(f"{ns}:offset", "float", -c.offset * f_scale)

    def collision(self, mesh: Prim, approx: str) -> None:
        super().collision(mesh, approx)
        if approx == CollisionApproximation.SDF.value:
            mesh.apply("PhysxSDFMeshCollisionAPI")

    def contact_sensor(self, link: Prim, threshold: float) -> None:
        link.apply("PhysxContactReportAPI")
        link.attr("physxContactReport:threshold", "float", threshold)


class NewtonSchemas(EngineSchemas):
    d6_free_axes_need_limits = True  # Newton 1.7 builds D6 DOFs from the authored limits only
    def articulation_root(self, prim: Prim, self_collisions: bool) -> None:
        # Newton enables self-collisions unless told otherwise.
        prim.apply("NewtonArticulationRootAPI")
        prim.attr("newton:selfCollisionEnabled", "bool", self_collisions)

    def joint_dynamics(self, prim, armature, friction, velocity) -> None:
        values = {"newton:velocityLimit": velocity, "newton:armature": armature, "newton:friction": friction}
        values = {k: v for k, v in values.items() if v is not None}
        if values:
            prim.apply("NewtonJointAPI")
            for name, value in values.items():
                prim.attr(name, "float", value)

    def mimic(self, prim, leader_path, c, f_scale, l_scale) -> None:
        # q_f = coef1 q_l + coef0, coef0 in follower USD units. NVIDIA's schema doc puts coef1 in USD units
        # too, but Newton's importer (1.7) applies it unconverted in joint coordinates (rad / m).
        prim.apply("NewtonMimicAPI")
        prim.rel("newton:mimicJoint", [leader_path])
        prim.attr("newton:mimicCoef0", "float", c.offset * f_scale)
        prim.attr("newton:mimicCoef1", "float", c.ratio)

    def collision(self, mesh: Prim, approx: str) -> None:
        if approx != CollisionApproximation.SDF.value:
            return super().collision(mesh, approx)
        # Newton ignores physics:approximation on SDF colliders, and the two collision APIs are exclusive.
        mesh.apply("PhysicsCollisionAPI", "NewtonSDFCollisionAPI")

    def contact_sensor(self, link: Prim, threshold: float) -> None:
        pass  # Newton has no contact-report schema; contact sensors are created by the simulation setup


def engine_schemas(engine: PhysicsEngine) -> EngineSchemas:
    return {PhysicsEngine.PHYSX: PhysxSchemas, PhysicsEngine.NEWTON: NewtonSchemas}[engine]()
