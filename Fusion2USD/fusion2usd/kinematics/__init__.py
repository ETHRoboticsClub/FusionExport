"""Kinematic processing between the CAD reader and the USD writer."""

from __future__ import annotations

from typing import Optional

from .decompose import JointDecompositionPass
from .frames import BaseFramePass, UpAxisPass, WorldTransformPass
from .naming import NamingPass
from .pipeline import ModelPass, PassContext, Pipeline
from .properties import OverrideSet, PropertiesPass
from .sensors import SensorPropertiesPass
from .tree import KinematicTreePass
from .validation import MassSanitizePass, ValidationPass
from .zero_pose import ZeroPosePass


def default_pipeline(overrides: Optional[OverrideSet] = None) -> Pipeline:
    """Raw CAD model -> writer-ready model. Order matters (see each pass's docstring)."""
    return Pipeline([
        PropertiesPass(overrides),  # needs raw Fusion names for override matching
        SensorPropertiesPass(),
        NamingPass(),
        UpAxisPass(),
        KinematicTreePass(),        # needs properties (loop_closure / actuated hints)
        JointDecompositionPass(),   # needs exclusion flags from the tree
        ZeroPosePass(),
        BaseFramePass(),            # needs the final base and all (incl. helper) links
        MassSanitizePass(),
        ValidationPass(),
    ])


__all__ = [
    "default_pipeline", "Pipeline", "PassContext", "ModelPass", "OverrideSet", "PropertiesPass",
    "NamingPass", "UpAxisPass", "WorldTransformPass", "KinematicTreePass", "JointDecompositionPass",
    "ZeroPosePass", "BaseFramePass", "MassSanitizePass", "ValidationPass", "SensorPropertiesPass",
]
