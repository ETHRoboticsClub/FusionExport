"""Post-export actions: extra outputs derived from a finished export (Isaac Lab cfg, URDF, ...).

Write your own by subclassing ``PostExportAction`` and either registering it::

    @register_action("my_action")
    class MyAction(PostExportAction):
        def run(self, artifacts: ExportArtifacts) -> List[str]:
            ...

or referencing it as ``"my_package.my_module:MyAction"`` in ``ExportOptions.post_actions``.
"""

from __future__ import annotations

import importlib
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Type

from ..core.report import ExportReport
from ..model.robot import RobotModel
from ..options import ExportOptions


@dataclass
class ExportArtifacts:
    """Everything an action may need. ``model`` is the processed model the USD was written from."""

    model: RobotModel
    options: ExportOptions
    report: ExportReport
    usd_path: str
    written: List[str] = field(default_factory=list)

    @property
    def output_dir(self) -> str:
        return os.path.dirname(os.path.abspath(self.usd_path))

    @property
    def stem(self) -> str:
        return os.path.splitext(os.path.basename(self.usd_path))[0]

    def sibling(self, suffix: str) -> str:
        """Path next to the USD file, e.g. ``sibling("_cfg.py")``."""
        return os.path.join(self.output_dir, self.stem + suffix)


class PostExportAction(ABC):
    name: str = ""

    @abstractmethod
    def run(self, artifacts: ExportArtifacts) -> List[str]:
        """Produce outputs; return the paths written."""


_REGISTRY: Dict[str, Type[PostExportAction]] = {}


def register_action(name: str) -> Callable[[Type[PostExportAction]], Type[PostExportAction]]:
    def decorator(cls: Type[PostExportAction]) -> Type[PostExportAction]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return decorator


def registered_actions() -> List[str]:
    return sorted(_REGISTRY)


def create_action(spec: str) -> PostExportAction:
    """Instantiate a registered action by name, or a class given as ``module:Class``."""
    if spec in _REGISTRY:
        return _REGISTRY[spec]()
    if ":" in spec:
        module_name, cls_name = spec.split(":", 1)
        cls = getattr(importlib.import_module(module_name), cls_name)
        if not (isinstance(cls, type) and issubclass(cls, PostExportAction)):
            raise TypeError(f"{spec} is not a PostExportAction subclass")
        return cls()
    raise KeyError(f"unknown post-export action {spec!r} (known: {', '.join(registered_actions())})")
