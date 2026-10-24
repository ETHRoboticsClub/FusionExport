"""Model passes: each one transforms a RobotModel towards what the USD writer expects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Sequence

from ..core.report import ExportReport
from ..model.robot import RobotModel
from ..options import ExportOptions


@dataclass
class PassContext:
    options: ExportOptions
    report: ExportReport = field(default_factory=ExportReport)


class ModelPass(ABC):
    """A single, self-contained transformation of the model (mutates and returns it)."""

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def run(self, model: RobotModel, ctx: PassContext) -> RobotModel:
        ...


class Pipeline:
    def __init__(self, passes: Sequence[ModelPass]) -> None:
        self.passes: List[ModelPass] = list(passes)

    def run(self, model: RobotModel, ctx: PassContext) -> RobotModel:
        for p in self.passes:
            model = p.run(model, ctx)
        return model
