"""Top-level orchestration: source -> kinematic pipeline -> USD -> post-export actions."""

from __future__ import annotations

import os
import traceback
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence

from .actions import ExportArtifacts, PostExportAction, create_action
from .core.report import ExportReport
from .kinematics import OverrideSet, PassContext, Pipeline, default_pipeline
from .model.robot import RobotModel
from .options import ExportOptions
from .serialization import ModelSerializer


class ModelSource(ABC):
    """Where the raw (unprocessed) model comes from."""

    @abstractmethod
    def read(self, options: ExportOptions, report: ExportReport) -> RobotModel:
        ...

    def default_name(self) -> str:
        return "robot"


class FusionDesignSource(ModelSource):
    def __init__(self, design: Any) -> None:
        self.design = design

    def read(self, options: ExportOptions, report: ExportReport) -> RobotModel:
        from .fusion.reader import FusionDesignReader  # needs adsk; imported lazily

        return FusionDesignReader(self.design, options, report).read()

    def default_name(self) -> str:
        try:
            return self.design.parentDocument.name
        except Exception:  # noqa: BLE001 - adsk raises plain RuntimeError for unsaved docs
            return "robot"


class ModelFileSource(ModelSource):
    """A raw model previously dumped with ``write_model_json`` (no Fusion needed)."""

    def __init__(self, path: str) -> None:
        self.path = path

    def read(self, options: ExportOptions, report: ExportReport) -> RobotModel:
        return ModelSerializer().load(self.path)

    def default_name(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0].replace(".model", "")


class Exporter:
    def __init__(self, options: ExportOptions, pipeline: Optional[Pipeline] = None,
                 actions: Optional[Sequence[PostExportAction]] = None) -> None:
        self.options = options
        self._pipeline = pipeline
        self._actions = actions

    def export_design(self, design: Any) -> ExportReport:
        return self.export(FusionDesignSource(design))

    def export(self, source: ModelSource) -> ExportReport:
        from .usd.writer import UsdWriter

        opts = self.options
        if not opts.output_path:
            raise ValueError("no output path given")
        report = ExportReport()
        out_dir = os.path.dirname(os.path.abspath(opts.output_path))
        os.makedirs(out_dir, exist_ok=True)

        model = source.read(opts, report)
        if opts.robot_name:
            model.name = opts.robot_name
        elif not model.name:
            model.name = source.default_name()
        if opts.write_model_json:
            dump = os.path.splitext(opts.output_path)[0] + ".model.json"
            ModelSerializer().save(model, dump)
            report.outputs.append(dump)

        pipeline = self._pipeline or default_pipeline(OverrideSet.load(opts.overrides_path))
        model = pipeline.run(model, PassContext(opts, report))

        written = UsdWriter(opts, report).write(model, opts.output_path)
        report.outputs.extend(written)
        artifacts = ExportArtifacts(model, opts, report, usd_path=written[0], written=list(written))
        self._run_actions(artifacts)
        report.info(
            f"{len(model.links)} links, {len(model.joints)} joints"
            f" ({sum(j.exclude_from_articulation for j in model.joints)} loop-closing),"
            f" {len(model.couplings)} mimic couplings; base: {model.base_link}"
            f" ({'fixed' if model.fixed_base else 'floating'})"
        )
        return report

    def _run_actions(self, artifacts: ExportArtifacts) -> None:
        actions: List[PostExportAction] = list(self._actions) if self._actions is not None else []
        if self._actions is None:
            for spec in self.options.action_names():
                try:
                    actions.append(create_action(spec))
                except Exception as exc:  # noqa: BLE001 - a bad plugin must not lose the USD
                    artifacts.report.error(f"Post-export action {spec!r} could not be loaded: {exc}")
        for action in actions:
            try:
                paths = action.run(artifacts)
                artifacts.written.extend(paths)
                artifacts.report.outputs.extend(paths)
            except Exception:  # noqa: BLE001
                artifacts.report.error(f"Post-export action {action.name or type(action).__name__!r} failed:\n"
                                      f"{traceback.format_exc()}")
