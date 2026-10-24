"""Parses raw sensor attributes into typed parameters.

Camera parameters follow Isaac Lab's ``PinholeCameraCfg`` units (focal length and apertures are written
to USD as given). ``fov`` is the horizontal field of view; a bare number means degrees.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from ..model.robot import RobotModel, Sensor, SensorKind
from .pipeline import ModelPass, PassContext
from .properties import ValueParser

DEFAULT_FOCAL_LENGTH = 24.0          # Isaac Lab PinholeCameraCfg default
DEFAULT_HORIZONTAL_APERTURE = 20.955
DEFAULT_CLIPPING = (0.01, 1.0e6)

_COMMON = ("sensor", "name", "rate", "update_period")
_KEYS = {
    SensorKind.CAMERA: _COMMON + ("width", "height", "fov", "focal_length", "horizontal_aperture", "near", "far",
                                  "data_types"),
    SensorKind.IMU: _COMMON,
    SensorKind.CONTACT: _COMMON + ("history_length", "track_air_time", "force_threshold"),
    SensorKind.FRAME: _COMMON,
}


class SensorPropertiesPass(ModelPass):
    def run(self, model: RobotModel, ctx: PassContext) -> RobotModel:
        parser = ValueParser(ctx.report)
        kept = []
        for sensor in model.sensors:
            if sensor.link not in model.links:
                ctx.report.warn(f"Sensor {sensor.name!r} is attached to unknown link {sensor.link!r}; skipped")
                continue
            raw = {k.strip().lower(): v for k, v in (sensor.metadata.pop("attributes", {}) or {}).items()}
            where = f"sensor {sensor.name!r}"
            sensor.params.update(self._common(raw, parser, where))
            if sensor.kind is SensorKind.CAMERA:
                sensor.params.update(self._camera(raw, parser, where))
            elif sensor.kind is SensorKind.CONTACT:
                sensor.params.update(self._contact(raw, parser, where))
            if str(raw.get("name", "")).strip():
                sensor.metadata["export_name"] = str(raw["name"]).strip()
            user = {k: ValueParser.loose(v) for k, v in raw.items() if k not in _KEYS[sensor.kind]}
            if user:
                sensor.metadata.setdefault("user", {}).update(user)
            kept.append(sensor)
        model.sensors = kept
        return model

    @staticmethod
    def _common(raw: Dict[str, Any], parser: ValueParser, where: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if "update_period" in raw:
            out["update_period"] = parser.number(raw["update_period"], where)
        elif "rate" in raw:
            rate = parser.number(str(raw["rate"]).lower().replace("hz", ""), where)
            if rate:
                out["update_period"] = 1.0 / rate
        return {k: v for k, v in out.items() if v is not None}

    @staticmethod
    def _camera(raw: Dict[str, Any], parser: ValueParser, where: str) -> Dict[str, Any]:
        width = int(parser.number(raw.get("width", 640), where) or 640)
        height = int(parser.number(raw.get("height", 480), where) or 480)
        focal = parser.number(raw.get("focal_length", DEFAULT_FOCAL_LENGTH), where) or DEFAULT_FOCAL_LENGTH
        if "fov" in raw:
            fov_raw = str(raw["fov"]).strip()
            fov = parser.number(fov_raw if not fov_raw.replace(".", "", 1).isdigit() else fov_raw + " deg", where)
            if fov is None or not 0.0 < fov < math.pi:
                parser.report.warn(f"{where}: field of view {raw['fov']!r} out of range; using defaults")
                h_aperture = DEFAULT_HORIZONTAL_APERTURE
            else:
                h_aperture = 2.0 * focal * math.tan(fov / 2.0)
        else:
            h_aperture = parser.number(raw.get("horizontal_aperture", DEFAULT_HORIZONTAL_APERTURE), where) \
                or DEFAULT_HORIZONTAL_APERTURE
        near = parser.number(raw.get("near", DEFAULT_CLIPPING[0]), where) or DEFAULT_CLIPPING[0]
        far = parser.number(raw.get("far", DEFAULT_CLIPPING[1]), where) or DEFAULT_CLIPPING[1]
        data_types = [t.strip() for t in str(raw.get("data_types", "rgb")).split(",") if t.strip()]
        return {
            "width": width, "height": height, "focal_length": focal,
            "horizontal_aperture": h_aperture, "vertical_aperture": h_aperture * height / width,
            "clipping_range": (near, far), "data_types": data_types,
        }

    @staticmethod
    def _contact(raw: Dict[str, Any], parser: ValueParser, where: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if "history_length" in raw:
            value = parser.number(raw["history_length"], where)
            if value is not None:
                out["history_length"] = int(value)
        if "track_air_time" in raw:
            out["track_air_time"] = bool(parser.boolean(raw["track_air_time"], where))
        if "force_threshold" in raw:
            out["force_threshold"] = parser.number(raw["force_threshold"], where)
        return {k: v for k, v in out.items() if v is not None}


def camera_horizontal_fov(sensor: Sensor) -> float:
    p = sensor.params
    return 2.0 * math.atan(p["horizontal_aperture"] / (2.0 * p["focal_length"]))
