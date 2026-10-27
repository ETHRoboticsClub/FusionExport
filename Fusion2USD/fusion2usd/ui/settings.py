"""Persists the last-used export options in the user's config directory."""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

from ..options import ExportOptions


def config_dir() -> str:
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "fusion2usd")


class OptionsStore:
    # Every export starts from the default mesh settings.
    NOT_REMEMBERED = ("visual_quality", "collision_quality", "max_triangles")

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or os.path.join(config_dir(), "last_export_options.json")

    def load(self) -> ExportOptions:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key in self.NOT_REMEMBERED:
                data.pop(key, None)
            return ExportOptions.from_dict(data)
        except (OSError, ValueError, TypeError):
            return ExportOptions()

    def save(self, options: ExportOptions) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(options.to_json())
