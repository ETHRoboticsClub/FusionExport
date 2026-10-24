"""User attributes in the ``fusion2usd`` group, stored on Fusion entities."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional

from .units import items

GROUP = "fusion2usd"
JSON_KEY = "json"  # free-form extra metadata, stored as a JSON object string


def native(entity: Any) -> Any:
    """The native object of a proxy (attributes are persisted there), else the entity itself."""
    obj = getattr(entity, "nativeObject", None)
    return obj if obj is not None else entity


class AttributeReader:
    def __init__(self, group: str = GROUP) -> None:
        self.group = group

    def read_raw(self, entity: Any) -> Dict[str, str]:
        """Raw ``name -> value`` strings; falls back to the native object for proxies."""
        if entity is None:
            return {}
        values = self._read(entity)
        if not values:
            obj = native(entity)
            if obj is not entity:
                values = self._read(obj)
        return values

    def read(self, entity: Any) -> Dict[str, Any]:
        """Attributes with the ``json`` attribute parsed and merged (explicit attributes win)."""
        raw = self.read_raw(entity)
        out: Dict[str, Any] = {}
        text = raw.pop(JSON_KEY, None)
        if text:
            extra = json.loads(text)
            if not isinstance(extra, dict):
                raise ValueError(f"attribute {self.group}.{JSON_KEY} must hold a JSON object")
            out.update(extra)
        out.update(raw)
        return out

    def _read(self, entity: Any) -> Dict[str, str]:
        attrs = getattr(entity, "attributes", None)
        if attrs is None:
            return {}
        return {a.name: a.value for a in items(attrs.itemsByGroup(self.group))}


class AttributeWriter:
    def __init__(self, group: str = GROUP) -> None:
        self.group = group

    def write(self, entity: Any, values: Mapping[str, Optional[str]]) -> None:
        """Set each attribute on the native object; ``None`` or ``""`` deletes it."""
        attrs = native(entity).attributes
        for name, value in values.items():
            if value is None or value == "":
                existing = attrs.itemByName(self.group, name)
                if existing is not None:
                    existing.deleteMe()
            else:
                attrs.add(self.group, name, value)

    @staticmethod
    def encode_json(extra: Mapping[str, Any]) -> str:
        return json.dumps(dict(extra), sort_keys=True) if extra else ""
