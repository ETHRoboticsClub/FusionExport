"""Dependency-free USDA (USD ASCII) document model and serializer.

Values are formatted according to the attribute's USD type name, so callers pass plain Python
data: numbers, tuples, flat or nested lists, strings, ``Quat``-like objects (``as_tuple()``
returning ``(w, x, y, z)``) and dicts for dictionary metadata.
"""

from __future__ import annotations

import math
import re
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INDENT = "    "

_INT_TYPES = {"int", "uint", "int64", "uint64", "uchar"}
_F32_TYPES = {"float", "half"}
_F64_TYPES = {"double", "timecode"}
_STRING_TYPES = {"string", "token"}

# Tuple-valued types: (dimension, component kind).
_TUPLE_TYPES: Dict[str, tuple] = {}
for _n in ("2", "3", "4"):
    _TUPLE_TYPES[f"float{_n}"] = (int(_n), "f")
    _TUPLE_TYPES[f"half{_n}"] = (int(_n), "f")
    _TUPLE_TYPES[f"double{_n}"] = (int(_n), "d")
    _TUPLE_TYPES[f"int{_n}"] = (int(_n), "i")
for _role in ("point", "normal", "vector", "color"):
    _TUPLE_TYPES[f"{_role}3f"] = (3, "f")
    _TUPLE_TYPES[f"{_role}3h"] = (3, "f")
    _TUPLE_TYPES[f"{_role}3d"] = (3, "d")
_TUPLE_TYPES.update({
    "color4f": (4, "f"), "color4d": (4, "d"), "texCoord2f": (2, "f"), "texCoord2d": (2, "d"),
    "quatf": (4, "f"), "quath": (4, "f"), "quatd": (4, "d"),
})


# ---------------------------------------------------------------- scalar formatting


def format_f32(v: float) -> str:
    """Compact float formatting (9 significant digits round-trip any float32)."""
    return "%.9g" % v


def format_f64(v: float) -> str:
    """Shortest round-trip double formatting; ``inf``/``-inf``/``nan`` as USDA expects."""
    return repr(float(v))


def quote(s: str) -> str:
    """USDA string literal with escaping (UTF-8 is written verbatim)."""
    s = (s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
         .replace("\r", "\\r").replace("\t", "\\t"))
    return f'"{s}"'


def format_path(path: str) -> str:
    return f"<{path}>"


def format_asset(path: str) -> str:
    return f"@@@{path}@@@" if "@" in path else f"@{path}@"


def _as_components(v: Any) -> Sequence[Any]:
    return v.as_tuple() if hasattr(v, "as_tuple") else tuple(v)


def _component_fmt(kind: str) -> Callable[[Any], str]:
    if kind == "i":
        return lambda c: str(int(c))
    return format_f64 if kind == "d" else format_f32


def _scalar_formatter(base: str) -> Callable[[Any], str]:
    if base == "bool":
        return lambda v: "true" if v else "false"
    if base in _INT_TYPES:
        return lambda v: str(int(v))
    if base in _F32_TYPES:
        return format_f32
    if base in _F64_TYPES:
        return format_f64
    if base in _STRING_TYPES:
        return lambda v: quote(str(v.value if isinstance(v, Enum) else v))
    if base == "asset":
        return lambda v: format_asset(str(v))
    if base in _TUPLE_TYPES:
        comp = _component_fmt(_TUPLE_TYPES[base][1])
        return lambda v: "(" + ", ".join(comp(c) for c in _as_components(v)) + ")"
    if base.startswith("matrix"):
        return lambda v: "(" + ", ".join(
            "(" + ", ".join(format_f64(c) for c in row) + ")" for row in v) + ")"
    raise ValueError(f"unsupported USD value type {base!r}")


def _flat_tuple_array(base: str, flat: Sequence[Any]) -> str:
    """Format a flat number list (x0, y0, z0, x1, ...) as a tuple array without per-item calls."""
    dim, kind = _TUPLE_TYPES[base]
    if len(flat) % dim:
        raise ValueError(f"flat {base}[] length {len(flat)} is not a multiple of {dim}")
    spec = {"f": "%.9g", "d": "%r", "i": "%d"}[kind]
    fmt = "(" + ", ".join([spec] * dim) + ")"
    if kind == "d":
        flat = [float(c) for c in flat]
    it = iter(flat)
    return "[" + ", ".join([fmt % t for t in zip(*([it] * dim))]) + "]"


def format_value(type_name: str, value: Any) -> str:
    """Format ``value`` for a USD type name such as ``float``, ``point3f[]`` or ``token``."""
    is_array = type_name.endswith("[]")
    base = type_name[:-2] if is_array else type_name
    if not is_array:
        return _scalar_formatter(base)(value)
    items = list(value)
    if not items:
        return "[]"
    if base in _TUPLE_TYPES and isinstance(items[0], (int, float)):
        return _flat_tuple_array(base, items)
    if base in _INT_TYPES:
        return "[" + ", ".join(map(str, map(int, items))) + "]"
    if base in _F32_TYPES:
        return "[" + ", ".join(["%.9g" % v for v in items]) + "]"
    fmt = _scalar_formatter(base)
    return "[" + ", ".join([fmt(v) for v in items]) + "]"


# ---------------------------------------------------------------- dictionaries / metadata


def _dict_entry_type(v: Any) -> Optional[str]:
    """USD type for a Python value inside a ``dictionary``; ``None`` means skip."""
    if v is None:
        return None
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int" if -2 ** 31 <= v < 2 ** 31 else "int64"
    if isinstance(v, float):
        return "double"
    if isinstance(v, dict):
        return "dictionary"
    if isinstance(v, (list, tuple)):
        items = [x for x in v if x is not None]
        if items and all(isinstance(x, bool) for x in items):
            return "bool[]"
        if items and all(isinstance(x, int) and not isinstance(x, bool) for x in items):
            return "int64[]" if any(not -2 ** 31 <= x < 2 ** 31 for x in items) else "int[]"
        if items and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in items):
            return "double[]"
        return "string[]"
    return "string"


def _coerce_dict_value(type_name: str, v: Any) -> Any:
    if type_name == "string":
        return v.value if isinstance(v, Enum) else str(v)
    if type_name == "string[]":
        return [str(x.value if isinstance(x, Enum) else x) for x in v if x is not None]
    if type_name.endswith("[]"):
        return [x for x in v if x is not None]
    return v


def _dict_key(key: Any) -> str:
    k = str(key)
    return k if _IDENT.match(k) else quote(k)


def format_dictionary(d: Dict[Any, Any], indent: int, out: List[str], head: str) -> None:
    """Append ``<head>{ ... }`` for a (possibly nested) dict at the given indent level."""
    pad = _INDENT * indent
    entries = [(k, v, _dict_entry_type(v)) for k, v in d.items()]
    entries = [e for e in entries if e[2] is not None]
    out.append(f"{pad}{head}{{")
    for k, v, t in entries:
        if t == "dictionary":
            format_dictionary(v, indent + 1, out, f"dictionary {_dict_key(k)} = ")
        else:
            out.append(f"{pad}{_INDENT}{t} {_dict_key(k)} = {format_value(t, _coerce_dict_value(t, v))}")
    out.append(f"{pad}}}")


def _format_metadata_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return format_f64(v) if isinstance(v, float) else str(v)
    if isinstance(v, Enum):
        return quote(str(v.value))
    if isinstance(v, str):
        return quote(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_format_metadata_value(x) for x in v) + "]"
    raise ValueError(f"unsupported metadata value {v!r}")


def format_metadata_block(meta: Dict[str, Any], indent: int, out: List[str]) -> None:
    """Append ``key = value`` lines; dict values become dictionaries (e.g. ``customData``)."""
    pad = _INDENT * indent
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, dict):
            format_dictionary(value, indent, out, f"{key} = ")
        else:
            out.append(f"{pad}{key} = {_format_metadata_value(value)}")


# ---------------------------------------------------------------- document model


class Property:
    name: str

    def write(self, indent: int, out: List[str]) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class Attribute(Property):
    """``[custom] [uniform] <type> <name> [= value]`` plus optional connection and metadata."""

    def __init__(self, name: str, type_name: str, value: Any = None, *, custom: bool = False,
                 connect: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        parts = type_name.split()
        self.uniform = parts[0] == "uniform"
        self.type_name = parts[-1]
        self.name = name
        self.value = value
        self.custom = custom
        self.connect = connect
        self.metadata: Dict[str, Any] = dict(metadata or {})

    def _decl(self) -> str:
        prefix = ("custom " if self.custom else "") + ("uniform " if self.uniform else "")
        return f"{prefix}{self.type_name} {self.name}"

    def write(self, indent: int, out: List[str]) -> None:
        pad = _INDENT * indent
        line = pad + self._decl()
        if self.value is not None:
            line += " = " + format_value(self.type_name, self.value)
        if self.metadata:
            out.append(line + " (")
            format_metadata_block(self.metadata, indent + 1, out)
            out.append(pad + ")")
        elif self.value is not None or self.connect is None:
            out.append(line)
        if self.connect is not None:
            out.append(f"{pad}{self._decl()}.connect = {format_path(self.connect)}")


class Relationship(Property):
    """``[custom] rel <name> = </target>`` (or a target list; no targets -> declaration only)."""

    def __init__(self, name: str, targets: Iterable[str] = (), *, custom: bool = False) -> None:
        self.name = name
        self.targets = list(targets)
        self.custom = custom

    def write(self, indent: int, out: List[str]) -> None:
        line = _INDENT * indent + ("custom " if self.custom else "") + f"rel {self.name}"
        if len(self.targets) == 1:
            line += " = " + format_path(self.targets[0])
        elif self.targets:
            line += " = [" + ", ".join(format_path(t) for t in self.targets) + "]"
        out.append(line)


class Prim:
    """A prim spec with metadata, properties and ordered children."""

    def __init__(self, name: str, type_name: str = "", specifier: str = "def",
                 parent: Optional["Prim"] = None) -> None:
        if not _IDENT.match(name):
            raise ValueError(f"invalid prim name {name!r}")
        if specifier not in ("def", "over", "class"):
            raise ValueError(f"invalid specifier {specifier!r}")
        self.name = name
        self.type_name = type_name
        self.specifier = specifier
        self.parent = parent
        self.api_schemas: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.custom_data: Dict[str, Any] = {}
        self.properties: Dict[str, Property] = {}
        self.children: Dict[str, Prim] = {}

    @property
    def path(self) -> str:
        return (self.parent.path if self.parent else "") + "/" + self.name

    # -- building
    def define(self, name: str, type_name: str = "", specifier: str = "def") -> "Prim":
        if name in self.children:
            raise ValueError(f"prim {self.path}/{name} already exists")
        child = Prim(name, type_name, specifier, self)
        self.children[name] = child
        return child

    def child(self, name: str) -> Optional["Prim"]:
        return self.children.get(name)

    def apply(self, *schemas: str) -> "Prim":
        for s in schemas:
            if s not in self.api_schemas:
                self.api_schemas.append(s)
        return self

    def attr(self, name: str, type_name: str, value: Any = None, **kwargs: Any) -> Attribute:
        a = Attribute(name, type_name, value, **kwargs)
        self.properties[name] = a
        return a

    def rel(self, name: str, targets: Iterable[str] = (), **kwargs: Any) -> Relationship:
        r = Relationship(name, targets, **kwargs)
        self.properties[name] = r
        return r

    def get(self, name: str) -> Optional[Property]:
        return self.properties.get(name)

    # -- serialization
    def write(self, indent: int, out: List[str]) -> None:
        pad = _INDENT * indent
        head = f"{pad}{self.specifier} " + (f"{self.type_name} " if self.type_name else "") + quote(self.name)
        meta: Dict[str, Any] = dict(self.metadata)
        if self.custom_data:
            meta["customData"] = self.custom_data
        if self.api_schemas or meta:
            out.append(head + " (")
            format_metadata_block(meta, indent + 1, out)
            if self.api_schemas:
                out.append(f"{pad}{_INDENT}prepend apiSchemas = "
                           + "[" + ", ".join(quote(s) for s in self.api_schemas) + "]")
            out.append(pad + ")")
        else:
            out.append(head)
        out.append(pad + "{")
        for p in self.properties.values():
            p.write(indent + 1, out)
        for i, c in enumerate(self.children.values()):
            if self.properties or i:
                out.append("")
            c.write(indent + 1, out)
        out.append(pad + "}")


class UsdaStage:
    """Root layer: layer metadata plus root prims."""

    def __init__(self, default_prim: Optional[str] = None, up_axis: str = "Z",
                 meters_per_unit: float = 1.0, kilograms_per_unit: float = 1.0, doc: str = "") -> None:
        self.default_prim = default_prim
        self.up_axis = up_axis
        self.meters_per_unit = meters_per_unit
        self.kilograms_per_unit = kilograms_per_unit
        self.doc = doc
        self.custom_layer_data: Dict[str, Any] = {}
        self.root_prims: Dict[str, Prim] = {}

    def define(self, name: str, type_name: str = "", specifier: str = "def") -> Prim:
        if name in self.root_prims:
            raise ValueError(f"prim /{name} already exists")
        p = Prim(name, type_name, specifier)
        self.root_prims[name] = p
        return p

    def get(self, path: str) -> Optional[Prim]:
        parts = [p for p in path.split("/") if p]
        if not parts:
            return None
        prim = self.root_prims.get(parts[0])
        for name in parts[1:]:
            if prim is None:
                return None
            prim = prim.child(name)
        return prim

    def to_string(self) -> str:
        out: List[str] = ["#usda 1.0", "("]
        meta: Dict[str, Any] = {}
        if self.custom_layer_data:
            meta["customLayerData"] = self.custom_layer_data
        if self.default_prim:
            meta["defaultPrim"] = self.default_prim
        if self.doc:
            meta["doc"] = self.doc
        meta["kilogramsPerUnit"] = _unit(self.kilograms_per_unit)
        meta["metersPerUnit"] = _unit(self.meters_per_unit)
        meta["upAxis"] = self.up_axis
        format_metadata_block(meta, 1, out)
        out.append(")")
        for p in self.root_prims.values():
            out.append("")
            p.write(0, out)
        out.append("")
        return "\n".join(out)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(self.to_string())


def _unit(v: float) -> Any:
    return int(v) if float(v).is_integer() and not math.isinf(v) else float(v)
