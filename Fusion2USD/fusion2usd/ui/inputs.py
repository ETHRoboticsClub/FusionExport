"""Small helpers for building and reading Fusion command inputs."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import adsk.core

Choice = Tuple[str, str]  # (label shown in the dialog, value)


def add_dropdown(inputs: Any, input_id: str, name: str, choices: Sequence[Choice], current: str) -> Any:
    dd = inputs.addDropDownCommandInput(input_id, name, adsk.core.DropDownStyles.TextListDropDownStyle)
    for label, value in choices:
        dd.listItems.add(label, value == current, "")
    return dd


def dropdown_value(inputs: Any, input_id: str, choices: Sequence[Choice]) -> str:
    item = inputs.itemById(input_id).selectedItem
    label = item.name if item is not None else choices[0][0]
    return next(v for lbl, v in choices if lbl == label)


def string_value(inputs: Any, input_id: str) -> str:
    return inputs.itemById(input_id).value.strip()


def bool_value(inputs: Any, input_id: str) -> bool:
    return bool(inputs.itemById(input_id).value)


def optional_float(text: str) -> Optional[float]:
    """Empty -> None; otherwise a float (raises ValueError on junk)."""
    text = text.strip()
    return float(text) if text else None
