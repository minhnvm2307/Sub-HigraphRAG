from __future__ import annotations

import html
import json
import re
from typing import Any


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    if not markers:
        return [content]
    parts = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [part.strip() for part in parts if part.strip()]


def clean_str(value: Any) -> str:
    if not isinstance(value, str):
        return value
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", html.unescape(value.strip()))


def is_float_regex(value: str) -> bool:
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def extract_first_complete_json(text: str):
    stack = []
    start = None
    for idx, char in enumerate(text):
        if char == "{":
            stack.append(idx)
            if start is None:
                start = idx
        elif char == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    try:
                        return json.loads(text[start : idx + 1])
                    except json.JSONDecodeError:
                        start = None
    return None


def list_of_list_to_csv(rows: list[list]) -> str:
    out = []
    for row in rows:
        out.append(",\t".join(f'"{str(item).strip().strip(chr(34))}"' for item in row))
    return "\n".join(out)
