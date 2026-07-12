# lc_json.py
from __future__ import annotations

import json
import re
from typing import Any, Dict


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Extract the first {...} block and parse.
    Helps when a model accidentally returns extra text.
    """
    text = (text or "").strip()
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return {}
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        return {}
