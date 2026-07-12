# canonicalize.py
"""
Entity canonicalization (rule-based, no seeds, no ML)

Goal:
- Fix "section label" entities like ALLCAPS tokens (NUMBA) when a better mixed-case variant
  exists in the post text (Numba).
- Also optionally fix casing variants like "Pycuda" -> "PyCUDA" if the post contains "PyCUDA".

This is fully automated because:
- It uses only the current post text (local evidence)
- No external gazetteer/seed list required
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_+\-]{1,}\b")


def _tokenize_words(text: str) -> List[str]:
    return WORD_RE.findall(text)


def build_case_map_from_post(post_text: str) -> Dict[str, str]:
    """
    Build a mapping from lowercase form -> best surface form found in the post.

    "Best" is chosen by:
    1) prefer mixed-case (contains both lower and upper)
    2) then prefer TitleCase (first upper, rest lower)
    3) then prefer ALLCAPS
    4) tie-breaker: longer token

    Example:
    post contains: PyCUDA, Pycuda
    -> key 'pycuda' maps to 'PyCUDA'
    """
    candidates: Dict[str, List[str]] = {}
    for w in _tokenize_words(post_text):
        k = w.lower()
        candidates.setdefault(k, []).append(w)

    def score(v: str) -> Tuple[int, int]:
        has_upper = any(c.isupper() for c in v)
        has_lower = any(c.islower() for c in v)
        mixed = 1 if (has_upper and has_lower) else 0
        title = 1 if (v[0].isupper() and v[1:].islower()) else 0
        allcaps = 1 if v.isupper() else 0

        # Priority: mixed > title > allcaps
        tier = 3 if mixed else (2 if title else (1 if allcaps else 0))
        return (tier, len(v))

    case_map: Dict[str, str] = {}
    for k, vars_ in candidates.items():
        best = max(vars_, key=score)
        case_map[k] = best

    return case_map


def canonicalize_entities(entities: List[str], post_text: str) -> List[str]:
    """
    Canonicalize entity list using evidence from the post.
    Special handling:
    - If entity is ALLCAPS and post contains a mixed/title-case variant for same letters, use that variant.
    - Also unify general casing using the best surface form from post.

    Returns a new list, order preserved (first occurrence wins), duplicates removed.
    """
    case_map = build_case_map_from_post(post_text)

    out: List[str] = []
    seen = set()

    for e in entities:
        k = e.lower()
        best = case_map.get(k, e)

        # If original is ALLCAPS and best is "better", replace it
        if e.isupper() and best != e:
            e2 = best
        else:
            e2 = best

        k2 = e2.lower()
        if k2 not in seen:
            seen.add(k2)
            out.append(e2)

    return out


if __name__ == "__main__":
    post = """
    NUMBA/NumbaPro:
    Numba supports compilation...
    As @Wang has mentioned, Pycuda is faster than Numba.
    PyCUDA:
    PyCUDA is a Python programming environment for CUDA.
    """
    entities = ["NUMBA", "NumbaPro", "Pycuda", "PyCUDA", "CUDA"]

    print("Before:", entities)
    print("After :", canonicalize_entities(entities, post))
