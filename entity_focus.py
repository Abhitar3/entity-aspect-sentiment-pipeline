# entity_focus.py
"""
Step 4: Entity-focused sentence generation

Goal:
- Input: a sentence that may contain MULTIPLE entities wrapped in <E>...</E>
- Output: a list of sentences, each containing EXACTLY ONE <E>...</E> tag
  (all other entities remain as plain text, preserving context)

Why we do this:
- Aspect classification is entity-specific.
- A sentence can mention multiple entities; we must generate one "view" per entity.

Example:
Input:
  "<E>PyCUDA</E> gives you access to Nvidia's <E>CUDA</E> API."
Output:
  1) "<E>PyCUDA</E> gives you access to Nvidia's CUDA API."
  2) "PyCUDA gives you access to Nvidia's <E>CUDA</E> API."
"""

from __future__ import annotations

import re
from typing import List, Tuple


E_TAG_RE = re.compile(r"<E>(.*?)</E>", re.DOTALL)


def extract_tagged_entities(tagged_sentence: str) -> List[str]:
    """
    Extract entity surface forms inside <E>...</E> in left-to-right order.
    Duplicates are kept if the same entity is tagged multiple times.
    """
    return E_TAG_RE.findall(tagged_sentence)


def strip_E_tags(tagged_sentence: str) -> str:
    """Remove <E> and </E> tags but keep the text inside."""
    return re.sub(r"</?E>", "", tagged_sentence)


def generate_entity_focused_sentences(tagged_sentence: str) -> List[str]:
    """
    Main function:
    - Find all entities currently tagged
    - Remove all tags -> clean_sentence
    - For each entity occurrence (in order), re-insert <E>...</E> on the first matching occurrence
      that corresponds to that entity's position.

Important detail:
- We do NOT want to accidentally tag the wrong occurrence if the same entity text appears multiple times.
- To handle this, we build focused sentences by:
  1) locating spans of tagged entities in the original tagged sentence
  2) mapping those spans onto the clean sentence via incremental reconstruction
"""
    entities = extract_tagged_entities(tagged_sentence)
    if not entities:
        return []

    # Build a "clean" sentence and also record the clean-span positions for each entity tag occurrence.
    # We do this by scanning the tagged sentence and reconstructing the clean sentence piece by piece.
    clean_parts: List[str] = []
    clean_spans: List[Tuple[int, int, str]] = []  # (start, end, entity_text)

    i = 0
    clean_idx = 0
    while i < len(tagged_sentence):
        m = E_TAG_RE.search(tagged_sentence, i)
        if not m:
            # append remaining tail
            tail = tagged_sentence[i:]
            clean_parts.append(tail)
            clean_idx += len(tail)
            break

        # append text before <E>
        before = tagged_sentence[i:m.start()]
        clean_parts.append(before)
        clean_idx += len(before)

        ent_text = m.group(1)
        start = clean_idx
        clean_parts.append(ent_text)
        clean_idx += len(ent_text)
        end = clean_idx

        clean_spans.append((start, end, ent_text))

        i = m.end()

    clean_sentence = "".join(clean_parts)

    # Now generate one focused sentence per tagged occurrence
    focused: List[str] = []
    for span_start, span_end, ent_text in clean_spans:
        focused_sentence = (
            clean_sentence[:span_start] +
            f"<E>{clean_sentence[span_start:span_end]}</E>" +
            clean_sentence[span_end:]
        )
        focused.append(focused_sentence)

    return focused


def extract_single_entity(focused_sentence: str) -> str:
    """
    Utility: given a focused sentence that should contain exactly one <E>...</E>,
    return that entity text (or "" if not found).
    """
    m = E_TAG_RE.search(focused_sentence)
    return m.group(1) if m else ""


# -----------------------------
# Self-test
# -----------------------------

if __name__ == "__main__":
    s1 = "<E>PyCUDA</E> gives you access to Nvidia's <E>CUDA</E> parallel computation API from Python."
    s2 = "As @Wang has mentioned, <E>Pycuda</E> is faster than <E>Numba</E>."

    for s in [s1, s2]:
        print("\nINPUT TAGGED:")
        print(s)
        outs = generate_entity_focused_sentences(s)
        print("\nFOCUSED OUTPUTS:")
        for out in outs:
            print("-", out, "| entity =", extract_single_entity(out))
