# text_utils.py
from __future__ import annotations

import re
from typing import List

from html_sentence_splitter import split_sentences_from_html, strip_html


def preprocess_text(text: str) -> str:
    """
    HTML-first lightweight cleaning used by the pipeline:
    - strip HTML tags/content wrappers
    - preserve structural newlines from blocks/lists
    - minimal whitespace normalization (inside strip_html)
    """
    if not text:
        return ""
    return strip_html(text)


def _is_noise_fragment(sentence: str) -> bool:
    """
    Very conservative filter for obvious splitter residue.
    Keeps normal short sentences, but drops fragments such as a lone trailing "s".
    """
    s = (sentence or "").strip()
    if not s:
        return True

    alnum = re.sub(r"[^A-Za-z0-9]+", "", s)
    if len(alnum) <= 1:
        return True

    return False


def split_into_sentences(text: str) -> List[str]:
    """
    Pipeline sentence splitter:
    - uses HTML stripping + structural chunking + NLTK sentence tokenization
    - keeps list/bullet lines as separate units when punctuation is missing
    """
    return [s for s in split_sentences_from_html(text) if not _is_noise_fragment(s)]


# ---------------------------------------------------------------------
# Built-in demo
# ---------------------------------------------------------------------
DEMO_POST = """
<p>Celery is a fantastic solution to this problem.</p>
<ul>
  <li>Works with Redis and RabbitMQ</li>
  <li>Easy to integrate</li>
</ul>
"""


def demo() -> None:
    print("=" * 70)
    print("TEXT_UTILS DEMO: Sentence Splitting")
    print("=" * 70)

    print("\n[1] RAW INPUT")
    print("-" * 70)
    print(DEMO_POST.strip())

    cleaned = preprocess_text(DEMO_POST)
    print("\n[2] PREPROCESSED TEXT")
    print("-" * 70)
    print(cleaned)

    sents = split_into_sentences(DEMO_POST)
    print("\n[3] SPLIT SENTENCES (TURNS)")
    print("-" * 70)
    if not sents:
        print("(No sentences found.)")
        return

    for i, s in enumerate(sents, start=1):
        print(f"{i:02d}. {s}")

    print("\n[4] SUMMARY")
    print("-" * 70)
    print(f"Total sentences: {len(sents)}")
    print("=" * 70)


if __name__ == "__main__":
    demo()
