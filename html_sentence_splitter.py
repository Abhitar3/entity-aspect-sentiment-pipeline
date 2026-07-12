from __future__ import annotations

import argparse
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import List

import nltk
from nltk.tokenize import sent_tokenize


# Block-level tags where we want a hard boundary in plain text.
BLOCK_TAGS = {
    "p",
    "div",
    "li",
    "ul",
    "ol",
    "blockquote",
    "section",
    "article",
    "header",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
}


class HTMLToTextExtractor(HTMLParser):
    """
    Lightweight HTML-to-text extractor:
    - Drops script/style content
    - Preserves link text
    - Adds line breaks for block elements to improve sentence segmentation
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._ignore_depth = 0
        self._ignore_tag: str | None = None

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        t = tag.lower()
        if t in {"script", "style"}:
            self._ignore_depth += 1
            self._ignore_tag = t
            return
        if t == "br" or t in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        t = tag.lower()
        if self._ignore_tag == t and self._ignore_depth > 0:
            self._ignore_depth -= 1
            if self._ignore_depth == 0:
                self._ignore_tag = None
            return
        if t in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._ignore_depth > 0:
            return
        self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        # Keep this minimal: whitespace-only normalization after HTML stripping.
        raw = raw.replace("\r", "\n").replace("\t", " ")
        raw = re.sub(r"[ \u00a0]+", " ", raw)
        raw = re.sub(r"\n+", "\n", raw)
        return raw.strip()


def strip_html(raw_html: str) -> str:
    parser = HTMLToTextExtractor()
    parser.feed(raw_html or "")
    parser.close()
    return parser.text()


def ensure_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def split_sentences_from_html(raw_html: str) -> List[str]:
    ensure_punkt()
    plain = strip_html(raw_html)
    if not plain:
        return []

    # Stage A: structural chunking by preserved line breaks from HTML blocks/bullets.
    # This prevents unrelated bullet lines from being merged when punctuation is missing.
    chunks: List[str] = []
    for line in plain.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        chunks.append(line)

    # Stage B: sentence tokenization inside each chunk.
    # If a chunk lacks terminal punctuation (common in bullets), keep it as one unit.
    out: List[str] = []
    for chunk in chunks:
        tokenized = [s.strip() for s in sent_tokenize(chunk) if s and s.strip()]
        if tokenized:
            out.extend(tokenized)
        else:
            out.append(chunk)

    return out


def load_input(input_path: str, json_key: str) -> str:
    path = Path(input_path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        value = data.get(json_key, "")
        if not isinstance(value, str):
            raise ValueError(f"JSON key '{json_key}' must be a string.")
        return value
    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone HTML stripping + sentence splitter (not connected to pipeline)."
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Raw post text/HTML.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to input file (.txt/.html or .json).",
    )
    parser.add_argument(
        "--json-key",
        default="post",
        help="When --input is .json, read this key (default: post).",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Print sentence list as JSON instead of numbered lines.",
    )
    args = parser.parse_args()

    if args.text is not None:
        raw = args.text
    elif args.input:
        raw = load_input(args.input, args.json_key)
    else:
        # Allow piping input through stdin.
        raw = sys.stdin.read()

    sentences = split_sentences_from_html(raw)

    if args.json_output:
        print(json.dumps(sentences, ensure_ascii=False, indent=2))
        return

    if not sentences:
        print("(No sentences found.)")
        return

    for i, s in enumerate(sentences, 1):
        print(f"- (sent {i}) {s}")


if __name__ == "__main__":
    main()
