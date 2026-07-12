from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from aspect_langchain import classify_aspects_labels_llm
from coref_langchain import (
    find_entity_mentions,
    find_ref_occurrences,
    resolve_ref_occurrence_llm,
)
from entity_extraction_langchain import extract_entities_from_post
from lc_llm import get_llm
from text_utils import preprocess_text, split_into_sentences

load_dotenv()


def load_input_json(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_items: List[Any] = []

    if isinstance(data, dict):
        if "posts" in data:
            posts_obj = data.get("posts")
            if not isinstance(posts_obj, list) or not posts_obj:
                raise ValueError("Input JSON key 'posts' must be a non-empty list.")
            raw_items = posts_obj
        elif "post" in data:
            raw_items = [data.get("post", "")]
        else:
            raise ValueError("Input JSON must contain either 'post' (string) or 'posts' (list).")
    elif isinstance(data, list):
        if not data:
            raise ValueError("Input JSON list must be non-empty.")
        raw_items = data
    else:
        raise ValueError("Input JSON must be an object or list.")

    posts: List[str] = []
    for i, item in enumerate(raw_items, 1):
        post_val: Any = item
        if isinstance(item, dict):
            post_val = item.get("post", "")

        if not isinstance(post_val, str) or not post_val.strip():
            raise ValueError(f"Post at index {i} is invalid. Expected a non-empty string.")
        posts.append(post_val)

    return posts


def preview(text: str, n: int = 280) -> str:
    t = text.replace("\n", " ").strip()
    return t if len(t) <= n else t[:n] + " ..."


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def format_triples_table(triples: List[Dict[str, str]]) -> str:
    headers = ["Post#", "Sentence#", "Entity", "Aspect", "Sentiment"]
    rows: List[List[str]] = [
        [
            str(t.get("post_id", "")),
            str(t.get("sentence_id", "")),
            str(t.get("entity", "")),
            str(t.get("aspect", "")),
            str(t.get("sentiment", "")),
        ]
        for t in triples
    ]

    if not rows:
        return "(No triples)"

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _sep(ch: str = "-") -> str:
        return "+" + "+".join(ch * (w + 2) for w in widths) + "+"

    def _render_row(row: List[str], is_header: bool = False) -> str:
        cells: List[str] = []
        for i, cell in enumerate(row):
            if is_header:
                cells.append(cell.ljust(widths[i]))
            elif i in {0, 1}:
                cells.append(cell.rjust(widths[i]))
            else:
                cells.append(cell.ljust(widths[i]))
        return "| " + " | ".join(cells) + " |"

    lines = [_sep("-"), _render_row(headers, is_header=True), _sep("=")]
    for row in rows:
        lines.append(_render_row(row))
    lines.append(_sep("-"))
    return "\n".join(lines)


def dedup_triples(triples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out: List[Dict[str, str]] = []
    for triple in triples:
        key = (
            str(triple["post_id"]).strip().lower(),
            str(triple["sentence_id"]).strip().lower(),
            str(triple["entity"]).strip(),
            str(triple["aspect"]).strip().lower(),
            str(triple.get("sentiment", "")).strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(triple)
    return out


def _norm_sentiment(value: Any) -> str:
    s = str(value or "").strip().lower()
    if s == "positive":
        return "Positive"
    if s == "negative":
        return "Negative"
    if s == "neutral":
        return "Neutral"
    return "None"


def labels_to_aspects(label_out: Any) -> List[Tuple[str, str, str]]:
    """
    Returns list of tuples: (aspect_label, sentiment, presence)
    aspect_label: "Aspect A" or "Aspect B"
    sentiment: "Positive", "Negative", "Neutral", or "None"
    presence: "Positive" (yes, discussed) or "Negative" (no, not discussed)
    """
    aspects: List[Tuple[str, str, str]] = []

    if isinstance(label_out, dict):
        # Aspect A
        a_answer = str(label_out.get("A", "")).strip().lower()
        if a_answer == "yes":
            a_sentiment = _norm_sentiment(label_out.get("A_sentiment", "None"))
            if a_sentiment not in {"Positive", "Negative", "Neutral"}:
                a_sentiment = "Neutral"
        else:
            a_sentiment = "None"

        a_presence = "Positive" if a_answer == "yes" else "Negative"
        aspects.append(("Aspect A", a_sentiment, a_presence))

        # Aspect B
        b_answer = str(label_out.get("B", "")).strip().lower()
        if b_answer == "yes":
            b_sentiment = _norm_sentiment(label_out.get("B_sentiment", "None"))
            if b_sentiment not in {"Positive", "Negative", "Neutral"}:
                b_sentiment = "Neutral"
        else:
            b_sentiment = "None"

        b_presence = "Positive" if b_answer == "yes" else "Negative"
        aspects.append(("Aspect B", b_sentiment, b_presence))

    return aspects


def _tag_span(text: str, start: int, end: int, replacement: Optional[str] = None) -> str:
    target = replacement if replacement is not None else text[start:end]
    return text[:start] + f"<E>{target}</E>" + text[end:]


def _build_resolved_classification_view(sentence: str, start: int, end: int, entity: str, ref: str) -> str:
    if ref in {"it's", "itâ€™s"}:
        replacement = f"<E>{entity}</E> is"
        return sentence[:start] + replacement + sentence[end:]
    if ref == "its":
        replacement = f"<E>{entity}</E>'s"
        return sentence[:start] + replacement + sentence[end:]
    if ref == "they":
        suffix = sentence[end:]
        if suffix.lower().startswith("'re"):
            replacement = f"<E>{entity}</E> are"
            return sentence[:start] + replacement + sentence[end + 3 :]
    return sentence[:start] + f"<E>{entity}</E>" + sentence[end:]


def _build_explicit_mentions(
    post_id: int,
    sentence_id: int,
    sentence: str,
    entities: List[str],
    next_mention_id: int,
) -> Tuple[List[Dict[str, Any]], int]:
    mentions: List[Dict[str, Any]] = []
    explicit_spans = find_entity_mentions(sentence, entities)

    for occurrence_index, (entity, start, end) in enumerate(explicit_spans, 1):
        mentions.append(
            {
                "mention_id": next_mention_id,
                "post_id": post_id,
                "sentence_id": sentence_id,
                "entity": entity,
                "source": "text",
                "mention_type": "explicit_text",
                "original_context": sentence,
                "mention_text": sentence[start:end],
                "mention_span": {"start": start, "end": end},
                "occurrence_index": occurrence_index,
                "classification_ready": True,
                "confidence": 1.0,
                "classification_view": _tag_span(sentence, start, end),
            }
        )
        next_mention_id += 1

    return mentions, next_mention_id


def _build_ref_mentions(
    post_id: int,
    sentence_id: int,
    sentence: str,
    next_mention_id: int,
) -> Tuple[List[Dict[str, Any]], int]:
    mentions: List[Dict[str, Any]] = []
    ref_occurrence_counts: Dict[str, int] = {}

    for ref, start, end in find_ref_occurrences(sentence):
        ref_occurrence_counts[ref] = ref_occurrence_counts.get(ref, 0) + 1
        mention_text = sentence[start:end]
        mentions.append(
            {
                "mention_id": next_mention_id,
                "post_id": post_id,
                "sentence_id": sentence_id,
                "entity": mention_text,
                "source": "text",
                "mention_type": "referring_expression",
                "original_context": sentence,
                "mention_text": mention_text,
                "mention_span": {"start": start, "end": end},
                "occurrence_index": ref_occurrence_counts[ref],
                "classification_ready": True,
                "confidence": 1.0,
                "ref": ref,
                "classification_view": _tag_span(sentence, start, end),
            }
        )
        next_mention_id += 1

    return mentions, next_mention_id


def _build_coref_mentions(
    post_id: int,
    sentence_id: int,
    result: Any,
    next_mention_id: int,
) -> Tuple[List[Dict[str, Any]], int]:
    mentions: List[Dict[str, Any]] = []

    for resolution in result.resolved_refs:
        if not resolution.entities:
            continue

        for entity in resolution.entities:
            mentions.append(
                {
                    "mention_id": next_mention_id,
                    "post_id": post_id,
                    "sentence_id": sentence_id,
                    "entity": entity,
                    "source": "text",
                    "mention_type": "resolved_pronoun",
                    "original_context": result.original,
                    "mention_text": result.original[resolution.start:resolution.end],
                    "mention_span": {"start": resolution.start, "end": resolution.end},
                    "occurrence_index": resolution.occurrence_index,
                    "classification_ready": True,
                    "confidence": 1.0,
                    "ref": resolution.ref,
                    "resolved_entities": list(resolution.entities),
                    "classification_view": _build_resolved_classification_view(
                        result.original,
                        resolution.start,
                        resolution.end,
                        entity,
                        resolution.ref,
                    ),
                }
            )
            next_mention_id += 1

    return mentions, next_mention_id


def run_one_post(
    post_id: int,
    post: str,
    debug: bool = True,
    dedup: bool = True,
    print_table: bool = True,
    save_report_path: Optional[str] = None,
    return_report: bool = False,
) -> List[Dict[str, str]] | Tuple[List[Dict[str, str]], Dict[str, Any]]:
    llm = get_llm(temperature=0.0)
    post_clean = preprocess_text(post)
    split_sentences = split_into_sentences(post_clean)

    report: Dict[str, Any] = {
        "post_id": post_id,
        "input": {"post_preview": preview(post, 400)},
        "preprocess": {"post_clean_preview": preview(post_clean, 400)},
        "entity_extraction": {},
        "coreference": {"mode": "aspect_positive_ref_only", "sentences": []},
        "sentence_split": split_sentences,
        "mentions": [],
        "aspect_outputs": [],
        "ref_aspect_candidates": [],
        "final_triples": [],
    }

    if debug:
        print("\n==============================")
        print("1) INPUT (POST ONLY)")
        print("==============================")
        print("Post preview:", preview(post, 400))

        print("\n==============================")
        print("2) CLEANING")
        print("==============================")
        print("Cleaned post preview:", preview(post_clean, 400))

    extraction = extract_entities_from_post(post, debug=False)
    entities = extraction.entities

    report["entity_extraction"] = {
        "raw_entities": extraction.entities,
        "entities": entities,
        "evidence": getattr(extraction, "evidence", {}),
    }

    if debug:
        print("\n==============================")
        print("3) ENTITY EXTRACTION (LangChain)")
        print("==============================")
        print("Entities:")
        for entity in entities:
            print(" -", entity)

    report["coreference"]["sentences"] = [
        {
            "idx": sentence_id,
            "original": sentence,
            "refs_found": [ref for ref, _, _ in find_ref_occurrences(sentence)],
            "explicit_mentions": [entity for entity, _, _ in find_entity_mentions(sentence, entities)],
        }
        for sentence_id, sentence in enumerate(split_sentences, 1)
    ]

    if debug:
        print("\n==============================")
        print("4) REF DETECTION")
        print("==============================")
        print("Coreference is resolved only after a referring expression is aspect-positive.")
        for item in report["coreference"]["sentences"]:
            print(f"Sentence {item['idx']}:")
            print("  Original:", item["original"])
            print("  Refs found:", item["refs_found"])
            print("  Explicit mentions:", item["explicit_mentions"])

        print("\n==============================")
        print("4.5) SENTENCE SPLIT (text_utils)")
        print("==============================")
        for i, sentence in enumerate(split_sentences, 1):
            print(f"- (sent {i}) {sentence}")

    mentions: List[Dict[str, Any]] = []
    next_mention_id = 1

    for sentence_id, sentence in enumerate(split_sentences, 1):
        explicit_mentions, next_mention_id = _build_explicit_mentions(
            post_id=post_id,
            sentence_id=sentence_id,
            sentence=sentence,
            entities=entities,
            next_mention_id=next_mention_id,
        )
        mentions.extend(explicit_mentions)

        ref_mentions, next_mention_id = _build_ref_mentions(
            post_id=post_id,
            sentence_id=sentence_id,
            sentence=sentence,
            next_mention_id=next_mention_id,
        )
        mentions.extend(ref_mentions)

    report["mentions"] = mentions

    if debug:
        print("\n==============================")
        print("5) MENTION OBJECTS / CLASSIFICATION VIEWS")
        print("==============================")
        for mention in mentions:
            mention_type = mention["mention_type"]
            ref = mention.get("ref", "")
            suffix = f":{ref}" if mention_type == "resolved_pronoun" and ref else ""
            print(
                f"- (sent {mention['sentence_id']}, {mention_type}{suffix}, id={mention['mention_id']}) "
                f"{mention['classification_view']}"
            )

    classification_views = [m["classification_view"] for m in mentions if m.get("classification_ready")]
    label_outputs: List[Dict[str, str]] = []
    if classification_views:
        label_outputs = classify_aspects_labels_llm(classification_views, debug=debug)
    report["aspect_outputs"] = label_outputs

    triples: List[Dict[str, str]] = []
    ready_mentions = [m for m in mentions if m.get("classification_ready")]

    for mention, label_out in zip(ready_mentions, label_outputs):
        aspects = labels_to_aspects(label_out)
        if not aspects:
            continue

        sentence_id = str(mention["sentence_id"])
        positive_aspects = [
            (aspect_name, sentiment, presence)
            for aspect_name, sentiment, presence in aspects
            if presence == "Positive"
        ]

        if mention.get("mention_type") == "referring_expression":
            ref_record: Dict[str, Any] = {
                "mention_id": mention["mention_id"],
                "post_id": mention["post_id"],
                "sentence_id": mention["sentence_id"],
                "ref": mention.get("ref"),
                "mention_text": mention.get("mention_text"),
                "original_context": mention.get("original_context"),
                "classification_view": mention.get("classification_view"),
                "aspect_output": label_out,
                "positive_aspects": [
                    {
                        "aspect": aspect_name,
                        "sentiment": sentiment,
                    }
                    for aspect_name, sentiment, _ in positive_aspects
                ],
                "resolved_entities": [],
                "status": "not_aspect_positive",
            }

            if positive_aspects:
                span = mention.get("mention_span", {})
                resolved_entities = resolve_ref_occurrence_llm(
                    post_text=post,
                    sentence_id=int(mention["sentence_id"]),
                    ref=str(mention.get("ref", "")),
                    start=int(span.get("start", 0)),
                    end=int(span.get("end", 0)),
                    entities=entities,
                    llm=llm,
                )
                ref_record["resolved_entities"] = resolved_entities

                if resolved_entities:
                    ref_record["status"] = "resolved_to_entity"
                    for resolved_entity in resolved_entities:
                        for aspect_name, sentiment, presence in positive_aspects:
                            aspect_label = f"{aspect_name} {presence}"
                            triples.append(
                                {
                                    "post_id": str(post_id),
                                    "sentence_id": sentence_id,
                                    "entity": resolved_entity,
                                    "aspect": aspect_label,
                                    "sentiment": sentiment,
                                    "source_ref": str(mention.get("mention_text", "")),
                                }
                            )
                else:
                    ref_record["status"] = "aspect_positive_but_unresolved"

            report["ref_aspect_candidates"].append(ref_record)
            continue

        entity = str(mention["entity"]).strip()

        for aspect_name, sentiment, presence in aspects:
            # Format: "Aspect A Positive" or "Aspect A Negative"
            aspect_label = f"{aspect_name} {presence}"
            triples.append(
                {
                    "post_id": str(post_id),
                    "sentence_id": sentence_id,
                    "entity": entity,
                    "aspect": aspect_label,
                    "sentiment": sentiment,
                }
            )

    if dedup:
        triples = dedup_triples(triples)

    report["final_triples"] = triples

    if debug:
        print("\n==============================")
        print("6) FINAL TRIPLES")
        print("==============================")
        for triple in triples:
            print(triple)

        if print_table:
            print("\n==============================")
            print("OUTPUT FORMAT: Post# | Sentence# | Entity | Aspect | Sentiment")
            print("==============================")
            print(format_triples_table(triples))

    if save_report_path:
        ensure_dir(os.path.dirname(save_report_path) or ".")
        with open(save_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        if debug:
            print("\nSaved report to:", save_report_path)

    if return_report:
        return triples, report

    return triples


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LangChain-based entity/aspect pipeline (POST-only).")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSON with either key 'post' (string) or 'posts' (list).",
    )
    parser.add_argument("--debug", action="store_true", help="Print outputs")
    parser.add_argument("--save", default=None, help="Optional path to save a JSON run report")
    args = parser.parse_args()

    posts = load_input_json(args.input)
    all_triples: List[Dict[str, str]] = []

    for i, post in enumerate(posts, 1):
        if args.debug and len(posts) > 1:
            print("\n############################################")
            print(f"RUNNING POST {i}/{len(posts)}")
            print("############################################")

        triples = run_one_post(
            post_id=i,
            post=post,
            debug=args.debug,
            dedup=True,
            print_table=False,
            save_report_path=None,
        )
        all_triples.extend(triples)

    print("\n==============================")
    print("FINAL OUTPUT TABLE (ALL POSTS)")
    print("==============================")
    print(format_triples_table(all_triples))

    if args.save:
        ensure_dir(os.path.dirname(args.save) or ".")
        payload = {
            "num_posts": len(posts),
            "final_triples": all_triples,
        }
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        if args.debug:
            print("\nSaved report to:", args.save)


if __name__ == "__main__":
    main()
