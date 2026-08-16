from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

from aspect_b_subaspect_langchain import classify_aspect_b_subaspects_llm
from coref_langchain import find_entity_mentions, find_ref_occurrences
from entity_extraction_langchain import extract_entities_from_post
from pipeline_langchain import load_input_json, preview
from text_utils import preprocess_text, split_into_sentences


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _tag_span(text: str, start: int, end: int) -> str:
    return text[:start] + f"<E>{text[start:end]}</E>" + text[end:]


def _build_mentions(
    post_id: int,
    sentence_id: int,
    sentence: str,
    entities: List[str],
    next_mention_id: int,
) -> Tuple[List[Dict[str, Any]], int]:
    mentions: List[Dict[str, Any]] = []

    for occurrence_index, (entity, start, end) in enumerate(find_entity_mentions(sentence, entities), 1):
        mentions.append(
            {
                "mention_id": next_mention_id,
                "post_id": post_id,
                "sentence_id": sentence_id,
                "target_mention": entity,
                "mention_type": "explicit_text",
                "original_context": sentence,
                "mention_text": sentence[start:end],
                "mention_span": {"start": start, "end": end},
                "occurrence_index": occurrence_index,
                "classification_view": _tag_span(sentence, start, end),
            }
        )
        next_mention_id += 1

    ref_occurrence_counts: Dict[str, int] = {}
    for ref, start, end in find_ref_occurrences(sentence):
        ref_occurrence_counts[ref] = ref_occurrence_counts.get(ref, 0) + 1
        mentions.append(
            {
                "mention_id": next_mention_id,
                "post_id": post_id,
                "sentence_id": sentence_id,
                "target_mention": sentence[start:end],
                "mention_type": "referring_expression",
                "ref": ref,
                "original_context": sentence,
                "mention_text": sentence[start:end],
                "mention_span": {"start": start, "end": end},
                "occurrence_index": ref_occurrence_counts[ref],
                "classification_view": _tag_span(sentence, start, end),
            }
        )
        next_mention_id += 1

    return mentions, next_mention_id


def run_one_post(
    post_id: int,
    post: str,
    debug: bool = False,
) -> Dict[str, Any]:
    post_clean = preprocess_text(post)
    sentences = split_into_sentences(post_clean)
    extraction = extract_entities_from_post(post, debug=False)
    entities = extraction.entities

    mentions: List[Dict[str, Any]] = []
    next_mention_id = 1
    for sentence_id, sentence in enumerate(sentences, 1):
        sentence_mentions, next_mention_id = _build_mentions(
            post_id=post_id,
            sentence_id=sentence_id,
            sentence=sentence,
            entities=entities,
            next_mention_id=next_mention_id,
        )
        mentions.extend(sentence_mentions)

    classification_views = [mention["classification_view"] for mention in mentions]
    aspect_outputs = classify_aspect_b_subaspects_llm(classification_views, debug=debug)

    results: List[Dict[str, Any]] = []
    for mention, aspect_output in zip(mentions, aspect_outputs):
        results.append(
            {
                "post_id": str(post_id),
                "sentence_id": str(mention["sentence_id"]),
                "mention_id": mention["mention_id"],
                "target_mention": mention["target_mention"],
                "mention_type": mention["mention_type"],
                "classification_view": mention["classification_view"],
                "aspect_b": aspect_output.get("answer", "No"),
                "sentiment": aspect_output.get("sentiment", "None"),
                "codes": aspect_output.get("codes", []),
                "reason": aspect_output.get("reason", ""),
            }
        )

    return {
        "post_id": post_id,
        "input": {"post_preview": preview(post, 400)},
        "preprocess": {"post_clean_preview": preview(post_clean, 400)},
        "sentence_split": sentences,
        "entity_extraction": {
            "entities": entities,
            "evidence": getattr(extraction, "evidence", {}),
        },
        "mentions": mentions,
        "aspect_b_subaspect_outputs": aspect_outputs,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Aspect B sub-aspect pipeline.")
    parser.add_argument("--input", required=True, help="Input JSON with post/posts.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--debug-dir", default=None, help="Optional directory for per-post debug reports.")
    parser.add_argument("--debug-print", action="store_true", help="Print per-post progress.")
    args = parser.parse_args()

    posts = load_input_json(args.input)
    all_results: List[Dict[str, Any]] = []
    reports: List[Dict[str, Any]] = []

    if args.debug_dir:
        ensure_dir(args.debug_dir)

    for post_id, post in enumerate(posts, 1):
        if args.debug_print:
            print(f"Running post {post_id}/{len(posts)}")

        report = run_one_post(post_id=post_id, post=post, debug=args.debug_print)
        reports.append(report)
        all_results.extend(report["results"])

        if args.debug_dir:
            report_path = os.path.join(args.debug_dir, f"post_{post_id:03d}_aspect_b_subaspect_debug.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

    ensure_dir(os.path.dirname(args.output) or ".")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_posts": len(posts),
                "num_results": len(all_results),
                "results": all_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    if args.debug_print:
        print(f"Saved output to {args.output}")
        if args.debug_dir:
            print(f"Saved debug reports to {args.debug_dir}")


if __name__ == "__main__":
    main()
