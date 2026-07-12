from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


GOLD_COLUMN = "Tuples_FormattedlikePipelineOutput"
SYSTEM_COLUMN = "System_Output"

ASPECT_CATEGORIES = [
    "Aspect A Positive",
    "Aspect A Negative",
    "Aspect B Positive",
    "Aspect B Negative",
]


def parse_json_like_output(text: Any) -> List[Dict[str, Any]]:
    """
    Parse JSON-like triple text into a list of dictionaries.

    Handles cells like:
    - [{...}]
    - {...}, {...}
    - empty cells

    If the text does not start with '[', it is wrapped in square brackets before parsing.
    """
    if text is None:
        return []

    value = str(text).strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        return []

    if not value.startswith("["):
        value = f"[{value}]"

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def _normalize_aspect(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    low = text.lower()
    for aspect in ASPECT_CATEGORIES:
        if low == aspect.lower():
            return aspect
    return text


def _normalize_sentiment(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    low = text.lower()
    if low == "positive":
        return "Positive"
    if low == "negative":
        return "Negative"
    if low == "neutral":
        return "Neutral"
    if low == "none":
        return "None"
    return text


def normalize_triple(record: Dict[str, Any]) -> Dict[str, str]:
    """
    Keep only entity, aspect, and sentiment.

    post_id and sentence_id are intentionally ignored because comparison is row-aligned.
    """
    return {
        "entity": " ".join(str(record.get("entity", "")).strip().lower().split()),
        "aspect": _normalize_aspect(record.get("aspect", "")),
        "sentiment": _normalize_sentiment(record.get("sentiment", "")),
    }


def standardize_triples(records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    standardized: List[Dict[str, str]] = []
    for record in records:
        triple = normalize_triple(record)
        if triple["entity"] and triple["aspect"]:
            standardized.append(triple)
    return standardized


def _triple_key(triple: Dict[str, str]) -> Tuple[str, str, str]:
    return (triple["entity"], triple["aspect"], triple["sentiment"])


def _counter_to_json(counter: Counter[Tuple[str, str, str]]) -> str:
    rows: List[Dict[str, str]] = []
    for (entity, aspect, sentiment), count in counter.items():
        for _ in range(count):
            rows.append(
                {
                    "entity": entity,
                    "aspect": aspect,
                    "sentiment": sentiment,
                }
            )
    return json.dumps(rows, ensure_ascii=False)


def _aspect_count(counter: Counter[Tuple[str, str, str]], aspect: str) -> int:
    return sum(count for (_, triple_aspect, _), count in counter.items() if triple_aspect == aspect)


def compare_row(gold_text: Any, system_text: Any) -> Dict[str, Any]:
    """
    Compare one row using gold/reference triples as the only required targets.

    Extra system triples are ignored. A full match requires:
    entity -> aspect -> sentiment
    """
    gold_records = parse_json_like_output(gold_text)
    system_records = parse_json_like_output(system_text)

    gold_triples = standardize_triples(gold_records)
    system_triples = standardize_triples(system_records)

    gold_keys = [_triple_key(triple) for triple in gold_triples]
    system_keys = [_triple_key(triple) for triple in system_triples]

    gold_counter = Counter(gold_keys)
    system_counter = Counter(system_keys)
    matched_counter = gold_counter & system_counter
    missing_counter = gold_counter - system_counter

    gold_entity_counter = Counter(entity for entity, _, _ in gold_keys)
    system_entity_counter = Counter(entity for entity, _, _ in system_keys)

    gold_entity_aspect_counter = Counter((entity, aspect) for entity, aspect, _ in gold_keys)
    system_entity_aspect_counter = Counter((entity, aspect) for entity, aspect, _ in system_keys)

    entity_matches = sum((gold_entity_counter & system_entity_counter).values())
    aspect_matches = sum((gold_entity_aspect_counter & system_entity_aspect_counter).values())
    exact_matches = sum(matched_counter.values())

    result: Dict[str, Any] = {
        "gold_total": len(gold_keys),
        "system_total": len(system_keys),
        "exact_matches": exact_matches,
        "missing_gold_triples": _counter_to_json(missing_counter),
        "entity_matches_against_gold": entity_matches,
        "aspect_matches_after_entity_match": aspect_matches,
        "sentiment_matches_after_entity_aspect_match": exact_matches,
    }

    for aspect in ASPECT_CATEGORIES:
        prefix = aspect.lower().replace(" ", "_")
        result[f"{prefix}_gold"] = _aspect_count(gold_counter, aspect)
        result[f"{prefix}_matched"] = _aspect_count(matched_counter, aspect)

    return result


def _safe_compare_row(gold_text: Any, system_text: Any) -> Dict[str, Any]:
    try:
        result = compare_row(gold_text, system_text)
        result["parse_error"] = ""
        return result
    except Exception as exc:
        result = {
            "gold_total": 0,
            "system_total": 0,
            "exact_matches": 0,
            "missing_gold_triples": "[]",
            "aspect_a_positive_gold": 0,
            "aspect_a_positive_matched": 0,
            "aspect_a_negative_gold": 0,
            "aspect_a_negative_matched": 0,
            "aspect_b_positive_gold": 0,
            "aspect_b_positive_matched": 0,
            "aspect_b_negative_gold": 0,
            "aspect_b_negative_matched": 0,
            "entity_matches_against_gold": 0,
            "aspect_matches_after_entity_match": 0,
            "sentiment_matches_after_entity_aspect_match": 0,
            "parse_error": str(exc),
        }
        return result


def compare_file(
    input_path: str | Path,
    output_path: str | Path = "comparison_results.csv",
    summary_path: str | Path = "comparison_summary.csv",
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    summary_path = Path(summary_path)

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Input CSV has no header row: {input_path}")
        if GOLD_COLUMN not in reader.fieldnames:
            raise ValueError(f"Missing gold column '{GOLD_COLUMN}'. Available columns: {reader.fieldnames}")
        if SYSTEM_COLUMN not in reader.fieldnames:
            raise ValueError(f"Missing system column '{SYSTEM_COLUMN}'. Available columns: {reader.fieldnames}")

        input_rows = list(reader)
        input_fieldnames = list(reader.fieldnames)

    result_rows: List[Dict[str, Any]] = []
    for row in input_rows:
        comparison = _safe_compare_row(row.get(GOLD_COLUMN, ""), row.get(SYSTEM_COLUMN, ""))
        result_rows.append({**row, **comparison})

    comparison_fields = [
        "gold_total",
        "system_total",
        "exact_matches",
        "missing_gold_triples",
        "aspect_a_positive_gold",
        "aspect_a_positive_matched",
        "aspect_a_negative_gold",
        "aspect_a_negative_matched",
        "aspect_b_positive_gold",
        "aspect_b_positive_matched",
        "aspect_b_negative_gold",
        "aspect_b_negative_matched",
        "entity_matches_against_gold",
        "aspect_matches_after_entity_match",
        "sentiment_matches_after_entity_aspect_match",
        "parse_error",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=input_fieldnames + comparison_fields)
        writer.writeheader()
        writer.writerows(result_rows)

    summary: Dict[str, Any] = {
        "rows": len(result_rows),
        "total_gold_triples": sum(int(row["gold_total"]) for row in result_rows),
        "total_system_triples": sum(int(row["system_total"]) for row in result_rows),
        "total_exact_matches": sum(int(row["exact_matches"]) for row in result_rows),
        "total_entity_matches_against_gold": sum(int(row["entity_matches_against_gold"]) for row in result_rows),
        "total_aspect_matches_after_entity_match": sum(int(row["aspect_matches_after_entity_match"]) for row in result_rows),
        "total_sentiment_matches_after_entity_aspect_match": sum(
            int(row["sentiment_matches_after_entity_aspect_match"]) for row in result_rows
        ),
        "parse_error_rows": sum(1 for row in result_rows if row.get("parse_error")),
    }

    for aspect in ASPECT_CATEGORIES:
        prefix = aspect.lower().replace(" ", "_")
        gold_count = sum(int(row[f"{prefix}_gold"]) for row in result_rows)
        matched_count = sum(int(row[f"{prefix}_matched"]) for row in result_rows)
        summary[f"{prefix}_gold"] = gold_count
        summary[f"{prefix}_matched"] = matched_count
        summary[f"{prefix}_match_rate"] = matched_count / gold_count if gold_count else ""

    total_gold = int(summary["total_gold_triples"])
    total_matches = int(summary["total_exact_matches"])
    summary["overall_exact_match_rate"] = total_matches / total_gold if total_gold else ""

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f"Saved row-level results to {output_path}")
    print(f"Saved summary to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare gold annotation triples against system triples.")
    parser.add_argument("input", help="Input CSV file containing gold and system output columns")
    parser.add_argument("--output", default="comparison_results.csv", help="Row-level output CSV path")
    parser.add_argument("--summary", default="comparison_summary.csv", help="Summary output CSV path")
    args = parser.parse_args()

    compare_file(args.input, args.output, args.summary)


if __name__ == "__main__":
    main()
