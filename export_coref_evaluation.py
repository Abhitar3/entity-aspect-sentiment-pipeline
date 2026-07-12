import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _join(values: list[Any]) -> str:
    return ", ".join(str(v) for v in values)


def _count_refs(coref_sentences: list[dict[str, Any]]) -> int:
    return sum(len(sentence.get("refs_found") or []) for sentence in coref_sentences)


def _flatten_refs(coref_sentences: list[dict[str, Any]]) -> str:
    rows = []
    for sentence in coref_sentences:
        sent_id = sentence.get("idx", "")
        for ref in sentence.get("refs_found") or []:
            rows.append(f"s{sent_id}:{ref}")
    return "; ".join(rows)


def _positive_aspects(candidate: dict[str, Any]) -> str:
    aspects = candidate.get("positive_aspects") or []
    return "; ".join(
        f"{item.get('aspect', '')}:{item.get('sentiment', '')}"
        for item in aspects
        if isinstance(item, dict)
    )


def _load_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def export_coref_evaluation(debug_dir: Path, output_csv: Path) -> None:
    report_paths = sorted(debug_dir.glob("post_*_debug.json"))
    if not report_paths:
        raise FileNotFoundError(f"No post_*_debug.json files found in {debug_dir}")

    rows: list[dict[str, Any]] = []

    for report_path in report_paths:
        report = _load_report(report_path)
        post_id = report.get("post_id", "")
        post_preview = (report.get("input") or {}).get("post_preview", "")
        coref_sentences = (report.get("coreference") or {}).get("sentences") or []
        candidates = report.get("ref_aspect_candidates") or []

        positive_candidates = [
            candidate for candidate in candidates
            if candidate.get("positive_aspects")
        ]
        resolved_positive = [
            candidate for candidate in positive_candidates
            if candidate.get("resolved_entities")
        ]
        unresolved_positive = [
            candidate for candidate in positive_candidates
            if not candidate.get("resolved_entities")
        ]

        if candidates:
            for candidate in candidates:
                rows.append(
                    {
                        "post_id": post_id,
                        "debug_file": str(report_path),
                        "post_preview": post_preview,
                        "total_system_refs_found": _count_refs(coref_sentences),
                        "system_refs_found": _flatten_refs(coref_sentences),
                        "total_ref_candidates": len(candidates),
                        "total_aspect_positive_refs": len(positive_candidates),
                        "total_resolved_aspect_positive_refs": len(resolved_positive),
                        "total_unresolved_aspect_positive_refs": len(unresolved_positive),
                        "sentence_id": candidate.get("sentence_id", ""),
                        "ref": candidate.get("ref", ""),
                        "mention_text": candidate.get("mention_text", ""),
                        "sentence": candidate.get("original_context", ""),
                        "classification_view": candidate.get("classification_view", ""),
                        "positive_aspects": _positive_aspects(candidate),
                        "resolved_entities": _join(candidate.get("resolved_entities") or []),
                        "status": candidate.get("status", ""),
                        "human_expected_entities": "",
                        "coref_correct": "",
                        "notes": "",
                    }
                )
        else:
            rows.append(
                {
                    "post_id": post_id,
                    "debug_file": str(report_path),
                    "post_preview": post_preview,
                    "total_system_refs_found": _count_refs(coref_sentences),
                    "system_refs_found": _flatten_refs(coref_sentences),
                    "total_ref_candidates": 0,
                    "total_aspect_positive_refs": 0,
                    "total_resolved_aspect_positive_refs": 0,
                    "total_unresolved_aspect_positive_refs": 0,
                    "sentence_id": "",
                    "ref": "",
                    "mention_text": "",
                    "sentence": "",
                    "classification_view": "",
                    "positive_aspects": "",
                    "resolved_entities": "",
                    "status": "no_ref_candidates",
                    "human_expected_entities": "",
                    "coref_correct": "",
                    "notes": "",
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "post_id",
        "debug_file",
        "post_preview",
        "total_system_refs_found",
        "system_refs_found",
        "total_ref_candidates",
        "total_aspect_positive_refs",
        "total_resolved_aspect_positive_refs",
        "total_unresolved_aspect_positive_refs",
        "sentence_id",
        "ref",
        "mention_text",
        "sentence",
        "classification_view",
        "positive_aspects",
        "resolved_entities",
        "status",
        "human_expected_entities",
        "coref_correct",
        "notes",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved coreference evaluation CSV to {output_csv}")
    print(f"Reports processed: {len(report_paths)}")
    print(f"Rows written: {len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export coreference evaluation rows from debug reports.")
    parser.add_argument("--debug-dir", default="posts_debug_reports", help="Directory with post_*_debug.json files")
    parser.add_argument("--output", default="coref_evaluation.csv", help="Output CSV path")
    args = parser.parse_args()

    export_coref_evaluation(Path(args.debug_dir), Path(args.output))


if __name__ == "__main__":
    main()
