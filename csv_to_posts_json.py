import argparse
import csv
import json
from pathlib import Path


def csv_to_posts_json(input_csv: Path, output_json: Path, column_name: str = "Sentences") -> int:
    if not input_csv.exists():
        raise FileNotFoundError(f"CSV file not found: {input_csv}")

    posts = []
    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV file has no header row: {input_csv}")

        if column_name not in reader.fieldnames:
            raise ValueError(
                f"Column '{column_name}' not found in {input_csv}. "
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:
            text = (row.get(column_name) or "").strip()
            if text:
                posts.append(text)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump({"posts": posts}, f, indent=2, ensure_ascii=False)

    return len(posts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a CSV Sentences column to pipeline input JSON.")
    parser.add_argument("input", help="Input CSV file path")
    parser.add_argument("output", help="Output JSON file path")
    parser.add_argument(
        "--column",
        default="Sentences",
        help="CSV column header containing the post/sentence text",
    )
    args = parser.parse_args()

    count = csv_to_posts_json(Path(args.input), Path(args.output), args.column)
    print(f"Created {args.output} with {count} posts.")


if __name__ == "__main__":
    main()
