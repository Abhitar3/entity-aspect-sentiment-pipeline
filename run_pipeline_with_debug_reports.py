import argparse
import json
import traceback
from pathlib import Path

from pipeline_langchain import load_input_json, run_one_post


def _load_existing_triples(report_path: Path) -> list[dict[str, str]]:
    with report_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    triples = data.get("final_triples", [])
    return triples if isinstance(triples, list) else []


def main() -> None:
    parser = argparse.ArgumentParser(description="Run old post-level pipeline and save per-post debug reports.")
    parser.add_argument("--input", required=True, help="Input JSON file with posts")
    parser.add_argument("--output", default="posts_result.json", help="Combined final triples JSON output")
    parser.add_argument("--debug-dir", default="debug_reports", help="Directory for per-post debug reports")
    parser.add_argument("--debug-print", action="store_true", help="Also print verbose debug logs to terminal")
    parser.add_argument("--overwrite", action="store_true", help="Re-run posts even if debug reports already exist")
    args = parser.parse_args()

    posts = load_input_json(args.input)
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    all_triples = []
    errors = []
    for i, post in enumerate(posts, 1):
        report_path = debug_dir / f"post_{i:03d}_debug.json"

        if report_path.exists() and not args.overwrite:
            print(f"Skipping post {i}: {report_path} already exists")
            triples = _load_existing_triples(report_path)
            all_triples.extend(triples)
            continue

        try:
            print(f"Running post {i}/{len(posts)}")
            triples = run_one_post(
                post_id=i,
                post=post,
                debug=args.debug_print,
                dedup=True,
                print_table=False,
                save_report_path=str(report_path),
            )
        except Exception as exc:
            error_record = {
                "post_id": i,
                "post_preview": post[:500],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            error_path = debug_dir / f"post_{i:03d}_error.json"
            with error_path.open("w", encoding="utf-8") as f:
                json.dump(error_record, f, indent=2, ensure_ascii=False)
            errors.append(error_record)
            print(f"Post {i} failed; saved error to {error_path}")
            continue

        all_triples.extend(triples)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_posts": len(posts),
                "debug_dir": str(debug_dir),
                "num_errors": len(errors),
                "errors": [
                    {
                        "post_id": error["post_id"],
                        "error_type": error["error_type"],
                        "error": error["error"],
                    }
                    for error in errors
                ],
                "final_triples": all_triples,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved combined output to {output_path}")
    print(f"Saved per-post debug reports to {debug_dir}")
    if errors:
        print(f"Completed with {len(errors)} failed posts. See *_error.json files in {debug_dir}.")


if __name__ == "__main__":
    main()
