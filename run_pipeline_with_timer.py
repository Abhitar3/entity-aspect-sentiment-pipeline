import argparse
import json
import time
from pathlib import Path

from pipeline_langchain_new import load_input_json, run_one_post



def run_and_time(input_json: Path, output_json: Path | None = None, debug: bool = False) -> float:
    posts = load_input_json(str(input_json))
    all_triples = []

    start_time = time.perf_counter()
    for i, post in enumerate(posts, 1):
        triples = run_one_post(
            post_id=i,
            post=post,
            debug=debug,
            dedup=True,
            print_table=False,
            save_report_path=None,
        )
        all_triples.extend(triples)
    elapsed_seconds = time.perf_counter() - start_time

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_posts": len(posts),
            "elapsed_seconds": elapsed_seconds,
            "final_triples": all_triples,
        }
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    return elapsed_seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipeline and print elapsed time.")
    parser.add_argument("--input", required=True, help="Input JSON file with posts")
    parser.add_argument("--output", default=None, help="Optional JSON file to save results")
    parser.add_argument("--debug", action="store_true", help="Enable pipeline debug output")
    args = parser.parse_args()

    posts = load_input_json(args.input)
    elapsed = run_and_time(Path(args.input), Path(args.output) if args.output else None, debug=args.debug)
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Posts processed: {len(posts)}")


if __name__ == "__main__":
    main()
