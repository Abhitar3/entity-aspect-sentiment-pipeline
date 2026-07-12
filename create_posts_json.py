import json
import sys

def create_posts_json(input_file: str, output_file: str):
    """
    Reads a text file where each line is a post, and creates a JSON file
    with the format expected by pipeline_langchain.py.
    """
    posts = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                posts.append(line)

    data = {"posts": posts}

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Created {output_file} with {len(posts)} posts.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_posts_json.py <input_text_file> <output_json_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    create_posts_json(input_file, output_file)