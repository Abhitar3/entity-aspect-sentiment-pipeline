# API Usage

The file `api_app.py` exposes the existing research pipeline through a local FastAPI service.

## Start API

```powershell
python -m uvicorn api_app:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Endpoints

### `GET /health`

Checks whether the API is running.

Example response:

```json
{
  "status": "ok"
}
```

### `POST /analyze`

Runs the pipeline on one or more posts.

Example request:

```json
{
  "posts": [
    "BeautifulSoup is the de-facto standard library for parsing web pages in Python. It's great for server-rendered or static content."
  ],
  "include_debug": true
}
```

Request fields:

- `posts`: list of raw post strings.
- `include_debug`: boolean. If true, returns intermediate pipeline steps.

Example response shape:

```json
{
  "num_posts": 1,
  "results": [
    {
      "post_id": 1,
      "final_triples": [],
      "debug_report": {}
    }
  ]
}
```

## Pydantic Validation

FastAPI uses Pydantic models to validate input and document output.

- `AnalyzeRequest` validates incoming posts and debug options.
- `AnalyzeResponse` defines the API response wrapper.
- `AnalyzeResult` defines each post-level result.

If the input format is invalid, FastAPI rejects the request before the pipeline runs.

### `POST /analyze-text`

Accepts one raw post as text. This is useful for a simple frontend text box.

Example request:

```json
{
  "post": "BeautifulSoup is the de-facto standard library for parsing web pages in Python. It's great for server-rendered or static content.",
  "include_debug": true
}
```

Internally, the API converts this to the same post list format used by `/analyze`.

### `POST /analyze-csv-text`

Accepts pasted CSV content and extracts posts from a selected column.

Example request:

```json
{
  "csv_text": "Post\nBeautifulSoup is useful for static pages.\nCypress is powerful for e2e testing.",
  "column_name": "Post",
  "include_debug": false
}
```

Internally, the API reads the `Post` column, converts non-empty rows into a list of posts, and sends them to the same pipeline runner.

## Future Frontend

Swagger UI is for backend testing. A Streamlit frontend can provide:

- Raw text input for one post.
- Batch input for multiple posts or CSV files.
- Final triples displayed as a table.
- Debug reports shown separately or downloadable as JSON.
