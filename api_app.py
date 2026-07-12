from __future__ import annotations

import csv
from io import StringIO
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pipeline_langchain import run_one_post


app = FastAPI(
    title="Entity-Aspect-Sentiment Pipeline API",
    description="Local FastAPI wrapper for the research NLP/LLM pipeline.",
    version="0.1.0",
)


class AnalyzeRequest(BaseModel):
    posts: list[str] = Field(..., min_length=1, description="Full developer posts to analyze.")
    include_debug: bool = Field(False, description="Return intermediate pipeline steps when true.")


class AnalyzeTextRequest(BaseModel):
    post: str = Field(..., min_length=1, description="One raw developer post pasted as text.")
    include_debug: bool = Field(False, description="Return intermediate pipeline steps when true.")


class AnalyzeCsvTextRequest(BaseModel):
    csv_text: str = Field(..., min_length=1, description="CSV content pasted as text.")
    column_name: str = Field("Post", min_length=1, description="CSV column containing post text.")
    include_debug: bool = Field(False, description="Return intermediate pipeline steps when true.")


class AnalyzeResult(BaseModel):
    post_id: int
    final_triples: list[dict[str, Any]]
    debug_report: dict[str, Any] | None = None


class AnalyzeResponse(BaseModel):
    num_posts: int
    results: list[AnalyzeResult]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _analyze_posts(posts: list[str], include_debug: bool) -> AnalyzeResponse:
    results: list[AnalyzeResult] = []

    for post_id, post in enumerate(posts, 1):
        if not post.strip():
            raise HTTPException(status_code=400, detail=f"Post {post_id} is empty.")

        try:
            if include_debug:
                triples, report = run_one_post(
                    post_id=post_id,
                    post=post,
                    debug=False,
                    dedup=True,
                    print_table=False,
                    return_report=True,
                )
                results.append(
                    AnalyzeResult(
                        post_id=post_id,
                        final_triples=triples,
                        debug_report=report,
                    )
                )
            else:
                triples = run_one_post(
                    post_id=post_id,
                    post=post,
                    debug=False,
                    dedup=True,
                    print_table=False,
                )
                results.append(
                    AnalyzeResult(
                        post_id=post_id,
                        final_triples=triples,
                    )
                )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "post_id": post_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            ) from exc

    return AnalyzeResponse(num_posts=len(posts), results=results)


def _posts_from_csv_text(csv_text: str, column_name: str) -> list[str]:
    reader = csv.DictReader(StringIO(csv_text))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV input has no header row.")

    if column_name not in reader.fieldnames:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Column '{column_name}' was not found in the CSV.",
                "available_columns": reader.fieldnames,
            },
        )

    posts: list[str] = []
    for row_number, row in enumerate(reader, 2):
        value = (row.get(column_name) or "").strip()
        if value:
            posts.append(value)

    if not posts:
        raise HTTPException(
            status_code=400,
            detail=f"CSV column '{column_name}' does not contain any non-empty posts.",
        )

    return posts


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    return _analyze_posts(request.posts, request.include_debug)


@app.post("/analyze-text", response_model=AnalyzeResponse)
def analyze_text(request: AnalyzeTextRequest) -> AnalyzeResponse:
    return _analyze_posts([request.post], request.include_debug)


@app.post("/analyze-csv-text", response_model=AnalyzeResponse)
def analyze_csv_text(request: AnalyzeCsvTextRequest) -> AnalyzeResponse:
    posts = _posts_from_csv_text(request.csv_text, request.column_name)
    return _analyze_posts(posts, request.include_debug)
