from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from lc_json import safe_json_loads
from lc_llm import get_llm, is_rate_limit_error
from text_utils import preprocess_text


POST_ENTITY_PROMPT = (
    "We are interested in software entities, which are APIs, libraries, or frameworks used in software development.\n"
    "We are not interested in programming languages or language-specific bindings/wrappers unless the binding itself is the primary entity being discussed.\n"
    "Entities may be mentioned:\n"
    "Explicitly by name\n"
    "\n"
    "Task:\n"
    "You will be given a text. Identify all software entities mentioned in the text.\n"
    "If no software entities are present, return an empty list.\n"
    "\n"
    "Rules:\n"
    "1. Extract ONLY entities explicitly mentioned in the provided text.\n"
    "2. Do NOT infer, normalize, expand, or introduce entities not present in the text.\n"
    "3. Do NOT extract programming languages (e.g., Python, Java, C++).\n"
    "4. Do NOT extract generic terms (e.g., \"database\", \"framework\", \"API\" without a specific name).\n"
    "5. Preserve each entity exactly as it appears in the text — do NOT change casing or modify the name.\n"
    "6. Return each entity only once per text.\n"
    "\n"
    "Output format:\n"
    "{{\n"
    "  \"post\": \"<original post>\",\n"
    "  \"entities\": [\"<entity1>\", \"<entity2>\"]\n"
    "}}\n"
    "\n"
    "POST\n"
    "{post}\n"
)

SYSTEM_ENTITY = (
    "You are a strict JSON generator.\n"
    "Return ONLY valid JSON. No markdown. No explanations.\n"
    "Your JSON MUST contain exactly two keys: post and entities.\n"
    "post must be a JSON string.\n"
    "entities must be a JSON list of strings.\n"
    "Rules:\n"
    "1) Preserve entity strings exactly as written in the post.\n"
    "2) Do not change casing or normalize names.\n"
    "3) Extract only entities explicitly grounded in the provided post text.\n"
)

COMMON_LANGUAGES = {
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go", "rust",
    "php", "ruby", "swift", "kotlin", "scala", "r", "matlab", "perl", "sql",
    "html", "css",
}

GENERIC_NON_SOFTWARE = {
    "rdbms", "dbms", "database", "database engine", "engine",
    "search engine", "index", "indexes", "server", "servers",
    "cluster", "storage", "filesystem", "os", "operating system",
    "api", "framework", "library", "tool", "sdk", "compiler",
}

CONCEPT_TERMS = {
    "ajax", "rest", "http", "https", "json", "xml", "oauth", "jwt", "grpc",
    "tcp", "udp", "ip", "regex", "ssh",
}

FILELIKE_EXTENSIONS = {
    "xml", "yml", "yaml", "json", "toml", "ini", "cfg", "conf", "properties",
    "md", "txt", "csv", "tsv", "log", "lock", "env", "gradle", "iml",
    "sql", "db", "sqlite", "bak", "tmp",
}

GENERIC_PATTERNS = [
    r"^(database|search)\s+engine$",
    r"^operating\s+system$",
    r"^(index|indexes)$",
    r"^(server|servers)$",
]

GENERIC_ACRONYMS = {"api", "db", "dba", "rdbms", "dbms", "ui", "ux"}


@dataclass
class ExtractionResult:
    entities: List[str]
    evidence: Dict[str, List[str]]


def _default_llm(temperature: float = 0.0) -> BaseChatModel:
    return get_llm(temperature=temperature)


def _clean_surface(s: str) -> str:
    return s.strip()


def _looks_like_generic_acronym(entity: str) -> bool:
    return entity.lower().strip() in GENERIC_ACRONYMS


def _is_generic(entity: str) -> bool:
    low = entity.strip().lower()
    if not low:
        return True

    if "." in low and " " not in low:
        _, _, ext = low.rpartition(".")
        if ext in FILELIKE_EXTENSIONS:
            return True

    if low in COMMON_LANGUAGES:
        return True

    if low in GENERIC_NON_SOFTWARE:
        return True

    if low in CONCEPT_TERMS:
        return True

    for pat in GENERIC_PATTERNS:
        if re.fullmatch(pat, low):
            return True

    if len(low) <= 5 and entity.isupper() and _looks_like_generic_acronym(entity):
        return True

    return False


def _appears_in_post_exact(entity: str, post_text: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(entity)}(?!\w)", post_text))


def _dedup_exact(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def extract_entities_from_post(
    post_text: str,
    debug: bool = False,
    llm: Optional[BaseChatModel] = None,
) -> ExtractionResult:
    llm = llm or _default_llm(temperature=0.0)

    post_clean = preprocess_text(post_text)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_ENTITY),
            ("user", "{user_text}"),
        ]
    )
    user_text = POST_ENTITY_PROMPT.format(post=post_clean)

    chain = prompt | llm | StrOutputParser()
    raw = ""
    for attempt in range(4):
        try:
            raw = chain.invoke({"user_text": user_text})
            break
        except Exception as e:
            if is_rate_limit_error(e):
                if attempt == 3:
                    if debug:
                        print(f"[ENTITY DEBUG] Rate limit after retries: {e}")
                    return ExtractionResult(entities=[], evidence={})
                wait_s = min(8.0, 1.5 * (2 ** attempt))
                if debug:
                    print(f"[ENTITY DEBUG] Rate limit, retrying in {wait_s:.1f}s...")
                time.sleep(wait_s)
                continue
            if debug:
                print(f"[ENTITY DEBUG] LLM call failed ({type(e).__name__}): {e}")
            return ExtractionResult(entities=[], evidence={})

    out = safe_json_loads(raw)
    ents = out.get("entities", [])
    if not isinstance(ents, list):
        ents = []

    cleaned: List[str] = []
    for value in ents:
        if not isinstance(value, str):
            continue
        name = _clean_surface(value)
        if not name:
            continue
        if _is_generic(name):
            continue
        if not _appears_in_post_exact(name, post_clean):
            continue
        cleaned.append(name)

    cleaned = _dedup_exact(cleaned)
    evidence = {entity: ["context:post"] for entity in cleaned}

    if debug:
        print("\n[ENTITY EXTRACTION]")
        print("Raw LLM JSON:", out)
        print("After filters:", cleaned)

    return ExtractionResult(entities=cleaned, evidence=evidence)


if __name__ == "__main__":
    sample_post = """I am trying to use Chart.js bar chart and get rid of what appears on the left of the generated canvas.

I am using "chart.js": "^4.5.1" npm package (currently latest) and the following code snippet:
"""
    res = extract_entities_from_post(sample_post, debug=True)
    print("\nEntities:", res.entities)
    print("Evidence:", res.evidence)
