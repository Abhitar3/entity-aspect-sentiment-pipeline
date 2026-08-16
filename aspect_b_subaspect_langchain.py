from __future__ import annotations

import time
from typing import List, Optional

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

from entity_focus import extract_single_entity
from lc_llm import get_llm, is_rate_limit_error


ASPECT_B_PROMPT = (
    'I will give you a set of sentences or sentence. Each sentence will contain one target mention '
    'represented between <E> and </E>. The target mention may be a software entity name, API, library, '
    'framework, or a referring expression such as it, its, they, both, this, these, or those. '
    'We would like to know if for a given sentence does it discuss :'
    '(a) the ease of use of target mention'
    '(b) how easy it is for developers to learn and start using the target mention; '
    '(c) the target mention has attributes that make it easy to operate and control; '
    '(d) the target mention protects users against making errors; '
    '(e) how user-friendly the target mention interface or interactions are; '
    '(f) whether the target mention can be used by developers with different abilities and needs; '
    '(g) the ease of installing, updating, or uninstalling the target mention. '
    'If the sentence discusses one or more of the listed items, Select the corresponding codes from  a, b, c, d, e, f, g  and provide a short reason explaining why the selected codes apply.'
    'If none of those listed items are discussed in sentence, then answer NONE'
    'Please provide your output in JSON format. Do not provide any explanation or additional text. '
    'Sentences:\n'
)

ASPECT_B_SYSTEM = (
    "You are a strict JSON generator.\n"
    "Return ONLY schema-valid JSON.\n"
    "Task constraints:\n"
    "1) The target mention is ONLY the text wrapped in <E>...</E>.\n"
    "2) The target mention may be a software entity or a referring expression.\n"
    "3) Ignore other entities or mentions in the line.\n"
    "4) Your answer must be one or more of these codes: a, b, c, d, e, f, g or NONE \n"
    "5) The reason must be one short sentence."
    "Do not use the word aspect in the reason."
)

VALID_CODES = {"a", "b", "c", "d", "e", "f", "g"}

class CodeDecision(BaseModel):
    answer: str
    sentiment: str = "None"
    codes: List[str] = []
    reason: str = ""


def _norm_yes_no(value: str) -> Optional[str]:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v == "yes":
        return "Yes"
    if v == "no":
        return "No"
    return None


def _norm_sentiment(value: str) -> Optional[str]:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v == "positive":
        return "Positive"
    if v == "negative":
        return "Negative"
    if v == "neutral":
        return "Neutral"
    if v == "none":
        return "None"
    return None


def _norm_codes(values: object) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    seen = set()
    for value in values:
        code = str(value).strip().lower()
        if code in VALID_CODES and code not in seen:
            out.append(code)
            seen.add(code)
    return out


def classify_aspect_b_subaspects_llm(
    focused_sentences: List[str],
    debug: bool = False,
) -> List[dict[str, object]]:
    if not focused_sentences:
        return []

    llm = get_llm(temperature=0.0)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ASPECT_B_SYSTEM),
            ("user", "{user}"),
        ]
    )
    chain = prompt | llm.with_structured_output(CodeDecision)

    outputs: List[dict[str, object]] = []
    for sentence in focused_sentences:
        target = extract_single_entity(sentence) or ""
        if not target:
            outputs.append(
                {
                    "target_mention": "",
                    "answer": "No",
                    "sentiment": "None",
                    "codes": [],
                    "reason": "No target mention was marked in the sentence.",
                }
            )
            continue

        user = (
            ASPECT_B_PROMPT
            + sentence
            + "\n\nTarget mention in this line: "
            + target
        )

        result: Optional[CodeDecision] = None
        for phase in range(2):
            for attempt in range(3):
                try:
                    result = chain.invoke({"user": user})
                    break
                except Exception as exc:
                    if is_rate_limit_error(exc):
                        if attempt == 2:
                            break
                        wait_s = min(6.0, 1.0 * (2**attempt))
                        if debug:
                            print(f"[ASPECT B SUBASPECT DEBUG] Rate limit, retrying in {wait_s:.1f}s...")
                        time.sleep(wait_s)
                        continue
                    if debug:
                        print(
                            "[ASPECT B SUBASPECT DEBUG] Call failed "
                            f"({type(exc).__name__}) for target={target}."
                        )
                    break

            if result is not None:
                answer = _norm_yes_no(result.answer)
                sentiment = _norm_sentiment(result.sentiment)
                codes = _norm_codes(result.codes)
                reason = (result.reason or "").strip()

                if answer == "No":
                    outputs.append(
                        {
                            "target_mention": target,
                            "answer": "No",
                            "sentiment": "None",
                            "codes": [],
                            "reason": reason,
                        }
                    )
                    break

                if answer == "Yes" and sentiment in {"Positive", "Negative", "Neutral"} and codes:
                    outputs.append(
                        {
                            "target_mention": target,
                            "answer": "Yes",
                            "sentiment": sentiment,
                            "codes": codes,
                            "reason": reason,
                        }
                    )
                    break

            if phase == 0:
                result = None
                user = (
                    user
                    + "\n\nValidation error: answer must be Yes or No only."
                    + "\nIf Yes, sentiment must be Positive, Negative, or Neutral and codes must include one or more of a,b,c,d,e,f,g."
                    + "\nIf No, sentiment must be None and codes must be an empty list."
                )
        else:
            outputs.append(
                {
                    "target_mention": target,
                    "answer": "No",
                    "sentiment": "None",
                    "codes": [],
                    "reason": "The model did not return a valid decision.",
                }
            )

    return outputs
