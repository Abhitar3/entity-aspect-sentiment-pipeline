from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from lc_llm import get_llm, is_rate_limit_error
from lc_json import safe_json_loads
from text_utils import preprocess_text, split_into_sentences

COREF_PROMPT_TEMPLATE = (
    "I will give you:\n"
    "1. One sentence\n"
    "2. The full post text as context\n"
    "3. A set of software entities\n"
    "4. One specific referring expression from the sentence\n"
    "\n"
    "The specific referring expression in the sentence will be marked with <REF>...</REF>.\n"
    "The referring expression may be one of:\n"
    "'it', 'its', 'they', 'both', 'this', 'these', or 'those'.\n"
    "\n"
    "Your task is to decide whether this specific referring expression refers to one or more of the provided software entities.\n"
    "Use the FULL post as context, including earlier sentences, to determine the most likely antecedent.\n"
    "\n"
    "IMPORTANT RULES\n"
    "- Only map the referring expression to entities if the referent is clearly one of the provided software entities.\n"
    "- The referred entity or entities must come only from the given set of software entities.\n"
    "- Do NOT infer, normalize, expand, or introduce software entities outside the given set.\n"
    "- If the expression refers to a task, problem, situation, event, clause, idea, or anything other than a software entity, return no mapping.\n"
    "- If the expression is ambiguous or unclear, return no mapping.\n"
    "- Prefer returning no mapping over returning an incorrect mapping.\n"
    "\n"
    "CARDINALITY RULES\n"
    "- For 'it', 'its', and 'this': return exactly one entity or return an empty list.\n"
    "- For 'they': return one or more entities, or return an empty list only if the reference is clearly grounded in the context.\n"
    "- For 'both': return exactly two entities or return an empty list.\n"
    "- For 'these' and 'those': return one or more entities, or an empty list only if the reference is clearly grounded in the context.\n"
    "\n"
    "Return ONLY valid JSON in this exact form:\n"
    "{\n"
    '  "entities": ["<entity1>", "<entity2>"]\n'
    "}\n"
)

REF_WORDS = ["it", "its", "it's", "itâ€™s", "they", "both", "this", "these", "those"]

GENERIC_DEMONSTRATIVE_NOUNS = {
    "api",
    "apis",
    "problem",
    "issue",
    "question",
    "approach",
    "method",
    "case",
    "scenario",
    "task",
    "thing",
    "stuff",
    "example",
    "situation",
    "context",
    "event",
    "solution",
    "tool",
    "tools",
    "library",
    "libraries",
    "framework",
    "frameworks",
}

DUMMY_IT_PATTERNS = [
    r"\bit\s+can\s+happen\b",
    r"\bit\s+happens\b",
    r"\bit\s+is\s+possible\b",
    r"\bit\s+is\s+important\b",
    r"\bit\s+turns\s+out\b",
    r"\bit\s+seems\b",
    r"\bit\s+looks\b",
    r"\bit\s+means\b",
]

def should_skip_ref(sentence: str, ref: str) -> bool:
    s = sentence.lower()

    if ref in {"it", "its", "it's", "itâ€™s", "they", "both"}:
        return False

    if ref == "this":
        for n in GENERIC_DEMONSTRATIVE_NOUNS:
            if re.search(rf"\b{ref}\s+{re.escape(n)}s?\b", s):
                return True

        if re.search(r"\bthis\s+is\s+(why|how|what)\b", s):
            return True

    if ref in {"these", "those"}:
        for n in GENERIC_DEMONSTRATIVE_NOUNS:
            if re.search(rf"\b{ref}\s+{re.escape(n)}s?\b", s):
                return True

    return False


def is_dummy_it(sentence: str) -> bool:
    s = sentence.lower()
    return any(re.search(pat, s) for pat in DUMMY_IT_PATTERNS)


@dataclass
class CorefRefResolution:
    ref: str
    start: int
    end: int
    occurrence_index: int
    entities: List[str]


@dataclass
class CorefSentenceResult:
    idx: int
    original: str
    refs_found: List[str]
    ref_map: Dict[str, List[str]]
    resolved_refs: List[CorefRefResolution]
    tagged: str
    explicit_mentions: List[str]


def _find_refs(sentence: str) -> List[str]:
    found: List[str] = []
    seen = set()
    for ref, _, _ in find_ref_occurrences(sentence):
        if ref in seen:
            continue
        seen.add(ref)
        found.append(ref)
    return found


def _entity_regex(entity: str) -> re.Pattern:
    e = re.escape(entity)
    return re.compile(rf"(?<!\w){e}(?!\w)")


def find_ref_occurrences(sentence: str) -> List[Tuple[str, int, int]]:
    hits: List[Tuple[str, int, int]] = []
    for ref in sorted(REF_WORDS, key=len, reverse=True):
        pattern = rf"\b{re.escape(ref)}\b"
        if ref == "it":
            pattern = rf"\bit\b(?!['’]\w)"
        for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
            hits.append((ref, match.start(), match.end()))
    hits.sort(key=lambda item: item[1])
    return hits


def _mark_ref_occurrence(sentence: str, start: int, end: int) -> str:
    return sentence[:start] + "<REF>" + sentence[start:end] + "</REF>" + sentence[end:]


def find_entity_mentions(sentence: str, entities: List[str]) -> List[Tuple[str, int, int]]:
    sorted_entities = sorted(entities, key=len, reverse=True)
    occupied: List[Tuple[int, int]] = []
    hits: List[Tuple[str, int, int]] = []

    def overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return not (a[1] <= b[0] or b[1] <= a[0])

    for ent in sorted_entities:
        pat = _entity_regex(ent)
        for m in pat.finditer(sentence):
            span = (m.start(), m.end())
            if any(overlaps(span, occ) for occ in occupied):
                continue
            occupied.append(span)
            hits.append((ent, m.start(), m.end()))

    hits.sort(key=lambda x: x[1])
    return hits


def wrap_entities_with_E(sentence: str, mentions: List[Tuple[str, int, int]]) -> str:
    if not mentions:
        return sentence
    out: List[str] = []
    last = 0
    for ent, start, end in mentions:
        out.append(sentence[last:start])
        out.append(f"<E>{ent}</E>")
        last = end
    out.append(sentence[last:])
    return "".join(out)


def _post_filter_entities(
    ref: str,
    resolved: List[str],
    explicit_mentions: List[str],
) -> List[str]:
    if not resolved:
        return []

    # Singular refs should map to at most one entity.
    if ref in {"it", "its", "this"} and len(resolved) != 1:
        return []

    # "both" should be exactly two entities.
    if ref == "both" and len(resolved) != 2:
        return []

    if ref in {"they", "these", "those"}:
        if len(resolved) < 1:
            return []

    if explicit_mentions:
        explicit_by_lower: Dict[str, List[str]] = {}
        for mention in explicit_mentions:
            explicit_by_lower.setdefault(mention.lower(), []).append(mention)

        aligned: List[str] = []
        for entity in resolved:
            candidates = explicit_by_lower.get(entity.lower(), [])
            if len(candidates) == 1:
                aligned.append(candidates[0])
            else:
                aligned.append(entity)
        return aligned

    return resolved


def _append_unique(target: List[str], source: List[str], limit: int) -> None:
    seen = set(target)
    for item in source:
        if item in seen:
            continue
        target.append(item)
        seen.add(item)
        if len(target) >= limit:
            return


def _build_candidate_entities(
    sent_idx: int,
    ref: str,
    sentence_explicit_mentions: List[List[str]],
    entities: List[str],
) -> List[str]:
    plural_refs = {"they", "both", "these", "those"}
    desired = 2 if ref in plural_refs else 1
    limit = len(entities) if ref in plural_refs else 3

    candidates: List[str] = []

    if ref in plural_refs:
        current_explicit = sentence_explicit_mentions[sent_idx]
        if len(current_explicit) >= 2:
            return current_explicit[:limit]

        for prev_idx in range(sent_idx - 1, -1, -1):
            prev_explicit = sentence_explicit_mentions[prev_idx]
            if len(prev_explicit) >= 2:
                return prev_explicit[:limit]

        for next_idx in range(sent_idx + 1, len(sentence_explicit_mentions)):
            next_explicit = sentence_explicit_mentions[next_idx]
            if len(next_explicit) >= 2:
                return next_explicit[:limit]

    _append_unique(candidates, sentence_explicit_mentions[sent_idx], limit)

    for prev_idx in range(sent_idx - 1, -1, -1):
        if len(candidates) >= limit:
            break
        _append_unique(candidates, sentence_explicit_mentions[prev_idx], limit)
        if len(candidates) >= desired:
            break

    if len(candidates) < desired:
        for next_idx in range(sent_idx + 1, len(sentence_explicit_mentions)):
            if len(candidates) >= limit:
                break
            _append_unique(candidates, sentence_explicit_mentions[next_idx], limit)
            if len(candidates) >= desired:
                break

    if candidates:
        return candidates

    return list(entities[:limit] if entities else [])


def _call_coref_llm(
    llm: BaseChatModel,
    ref: str,
    sentence: str,
    marked_sentence: str,
    post: str,
    entities: List[str],
) -> List[str]:
    system = (
        "You are a strict JSON generator for coreference resolution.\n"
        "Return ONLY valid JSON.\n"
        "Do not return markdown.\n"
        "Do not return explanations.\n"
        "Do not return any text outside the JSON object.\n"
        "Your JSON MUST contain exactly one key: entities.\n"
        "Rules:\n"
        "1) entities must be a JSON list of strings.\n"
        "2) Every returned string must be chosen exactly from the provided software entity set.\n"
        "3) Do NOT infer, normalize, expand, or invent entity names.\n"
        "4) Use the full post context, including earlier sentences, when deciding the antecedent.\n"
        "5) If the referring expression is ambiguous, unclear, generic, dummy, or does not clearly refer to valid software entity/entities, return {{\"entities\": []}}.\n"
        "6) If the referring expression refers to a task, problem, event, clause, idea, or non-entity concept, return {{\"entities\": []}}.\n"
        "7) Prefer empty output over an incorrect mapping.\n"
        "8) For 'it', 'its', and 'this', return exactly one entity or [].\n"
        "9) For 'they', return one or more entities, or [] only when clearly grounded in context.\n"
        "10) For 'both', return exactly two entities or [].\n"
        "11) For 'these' and 'those', return one or more entities, or [] only when clearly grounded in context.\n"
        "12) Never return entities outside the provided candidate entity set."
    )

    prompt = ChatPromptTemplate.from_messages([("system", system), ("user", "{user}")])

    user_text = (
        COREF_PROMPT_TEMPLATE
        + "\n\n"
        + f"Referring expression: '{ref}'\n"
        + f"Sentence: '{sentence}'\n"
        + f"Sentence with target occurrence marked: '{marked_sentence}'\n"
        + f"Full post text: '{post}'\n"
        + f"Software entities: {entities}\n"
    )

    chain = prompt | llm | StrOutputParser()
    raw = ""
    for attempt in range(3):
        try:
            raw = chain.invoke({"user": user_text})
            break
        except Exception as e:
            if is_rate_limit_error(e):
                if attempt == 2:
                    return []
                time.sleep(min(6.0, 1.0 * (2 ** attempt)))
                continue
            return []
    out = safe_json_loads(raw)

    ents = out.get("entities", [])
    if not isinstance(ents, list):
        return []

    exact_allowed = {entity: entity for entity in entities}
    lower_allowed: Dict[str, List[str]] = {}
    for entity in entities:
        lower_allowed.setdefault(entity.lower(), []).append(entity)
    cleaned: List[str] = []
    for x in ents:
        if isinstance(x, str) and x.strip():
            token = x.strip()
            if token in exact_allowed:
                cleaned.append(exact_allowed[token])
                continue

            candidates = lower_allowed.get(token.lower(), [])
            if len(candidates) == 1:
                cleaned.append(candidates[0])

    deduped: List[str] = []
    seen = set()
    for e in cleaned:
        if e in seen:
            continue
        seen.add(e)
        deduped.append(e)
    return deduped


def resolve_ref_occurrence_llm(
    post_text: str,
    sentence_id: int,
    ref: str,
    start: int,
    end: int,
    entities: List[str],
    llm: Optional[BaseChatModel] = None,
) -> List[str]:
    """
    Resolve one referring-expression occurrence after it has been selected by
    the aspect stage. sentence_id is 1-based and refers to the cleaned/split post.
    """
    llm = llm or get_llm(temperature=0.0)

    post_clean = preprocess_text(post_text)
    sents = split_into_sentences(post_clean)
    if sentence_id < 1 or sentence_id > len(sents):
        return []

    sentence_explicit_mentions: List[List[str]] = []
    for sent in sents:
        mentions = find_entity_mentions(sent, entities)
        explicit_mentions: List[str] = []
        seen_explicit = set()
        for ent, _, _ in mentions:
            if ent in seen_explicit:
                continue
            seen_explicit.add(ent)
            explicit_mentions.append(ent)
        sentence_explicit_mentions.append(explicit_mentions)

    sent_idx = sentence_id - 1
    sentence = sents[sent_idx]

    if should_skip_ref(sentence, ref):
        return []
    if ref == "it" and is_dummy_it(sentence):
        return []

    marked_sentence = _mark_ref_occurrence(sentence, start, end)
    candidate_entities = _build_candidate_entities(sent_idx, ref, sentence_explicit_mentions, entities)
    resolved = _call_coref_llm(
        llm=llm,
        ref=ref,
        sentence=sentence,
        marked_sentence=marked_sentence,
        post=post_clean,
        entities=candidate_entities,
    )
    return _post_filter_entities(ref, resolved, sentence_explicit_mentions[sent_idx])


def resolve_post_coreference_llm(
    post_text: str,
    entities: List[str],
    llm: Optional[BaseChatModel] = None,
    debug: bool = False,
) -> List[CorefSentenceResult]:
    llm = llm or get_llm(temperature=0.0)

    post_clean = preprocess_text(post_text)
    sents = split_into_sentences(post_clean)

    # Cache by (ref, marked sentence, candidate set) to avoid redundant calls in repeated content.
    llm_cache: Dict[Tuple[str, str, Tuple[str, ...]], List[str]] = {}

    sentence_explicit_mentions: List[List[str]] = []
    for sent in sents:
        mentions = find_entity_mentions(sent, entities)
        explicit_mentions: List[str] = []
        seen_explicit = set()
        for ent, _, _ in mentions:
            if ent in seen_explicit:
                continue
            seen_explicit.add(ent)
            explicit_mentions.append(ent)
        sentence_explicit_mentions.append(explicit_mentions)

    results: List[CorefSentenceResult] = []

    for i, sent in enumerate(sents, 1):
        refs = _find_refs(sent)

        mentions = find_entity_mentions(sent, entities)
        explicit_mentions = list(sentence_explicit_mentions[i - 1])

        ref_map: Dict[str, List[str]] = {}
        resolved_refs: List[CorefRefResolution] = []
        ref_occurrence_counts: Dict[str, int] = {}

        # If the sentence already contains explicit entity mentions,
        # do not perform same-sentence coreference resolution.
        # The explicit entity view is sufficient for sentence-level classification,
        # and same-sentence refs would only add duplicate mention-level outputs.
        if explicit_mentions and refs:
            for ref, start, end in find_ref_occurrences(sent):
                ref_occurrence_counts[ref] = ref_occurrence_counts.get(ref, 0) + 1
                ref_map[ref] = []
                resolved_refs.append(
                    CorefRefResolution(
                        ref=ref,
                        start=start,
                        end=end,
                        occurrence_index=ref_occurrence_counts[ref],
                        entities=[],
                    )
                )
            tagged = wrap_entities_with_E(sent, mentions)
        else:
            for ref, start, end in find_ref_occurrences(sent):
                ref_occurrence_counts[ref] = ref_occurrence_counts.get(ref, 0) + 1

                if should_skip_ref(sent, ref):
                    resolved_refs.append(
                        CorefRefResolution(
                            ref=ref,
                            start=start,
                            end=end,
                            occurrence_index=ref_occurrence_counts[ref],
                            entities=[],
                        )
                    )
                    continue

                if ref == "it" and is_dummy_it(sent):
                    resolved_refs.append(
                        CorefRefResolution(
                            ref=ref,
                            start=start,
                            end=end,
                            occurrence_index=ref_occurrence_counts[ref],
                            entities=[],
                        )
                    )
                    continue

                marked_sentence = _mark_ref_occurrence(sent, start, end)
                candidate_entities = _build_candidate_entities(i - 1, ref, sentence_explicit_mentions, entities)
                cache_key = (ref, marked_sentence.lower().strip(), tuple(candidate_entities))
                if cache_key in llm_cache:
                    resolved = llm_cache[cache_key]
                else:
                    resolved = _call_coref_llm(
                        llm=llm,
                        ref=ref,
                        sentence=sent,
                        marked_sentence=marked_sentence,
                        post=post_clean,
                        entities=candidate_entities,
                    )
                    llm_cache[cache_key] = resolved

                filtered = _post_filter_entities(ref, resolved, explicit_mentions)
                ref_map[ref] = filtered
                resolved_refs.append(
                    CorefRefResolution(
                        ref=ref,
                        start=start,
                        end=end,
                        occurrence_index=ref_occurrence_counts[ref],
                        entities=filtered,
                    )
                )

            tagged = wrap_entities_with_E(sent, mentions)

        if debug:
            print(f"\nSentence {i}:")
            print("  Original:", sent)
            print("  Refs found:", refs)
            print("  Ref map:", ref_map)
            print("  Tagged:", tagged)

        results.append(
            CorefSentenceResult(
                idx=i,
                original=sent,
                refs_found=refs,
                ref_map=ref_map,
                resolved_refs=resolved_refs,
                tagged=tagged,
                explicit_mentions=explicit_mentions,
            )
        )

    return results
