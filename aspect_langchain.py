from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

from entity_focus import extract_single_entity
from lc_llm import get_llm, is_rate_limit_error

USE_ASPECT_VALIDATION = False

ASPECT_A_DEFINITION = (
    "Aspect A of entity E is about the amount of resources used by E. These include: "
    "Time behaviour: Entity E meets requirements of response, processing times, and throughput rates when performing its functions; "
    "Resource utilization: Entity E meets requirements of amounts and types of resources used when performing its functions; and "
    "Capacity: The maximum limits of an Entity E parameter meet requirements."
)

ASPECT_B_DEFINITION = (
    "Aspect B of entity E could "
    "(a) refer to the level of ease with which developers can interact with the Entity E and use it to accomplish their tasks; "
    "(b) refer to the extent to which an Entity E can be utilized by intended users in a particular usage scenario with effectiveness, efficiency, and satisfaction; "
    "(c) learnability: Entity E enables users to achieve specified goals of learning that in turn helps in using that API with effectiveness, efficiency, freedom from risk, and satisfaction; "
    "(d) operability: Entity E has attributes that make it easy to operate and control; "
    "(e) user error protection: Entity E protects users against making errors; "
    "(f) user interface aesthetics: Entity E enables pleasing and satisfying interaction for the user; "
    "(g) accessibility: Entity E can be used by people with the widest range of characteristics and capabilities; "
    "(h) installability: Entity E can be successfully installed and/or uninstalled in a specified environment."
)

ASPECT_C_DEFINITION = (
    "Aspect C of entity E is about whether entity E provides written or example-based resources "
    "that help users understand and use entity E, or whether the lack or poor quality of such resources "
    "makes entity E difficult to understand or use."
)

ASPECT_A_PROMPT = (
    'I will give you a set of sentences or sentence. Each of these sentences will contain a target mention '
    'represented between <E> and </E>. The target mention may be a software entity name, API, library, '
    'framework, or a referring expression such as it, its, they, both, this, these, or those. '
    'We would like to know if this sentence is discussing Aspect A of the target mention E, or the entity referred to by E. '
    'Aspect A of the target mention E is about the amount of resources used by E or the entity referred to by E. These include: '
    'Time behaviour: E meets requirements of response, processing times, and throughput rates when performing its functions; '
    'Resource utilization: E meets requirements of amounts and types of resources used when performing its functions; and '
    'Capacity: The maximum limits of an E parameter meet requirements. '
    'For each sentence provided: '
    '1) First answer only "Yes" or "No" to whether the sentence discusses Aspect A of the target mention E, or the entity referred to by E, in that sentence. '
    '2) If the answer is "Yes", then indicate the sentiment: '
    '   - Positive: if the sentence praises or positively describes the Aspect A of E '
    '   - Negative: if the sentence criticizes or negatively describes the Aspect A of E '
    '   - Neutral: if the sentence is neither praising nor criticizing the Aspect A (e.g., stating facts, observations, or neutral descriptions) '
    '3) If the answer is "No", sentiment must be "None". '
    'Please provide your output in JSON format. Do not provide any explanation or additional text. '
    'Sentences:\n'
)

# Previous Aspect B prompt kept for comparison while testing the revised prompt.
ASPECT_B_PROMPT = (
    'I will give you a set of sentences or sentence. Each of these sentences will contain a target mention '
    'represented between <E> and </E>. The target mention may be a software entity name, API, library, '
    'framework, or a referring expression such as it, its, they, both, this, these, or those. '
    'We would like to know if this sentence is discussing Aspect B of the target mention E, or the entity referred to by E. '
    'Aspect B of the target mention E could '
    '(a) refer to the level of ease with which developers can interact with E and use it to accomplish their tasks; '
    '(b) refer to the extent to which E can be utilized by intended users in a particular usage scenario with effectiveness, efficiency, and satisfaction; '
    '(c) learnability: E enables users to achieve specified goals of learning that in turn helps in using that API with effectiveness, efficiency, freedom from risk, and satisfaction; '
    '(d) operability: E has attributes that make it easy to operate and control; '
    '(e) user error protection: E protects users against making errors; '
    '(f) user interface aesthetics: E enables pleasing and satisfying interaction for the user; '
    '(g) accessibility: E can be used by people with the widest range of characteristics and capabilities; '
    '(h) installability: E can be successfully installed and/or uninstalled in a specified environment. '
    'For each sentence provided: '
    '1) First answer only "Yes" or "No" to whether the sentence discusses Aspect B of the target mention E, or the entity referred to by E, in that sentence. '
    '2) If the answer is "Yes", then indicate the sentiment: '
    '   - Positive: if the sentence praises or positively describes the Aspect B of E '
    '   - Negative: if the sentence criticizes or negatively describes the Aspect B of E '
    '   - Neutral: if the sentence is neither praising nor criticizing the  Aspect B  (e.g., stating facts, observations, or neutral descriptions) '
    '3) If the answer is "No", sentiment must be "None". '
    'Please provide your output in JSON format. Do not provide any explanation or additional text. '
    'Sentences:\n'
)


#New Aspect B updated prompt considering the functionality 
# ASPECT_B_PROMPT = (
#     'I will give you a set of sentences or sentence. Each of these sentences will contain an entity '
#     'which could be an API, library, or framework that is used in software development. '
#     'This entity is represented between <E> and </E> in these sentences. '
#     'We would like to know if this sentence is discussing Aspect B of the entity E. '
#     'Aspect B of entity E could '
#     '(a) refer to the level of ease with which developers can interact with the Entity E and use it to accomplish their tasks; '
#     '(b) refer to the extent to which an Entity E can be utilized by intended users in a particular usage scenario with effectiveness, efficiency, and satisfaction; '
#     '(c) learnability: Entity E enables users to achieve specified goals of learning that in turn helps in using that API with effectiveness, efficiency, freedom from risk, and satisfaction; '
#     '(d) operability: Entity E has attributes that make it easy to operate and control; '
#     '(e) user error protection: Entity E protects users against making errors; '
#     '(f) user interface aesthetics: Entity E enables pleasing and satisfying interaction for the user; '
#     '(g) accessibility: Entity E can be used by people with the widest range of characteristics and capabilities; '
#     '(h) installability: Entity E can be successfully installed and/or uninstalled in a specified environment. '
#     'Important boundary: Aspect B is about the developer/user experience of interacting with, learning, configuring, installing, controlling, or using the entity E. '
#     'Do not answer "Yes" for Aspect B if the sentence only describes what features, capabilities, architecture, integrations, security mechanisms, supported tasks, or runtime behavior the entity has. '
#     'However, answer "Yes" for Aspect B when a feature or capability is described as reducing developer effort, simplifying implementation, providing a useful abstraction, making configuration easier, making integration easier, making operation/control easier, improving convenience, or making interaction with the entity easier or harder. '
#     'Functionality-only statements are not Aspect B; developer-effort or interaction-focused statements are Aspect B. '
#     'Do not answer "Yes" for Aspect B if the sentence is only about speed, response time, memory, CPU/GPU usage, scalability, throughput, failover time, latency, or resource usage; those belong to Aspect A unless they are explicitly tied to ease or difficulty of use. '
#     'Do not answer "Yes" for Aspect B if the sentence is only about written resources, examples, guides, API references, README files, setup notes, tutorials, or project pages; those belong to Aspect C unless the sentence explicitly discusses the broader ease or difficulty of using the entity. '
#     'For each sentence provided: '
#     '1) First answer only "Yes" or "No" to whether the sentence discusses Aspect B of the entity E mentioned in that sentence. '
#     '2) If the answer is "Yes", then indicate the sentiment: '
#     '   - Positive: if the sentence says the entity is easy, simple, intuitive, convenient, useful, helpful, pluggable, controllable, easy to configure, easy to install, easy to operate, or helps users avoid errors '
#     '   - Negative: if the sentence says the entity is hard, difficult, confusing, inconvenient, frustrating, painful, error-prone, hard to configure, hard to install, hard to operate, or difficult to control '
#     '   - Neutral: if the sentence is neither praising nor criticizing Aspect B, but still clearly discusses developer/user interaction, learnability, operability, error protection, interface aesthetics, accessibility, or installability '
#     '3) If the answer is "No", sentiment must be "None". '
#     'Please provide your output in JSON format. Do not provide any explanation or additional text. '
#     'Sentences:\n'
# )

ASPECT_C_PROMPT = (
    'I will give you a set of sentences or sentence. Each of these sentences will contain an entity '
    'which could be an API, library, or framework that is used in software development. '
    'This entity is represented between <E> and </E> in these sentences. '
    'We would like to know if this sentence is discussing Aspect C of the entity E. '
    'Aspect C of entity E is about whether entity E provides written or example-based resources '
    'that help users understand, learn, install, configure, use, troubleshoot, or integrate entity E, '
    'or whether the lack, poor quality, incompleteness, incorrectness, outdatedness, unclear wording, '
    'or difficulty of finding such resources makes entity E harder to understand or use. '
    'These resources may include guides, manuals, tutorials, examples, API references, official pages, '
    'README files, sample code, usage notes, setup instructions, migration notes, or troubleshooting pages. '
    'For each sentence provided: '
    '1) First answer only "Yes" or "No" to whether the sentence discusses Aspect C of the entity E mentioned in that sentence. '
    '2) If the answer is "Yes", then indicate the sentiment: '
    '   - Positive: if the sentence praises or positively describes these resources for the entity '
    '   - Negative: if the sentence criticizes or negatively describes these resources for the entity '
    '   - Neutral: if the sentence is neither praising nor criticizing these resources, such as simply mentioning guides, examples, references, or setup notes '
    '3) If the answer is "No", sentiment must be "None". '
    'Please provide your output in JSON format. Do not provide any explanation or additional text. '
    'Sentences:\n'
)


class AspectDecision(BaseModel):
    answer: str
    evidence: str = ""
    sentiment: str = "None"


class VerificationDecision(BaseModel):
    valid: str
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


def _call_aspect_once(
    prompt_text: str,
    sentence: str,
    entity: str,
    debug: bool = False,
) -> Optional[Tuple[str, str, str]]:
    llm = get_llm(temperature=0.0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict JSON generator.\n"
                "Return ONLY schema-valid JSON.\n"
                "Task constraints:\n"
                "1) The target mention is ONLY the text wrapped in <E>...</E>.\n"
                "2) The target mention may be a software entity or a referring expression.\n"
                "3) Ignore other entities or mentions in the line.\n"
                "4) If evidence is ambiguous or mainly about another entity or mention, answer No.\n"
                "5) If answer is Yes, evidence must be a short direct quote from the same line.\n"
                "6) If answer is No, evidence must be empty string.\n"
                "7) If answer is Yes, sentiment must be one of: Positive, Negative, Neutral.\n"
                "8) If answer is No, sentiment must be None.",
            ),
            ("user", "{user}"),
        ]
    )
    chain = prompt | llm.with_structured_output(AspectDecision)
    user = (
        prompt_text
        + sentence
        + "\n\nTarget mention in this line: "
        + entity
    )

    for phase in range(2):
        for attempt in range(3):
            try:
                result = chain.invoke({"user": user})
                ans = _norm_yes_no(result.answer)
                sentiment = _norm_sentiment(result.sentiment)
                if ans is not None and sentiment is not None:
                    evidence = (result.evidence or "").strip()
                    if ans == "No":
                        evidence = ""
                        sentiment = "None"
                        return (ans, evidence, sentiment)

                    if sentiment in {"Positive", "Negative", "Neutral"}:
                        return (ans, evidence, sentiment)
            except Exception as e:
                if is_rate_limit_error(e):
                    if attempt == 2:
                        break
                    wait_s = min(6.0, 1.0 * (2**attempt))
                    if debug:
                        print(f"[ASPECT DEBUG] Rate limit, retrying in {wait_s:.1f}s...")
                    time.sleep(wait_s)
                    continue
                if debug:
                    print(
                        f"[ASPECT DEBUG] Aspect call failed ({type(e).__name__}) "
                        f"for entity={entity}."
                    )
                break

        # Repair pass.
        if phase == 0:
            user = (
                user
                + "\n\nValidation error: answer must be Yes or No only."
                + "\nIf Yes include evidence quote from this same line and sentiment in {Positive, Negative, Neutral}."
                + "\nIf No then evidence must be empty and sentiment must be None."
            )

    return None


def _verify_yes_label(
    sentence: str,
    entity: str,
    aspect_name: str,
    aspect_definition: str,
    evidence: str,
    debug: bool = False,
) -> bool:
    llm = get_llm(temperature=0.0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict JSON verifier.\n"
                "Return ONLY schema-valid JSON.\n"
                "Use the provided aspect definition, full line, target entity, and evidence quote.\n"
                "Set valid=Yes only if the evidence supports the aspect definition for the target entity in this line.\n"
                "If evidence is weak, off-target, or about another entity, return valid=No.",
            ),
            ("user", "{user}"),
        ]
    )
    chain = prompt | llm.with_structured_output(VerificationDecision)
    user = (
        f"Aspect: {aspect_name}\n"
        f"Aspect definition: {aspect_definition}\n"
        f"Line: {sentence}\n"
        f"Target entity: {entity}\n"
        f"Evidence quote: {evidence}\n"
        "Does this evidence justify a Yes label for this aspect definition and target entity only?"
    )

    for attempt in range(3):
        try:
            out = chain.invoke({"user": user})
            return _norm_yes_no(out.valid) == "Yes"
        except Exception as e:
            if is_rate_limit_error(e):
                if attempt == 2:
                    return False
                wait_s = min(6.0, 1.0 * (2**attempt))
                if debug:
                    print(f"[ASPECT DEBUG] Verify rate limit, retrying in {wait_s:.1f}s...")
                time.sleep(wait_s)
                continue
            if debug:
                print(f"[ASPECT DEBUG] Verify call failed ({type(e).__name__}).")
            return False
    return False


def _resolve_aspect(
    prompt_text: str,
    sentence: str,
    entity: str,
    aspect_name: str,
    aspect_definition: str,
    debug: bool = False,
) -> Tuple[str, str]:
    """
    Single-pass aspect resolution for GPT-4.1.
    Aspect validation can be enabled to verify positive labels with a second LLM call.
    """
    first = _call_aspect_once(prompt_text, sentence, entity, debug=debug)
    if first is None:
        if debug:
            print(f"[ASPECT DEBUG] {aspect_name} single-pass failed (entity={entity}).")
        return ("No", "None")

    ans, evidence, sentiment = first
    if ans == "No":
        return ("No", "None")

    if not USE_ASPECT_VALIDATION:
        return ("Yes", sentiment)

    if evidence and _verify_yes_label(
        sentence,
        entity,
        aspect_name,
        aspect_definition,
        evidence,
        debug=debug,
    ):
        return ("Yes", sentiment)

    return ("No", "None")


def classify_aspects_labels_llm(
    focused_sentences: List[str],
    debug: bool = False,
) -> List[Dict[str, str]]:
    if not focused_sentences:
        return []

    out: List[Dict[str, str]] = []
    for sentence in focused_sentences:
        ent = extract_single_entity(sentence) or ""
        if not ent:
            out.append(
                {
                    "entity": "",
                    "A": "No",
                    "A_sentiment": "None",
                    "B": "No",
                    "B_sentiment": "None",
                    # "C": "No",
                    # "C_sentiment": "None",
                }
            )
            continue

        a, a_sentiment = _resolve_aspect(
            prompt_text=ASPECT_A_PROMPT,
            sentence=sentence,
            entity=ent,
            aspect_name="Aspect A",
            aspect_definition=ASPECT_A_DEFINITION,
            debug=debug,
        )
        b, b_sentiment = _resolve_aspect(
            prompt_text=ASPECT_B_PROMPT,
            sentence=sentence,
            entity=ent,
            aspect_name="Aspect B",
            aspect_definition=ASPECT_B_DEFINITION,
            debug=debug,
        )
        # c, c_sentiment = _resolve_aspect(
        #     prompt_text=ASPECT_C_PROMPT,
        #     sentence=sentence,
        #     entity=ent,
        #     aspect_name="Aspect C",
        #     aspect_definition=ASPECT_C_DEFINITION,
        #     debug=debug,
        # )

        out.append(
            {
                "entity": ent,
                "A": a,
                "A_sentiment": a_sentiment,
                "B": b,
                "B_sentiment": b_sentiment,
                # "C": c,
                # "C_sentiment": c_sentiment,
            }
        )
    return out
