# Evaluation Summary

The pipeline has been evaluated against human annotations across multiple components.

## Entity Extraction

Entity extraction was evaluated on 30 full posts.

- Correctly identified entities: 105
- False negatives: 5
- False positives: 12
- Summary: strong post-level entity coverage, with false negatives reviewed as the higher-priority error type.

## Aspect Classification

Aspect classification was evaluated across two manually reviewed sets of approximately 100 sentences each.

Set 1:

- Aspect A Negative: 50/53 = 94%
- Aspect B Negative: 36/44 = 82%
- Aspect A Positive: 35/45 = 78%
- Aspect B Positive: 48/54 = 89%

Set 2:

- Aspect A Positive: 61/66 = 92%
- Aspect A Negative: 28/29 = 96%
- Aspect B Positive: 36/42 = 85%
- Aspect B Negative: 48/52 = 92%

Summary: aspect classification showed strong agreement overall, with the most useful error-analysis work focused on boundary cases between performance, usability, functionality, and referring-expression context.

## Coreference Resolution

Coreference was evaluated on 35 posts.

- Total referring expressions detected: 180
- Aspect-positive referring expressions: 99
- Resolved aspect-positive refs: 42
- Unresolved aspect-positive refs: 57
- Correctly resolved: 37/42
- Correctly unresolved: 52/57
- Decision-level correctness: 89/99 = 89.9%

Summary: the current coreference design prioritizes aspect-relevant referring expressions and avoids emitting unresolved pronouns as final entities. Debug reports preserve unresolved cases for audit and manual review.

## Resume-Ready Summary

Benchmarked LLM pipeline components against human annotations across entity extraction, aspect classification, sentiment labeling, and coreference resolution, achieving strong entity coverage, 78-96% aspect classification agreement across two evaluation sets, and 89.9% coreference decision accuracy on aspect-relevant referring expressions.
