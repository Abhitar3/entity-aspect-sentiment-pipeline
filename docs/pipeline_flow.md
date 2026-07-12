# Pipeline Flow

The current pipeline runs at the full-post level first, then creates sentence-level outputs.

## Flow

1. **Input**
   - Input is one post or a list of posts.
   - Supported JSON forms include `{"post": "..."}`, `{"posts": ["..."]}`, or a list.

2. **Preprocessing**
   - HTML is stripped.
   - Structural text is preserved where possible.
   - Whitespace and obvious noisy fragments are cleaned.

3. **Sentence Splitting**
   - The cleaned post is split into sentence-level units using the project sentence splitter.

4. **Post-Level Entity Extraction**
   - An LLM extracts explicitly mentioned software entities from the full post.
   - The output is filtered to remove generic terms, programming languages, and entities not exactly grounded in the post.

5. **Mention Detection**
   - Explicit entity mentions are located in each sentence.
   - Referring expressions such as `it`, `its`, `it's`, `they`, `both`, `this`, `these`, and `those` are also detected.

6. **Tagged Classification Views**
   - Each explicit mention or referring expression is wrapped with `<E>...</E>`.
   - Example: `Cypress is useful` becomes `<E>Cypress</E> is useful`.
   - Example: `It is easy to use` becomes `<E>It</E> is easy to use`.

7. **Aspect and Sentiment Classification**
   - Each tagged view is sent to the aspect classifier.
   - The current active output includes Aspect A and Aspect B.
   - Sentiment is assigned only when the aspect is present.

8. **Aspect-Positive Coreference**
   - Referring expressions are not always resolved immediately.
   - Coreference is called only when a referring expression is aspect-positive.
   - The coreference step maps the referring expression to one or more selected software entities from the post entity set.

9. **Final Triple Assembly**
   - Explicit mentions create direct triples.
   - Resolved referring expressions create triples for the resolved software entity/entities.
   - Unresolved referring expressions are stored in debug reports, not emitted as final entity triples.

10. **Deduplication and Output**
    - Duplicate triples are removed.
    - Final output contains sentence-level triples with `post_id`, `sentence_id`, `entity`, `aspect`, and `sentiment`.
    - Optional debug reports preserve intermediate decisions.

## LLM Call Locations

- Entity extraction: full-post software entity extraction.
- Aspect classification: Aspect A/B and sentiment classification for each tagged mention view.
- Coreference resolution: only for referring expressions that are aspect-positive.

## Final Output Shape

```json
{
  "post_id": "1",
  "sentence_id": "2",
  "entity": "Cypress",
  "aspect": "Aspect B Positive",
  "sentiment": "Positive"
}
```
