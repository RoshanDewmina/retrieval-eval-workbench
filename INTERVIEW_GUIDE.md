# Interview guide

## What I built

I built a local retrieval evaluation workbench around an original synthetic support-document corpus. It compares a transparent TF-IDF baseline with a real pretrained MiniLM embedding model, evaluates both on a fixed held-out question set, and exposes the result through a small FastAPI query UI.

## Decisions to defend

- I started with TF-IDF because it is cheap, deterministic, and gives a useful lexical baseline. A dense model only earns its complexity if the held-out results or inspection justify it.
- The semantic path uses `all-MiniLM-L6-v2` on CPU, producing normalized embeddings and ranking them with cosine-equivalent dot products. It does not call a paid model or an external inference API.
- I kept answer generation extractive. The answer is a sentence from retrieved corpus text, and the UI returns an exact line-level citation. That makes grounding failures inspectable without claiming an LLM judge proves factuality.
- Question IDs enforce a tuning/test separation. The unanswerable threshold is selected from tuning data; the test split is held out for the reported metrics.
- I hash each corpus document in the manifest to make an evaluation run traceable to exact inputs.

## Failures and limitations to discuss

- A semantic retriever can lose to TF-IDF on jargon-heavy or exact identifier queries. The benchmark reports both results rather than assuming dense retrieval wins.
- A fixed similarity threshold is fragile on a new corpus. It is evaluated on four unanswerable examples but has not been calibrated for production traffic.
- One-line extractive answers do not handle questions needing several documents. Adding generation would require a separate answer-support evaluation and a stronger citation verifier.
- The small synthetic dataset is intended to demonstrate evaluation mechanics. It does not establish performance on customer records or a real documentation estate.

## Reproduce and extend

```bash
uv sync --frozen
uv run pytest
make benchmark
make demo
```

Useful exercises:

1. Add a new corpus document, compute and record its SHA-256, then add held-out and tuning questions without mixing their IDs.
2. Add query/document-specific encoder methods and compare them against the current generic encoder while preserving the baseline.
3. Add multi-document answer extraction and define a new citation-support metric before reporting it.
4. Intentionally remove a corpus document or raise a baseline floor to observe the regression gate fail with a named metric.
