# Retrieval Eval Workbench

A local workbench for comparing a lexical TF-IDF baseline with real dense semantic retrieval, then producing a deterministic extractive answer with an inspectable line-level citation. It is deliberately narrow: the goal is to make retrieval and grounding behavior measurable, reproducible, and easy to inspect.


## What runs

```text
versioned corpus + hashes ──> lexical TF-IDF ───┐
                         └─> local MiniLM dense ─┼─> ranked documents
                                                   └─> extractive answer + citation
exposed development questions ───────────────────────> regression metrics and examples
```

The semantic path loads a pinned `sentence-transformers/all-MiniLM-L6-v2` revision on CPU and uses normalized dense-vector dot products. `data/model-manifest-v1.json` records the immutable revision, license, and SHA-256 values for the required snapshot files; a mismatched snapshot fails closed. The earlier receipt did not record that information and is explicitly marked as provenance-unknown. See the [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [license](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/LICENSE), and [Sentence Transformers semantic-search guidance](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html).

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/RoshanDewmina/retrieval-eval-workbench
cd retrieval-eval-workbench
uv sync --frozen
uv run pytest
make benchmark
make demo
```

Open `http://127.0.0.1:8113`. `HOST` and `PORT` can override the default; the service defaults to loopback and port 8113.

```bash
curl -s http://127.0.0.1:8113/health
curl -s -X POST http://127.0.0.1:8113/api/query \
  -H 'content-type: application/json' \
  -d '{"query":"Where do unmatched routing cases go?","strategy":"semantic","top_k":3}'
```

The initial semantic request can take longer because it downloads a roughly 91 MB model weight file. No paid API, hosted inference, or private data is used.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Returns the portfolio health contract: service, version, and status. |
| `POST /api/query` | Runs `lexical` or `semantic` retrieval and returns ranked documents, answer, and citations. |
| `GET /api/results` | Returns the immutable independent first-run receipt; reruns remain separate. |
| `GET /` | Small query UI for comparison and inspection. |

Example response:

```json
{
  "strategy": "semantic",
  "answerable": true,
  "answer": "If no routing rule matches, the case enters the General Review queue.",
  "citations": [{"document_id": "routing", "line_start": 5, "line_end": 5}],
  "hits": [{"document_id": "routing", "score": 0.73}]
}
```

## Dataset, evaluation, and regression protection

`data/corpus/` contains ten original, fictional Meridian Support Service documents. They are synthetic documentation, are licensed MIT with this repository, and have a SHA-256 recorded in `data/manifest.json`; the loader refuses a modified document. The `example.invalid` URLs are stable provenance identifiers, not external sources.

`data/question_sets/v1.0.0-original-exposed.json` preserves the original set and `data/questions.json` records its evolved v1.1.0 form. Both are exposed development/regression history, not independent held-out evaluation. `data/calibration-v1.json` freezes serving and evaluation defaults from that development history; the API returns the effective values and configuration fingerprint. An independent reviewer authored and froze twenty new cases after source `661852a`, then executed exactly one offline pass. The complete original runner, freeze, manifests and first-run receipt are preserved in `evidence/independent-v2/`; neither inference logic nor calibration changed after seeing those outcomes.

`make benchmark` re-runs that now-exposed independent set against the frozen first-run metric reference and writes a separate regression receipt to `evidence/benchmarks/frozen-run.json`. It reports Recall@3, MRR@3, citation integrity, labeled source support, critical-fact correctness, an explicitly named term-overlap proxy, unanswerable correctness, and every inspectable example. It fails closed for missing retrievers, metrics, and non-finite values; failed runs record their real nonzero exit status. `evidence/benchmarks/latest.json` is preserved historical development evidence only.

Grounding is intentionally deterministic and extractive: it returns one corpus sentence with an exact physical line. Evaluation verifies the citation's document ID, immutable source URL, physical range, exact quote, matching answer text, and a labeled supporting passage. Critical-fact checks require labeled phrases such as `eight hours`; the term-overlap value is retained as a separate, weaker proxy. This avoids relying on an uncalibrated model judge, but it does not measure fluent synthesis, comprehensive answers, or real-world factual accuracy.

## Tradeoffs and limits

- The corpus is small and original synthetic data. Results are a local regression signal, not a statistically conclusive or production claim.
- Dense retrieval is CPU-only and caches the corpus embeddings only in process. It is suitable for a local demo, not a large-corpus vector index.
- The unanswerable detector uses frozen development configuration. Out-of-domain queries can still retrieve superficially related material.
- The answerer quotes a single supporting sentence. It does not compose multi-document answers or establish semantic relevance beyond the versioned local corpus.
- The Docker image runs locally and downloads no model during build. Run a semantic query or benchmark after starting it to populate the runtime cache.

## Repository contents

- `src/`: FastAPI service, retrieval implementations, extractive answerer, and evaluator.
- `data/`: versioned corpus, exposed development lineage, frozen settings, labels, and model manifest.
- `evidence/`: benchmark receipts, regression baseline, and resume-safe claims.
- `tests/`: hash, retrieval, citation, semantic-vector, and API contract checks.
- `INTERVIEW_GUIDE.md`: design decisions, limitations, and exercises.

## Independent first-run outcomes

| Measure | Lexical | Semantic |
| --- | ---: | ---: |
| Any relevant document in top 3 | 14/14 | 14/14 |
| Exact valid physical citations | 14/14 | 14/14 |
| Labeled supporting passage | 12/14 | 13/14 |
| All required answer facts | 11/14 | 11/14 |
| Unsupported-question refusal | 1/6 | 2/6 |
| Complete multi-document answers | 0/2 | 0/2 |

These failures remain visible in the UI under **Inspect independent evaluation**;
expand any case to see the answer and citations. The field `recall_at_3` is an
any-hit rate, not fractional recall for questions requiring several documents.
Valid quotations can be irrelevant, and partial support is not completeness.
This twenty-case authored sample is not a population accuracy estimate.

`evidence/independent-v2/reference-baseline.json` freezes observed rates only for
future drop detection. It is not a retrospectively chosen success target. A
stored-metric mutation probe and tests verify that a critical-fact drop fails the
gate. `make benchmark` must never overwrite `first-run.json`; subsequent runs are
labeled exposed regression runs. The historical zero-floor baseline supports
coverage checks only and is no longer the default benchmark.

Linux installation explicitly pins Torch to the CPU wheel index in the frozen
lock; macOS retains its existing Torch version. The first-run receipt retains
its original lock hash and source revision. See [uv's PyTorch configuration](https://docs.astral.sh/uv/guides/integration/pytorch/).
