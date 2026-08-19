# Technical Overview — CPU LLM Lab v0.3

## 1. Problem statement

CPU LLM Lab studies a practical systems question:

**How small and inexpensive can a local language-model pipeline be while still answering a constrained corporate QA task with acceptable fidelity?**

Version 0.3 expands the independent variable. The lab now compares both the generation model and the retrieval strategy.

## 2. System architecture

```text
User question
    ↓
Retrieval strategy
    ├── lexical
    ├── embedding
    └── hybrid
    ↓
Top-k context
    ↓
Prompt construction
    ↓
Ollama / selected generation model
    ↓
Structured answer + cited sources
    ↓
Deterministic evaluation
    ↓
Retrieval + quality + runtime benchmark
```

The model never receives direct SQL access.

## 3. Retrieval strategies

### Lexical

The original baseline counts exact token overlap and applies field weights:

```text
title × 5
category × 3
content × 1
```

The weights are explicit heuristics.

### Embedding

Documents and queries are embedded with Ollama `/api/embed`. The same embedding model is used for both sides. Documents are cached in memory and ranked with cosine similarity.

No vector database is used in v0.3 because the corpus is intentionally tiny.

### Hybrid

Lexical scores are normalized by the maximum lexical score for the query. Cosine similarity is mapped from `[-1, 1]` to `[0, 1]`. The two signals are then linearly combined.

Default:

```text
embedding weight = 0.5
lexical weight   = 0.5
```

This is a baseline parameter and should eventually be calibrated experimentally.

## 4. Retrieval evaluation

For answerable cases, the benchmark computes:

- Recall@1
- Recall@3
- Recall@5
- Mean Reciprocal Rank (MRR)

The existing `retrieval_hit` remains useful because it evaluates whether all expected documents are present in the exact Top-K context passed to the LLM.

This distinction matters:

```text
Recall@5
→ quality of the ranking

retrieval_hit with top_k=3
→ quality of the actual context sent to generation
```

## 5. Generation evaluation

The evaluator still measures:

- required facts preserved;
- expected source citation;
- known forbidden facts;
- answer/abstention decision.

These metrics remain deterministic proxies rather than a semantic proof of faithfulness.

## 6. Experimental control

For a given case and retrieval mode, retrieval is executed once and the same retrieved context is sent to every selected generation model.

This preserves an important comparison property:

```text
same question
same retriever
same Top-K context
same prompt policy
different LLM
```

When comparing retrieval modes, the retriever is intentionally the changing variable.

## 7. Semantic paraphrase cases

The data set includes questions intentionally written with vocabulary that may not overlap strongly with the source documents.

Their purpose is to challenge the assumption that lexical overlap is sufficient.

## 8. Observability

`observability.py` adds a lightweight tracing layer.

Each request gets a `trace_id`, and nested operations emit JSON logs with operation name, duration and non-sensitive metadata.

Main spans:

```text
api.query
retrieval.search
ollama.embed
ollama.chat
evaluation.query
benchmark.run
benchmark.case
```

Document bodies and full prompts are not logged.

If the OpenTelemetry API exists in the environment, the same operations are also represented as spans through the global tracer. Export/backends remain deployment concerns rather than mandatory v0.3 dependencies.

## 9. Embedding cache

`Retriever` maintains an in-memory cache keyed by:

```text
embedding model
document id
SHA-256 of the embedded document text
```

If the document changes, its hash changes and a new vector is generated.

This keeps repeated benchmark runs cheaper while preserving correctness after content edits.

## 10. Runtime metrics

Generation metrics include:

- total inference duration;
- model load duration;
- prompt tokens;
- output tokens;
- tokens/s;
- Python process RSS;
- VRAM status.

Retrieval metrics include:

- total retrieval latency;
- embedding generation latency;
- ranked document IDs and scores.

## 11. CPU verification improvement

Previously, failure to read `/api/ps` could be interpreted as zero VRAM.

v0.3 distinguishes:

```text
0      → VRAM value was observed as zero
> 0    → VRAM was used
None   → the project could not verify it
```

This avoids a false CPU-only claim.

## 12. Current experimental limitations

- small corpus;
- no chunking;
- no persistent vector index;
- heuristic hybrid weight;
- substring-based factual evaluator;
- no repeated inference runs;
- no p50/p95;
- no persistent observability backend;
- source accuracy still checks whether at least one expected source was cited.

## 13. Next scientific questions

The v0.3 benchmark can answer:

1. Does semantic retrieval improve ranking on paraphrases?
2. Does higher retrieval recall translate into higher answer factuality?
3. What latency overhead does embedding introduce?
4. Does hybrid retrieval dominate either individual retriever?
5. Which LLM remains acceptable once the retrieval quality is improved?

Future versions can then improve factual evaluation, repeated measurements, hardware telemetry and semantic routing.
