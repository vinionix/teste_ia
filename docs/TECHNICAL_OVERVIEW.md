# Technical Overview — CPU LLM Lab

## 1. Problem statement

CPU LLM Lab studies a practical engineering question: **how small can a local language model be while still answering a constrained corporate question-answering task with acceptable fidelity?**

The project intentionally targets CPU-only execution and evaluates models not by popularity or parameter count, but by their behavior on a repeatable task grounded in a controlled document base.

## 2. System architecture

```text
User question
    ↓
Lexical retrieval over SQLite documents
    ↓
Top-k relevant context
    ↓
Prompt construction
    ↓
Ollama / selected local model
    ↓
Structured answer + cited sources
    ↓
Deterministic benchmark
```

The model never receives direct SQL access. Retrieval happens before inference and only the selected context is sent to the model. This keeps the experiment focused on grounded answering instead of text-to-SQL behavior.

## 3. Evaluation dimensions

The benchmark records multiple dimensions because a single score can hide important trade-offs:

- required-fact preservation;
- expected document retrieval;
- expected source citation;
- known factual mutation checks;
- refusal when the source base does not contain the answer;
- input/output token counts;
- latency;
- tokens per second;
- model identity and execution environment.

The factuality checks are deterministic proxies defined by the test cases. They do not claim to prove that every generated sentence is fully entailed by the source.

## 4. Why CPU-only matters

The project is designed around environments where GPU inference is unavailable or too expensive. That changes the optimization target: a smaller model with reliable task behavior can be more useful than a larger model with better general capability but unacceptable latency or infrastructure cost.

This makes the benchmark a study of **quality × cost × latency**, not only raw model quality.

## 5. Data model

The local data set represents a fictional company and contains policy-like documents covering areas such as benefits, remote work, education, travel and information security.

At startup, JSON source documents are synchronized into SQLite. Retrieval operates over this controlled base so benchmark expectations remain reproducible.

## 6. Negative cases and hallucination pressure

The test suite includes questions whose answers are absent from the document base. These cases are important because a grounded assistant must know when to refuse rather than filling gaps with external knowledge or invented details.

A good model for this experiment therefore needs both:

- useful extraction when evidence exists;
- disciplined abstention when evidence does not exist.

## 7. Experimental workflow

A normal comparison cycle is:

1. install one or more local models in Ollama;
2. start the application;
3. run the same benchmark cases for each model;
4. collect fidelity and performance metrics;
5. compare failure patterns rather than only aggregate scores;
6. identify the smallest model that remains acceptable for the target task.

## 8. Engineering boundaries

This repository is a lab, not a production RAG platform. Current design choices deliberately simplify some areas so model behavior is easier to study.

Examples of boundaries:

- lexical retrieval instead of a full embedding/vector pipeline;
- deterministic benchmark checks instead of an LLM judge;
- a fictional local data set;
- local Ollama execution;
- task-specific evaluation rather than broad model benchmarking.

These constraints are features of the experiment: they make the system cheap, reproducible and easier to reason about.

## 9. Future experiments

Useful extensions include:

- compare lexical retrieval with embeddings or hybrid retrieval;
- measure context-size sensitivity;
- test quantization variants of the same model family;
- record memory and CPU utilization;
- add semantic or model-assisted evaluation while retaining deterministic checks;
- test prompt variants under the same cases;
- add adversarial prompts and prompt-injection cases;
- export benchmark runs for longitudinal comparison.

## 10. Portfolio value

This project demonstrates practical LLM engineering beyond prompt demos:

- evaluation design;
- grounded generation;
- local inference;
- cost/performance trade-offs;
- explicit negative cases;
- structured outputs;
- API design;
- reproducible experimentation.
