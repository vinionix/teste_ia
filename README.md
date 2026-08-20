# CPU LLM Lab

Laboratório local para comparar **retrieval + LLMs pequenos em CPU** em consultas corporativas fundamentadas em documentos.

A pergunta central continua sendo:

> Qual é a configuração mais barata que mantém qualidade suficiente para uma tarefa específica sem inventar ou alterar informações da base?

A versão **0.3.1** mantém o Retrieval Lab da v0.3 e melhora o rigor experimental: controla o cache dos embeddings, repete cada combinação, separa cold/warm retrieval e passa a reportar mediana e p95.

## v0.3 — Retrieval Lab

O projeto oferece três retrievers:

- `lexical`: baseline original baseada em palavras e pesos `5/3/1`;
- `embedding`: busca semântica com embeddings gerados pelo Ollama;
- `hybrid`: combinação normalizada do score lexical com similaridade semântica.

A baseline lexical foi mantida de propósito. O objetivo é medir se a complexidade extra do embedding realmente melhora o retrieval.

## v0.3.1 — Rigor experimental

A v0.3 revelou um viés importante: se `embedding` roda antes de `hybrid`, o segundo modo pode herdar os embeddings dos documentos já presentes em cache e parecer artificialmente mais rápido.

A v0.3.1 controla isso da seguinte forma:

- cada caso/retriever possui uma primeira repetição `cold`;
- antes do `cold` de `embedding` ou `hybrid`, o cache de embeddings dos documentos é limpo;
- as repetições seguintes são `warm` e reutilizam o cache;
- retrievers e modelos são embaralhados com uma seed reproduzível para reduzir viés de ordem;
- cada combinação pode ser repetida de 1 a 10 vezes;
- o benchmark reporta média, mediana e p95 de latência;
- os detalhes registram `benchmark_case_id` e número da repetição.

Isso não torna o benchmark estatisticamente completo, mas remove o principal viés de cache observado na v0.3.

## Arquitetura

```text
Pergunta
  ↓
Retriever escolhido
  ├── lexical
  ├── embedding
  └── hybrid
  ↓
Top-K documentos
  ↓
contexto controlado
  ↓
Ollama + modelo de geração
  ↓
resposta estruturada + fontes
  ↓
avaliação
  ↓
benchmark de qualidade + retrieval + desempenho
```

O modelo de geração **não recebe acesso direto ao SQL**. A aplicação recupera os documentos primeiro e entrega apenas o contexto selecionado ao LLM.

## Retrieval lexical

A baseline continua simples e interpretável:

```text
match no título     × 5
match na categoria  × 3
match no conteúdo   × 1
```

Esses pesos são uma heurística do laboratório, não valores derivados de um paper. Eles existem para servir como baseline controlada.

## Retrieval por embedding

O retriever semântico usa `POST /api/embed` do Ollama.

Por padrão:

```text
embedding_model = embeddinggemma:latest
```

Documentos são transformados em vetores e mantidos em cache em memória. A pergunta também é transformada em vetor e o ranking é calculado por **similaridade de cosseno**.

Não existe vector database na v0.3.1. Com apenas dez documentos, calcular o ranking diretamente deixa o experimento mais transparente.

Documentação oficial:

- https://docs.ollama.com/api/embed
- https://docs.ollama.com/capabilities/embeddings

## Retrieval híbrido

O modo `hybrid` combina os dois sinais:

```text
score híbrido =
    peso_embedding × score_semântico_normalizado
    +
    (1 - peso_embedding) × score_lexical_normalizado
```

O valor padrão é:

```text
HYBRID_EMBEDDING_WEIGHT = 0.5
```

Ele pode ser alterado pela variável de ambiente de mesmo nome. O valor `0.5` é uma baseline neutra, não uma afirmação de que metade/metade seja universalmente ótimo.

## Métricas de retrieval

O benchmark mede:

- `Recall@1`;
- `Recall@3`;
- `Recall@5`;
- `MRR`;
- latência média do retrieval;
- mediana da latência;
- p95 da latência;
- média cold;
- média warm;
- tempo cold/warm gasto em embeddings.

Perguntas sem documento esperado não entram no cálculo de Recall/MRR.

### Cold vs warm

Para `embedding` e `hybrid`:

```text
cold
→ cache dos embeddings de documentos limpo
→ documentos + pergunta precisam de embedding

warm
→ embeddings dos documentos já estão no cache
→ apenas a pergunta precisa de embedding novamente
```

Assim os dois retrievers semânticos pagam seu próprio cold start, em vez de um herdar o cache criado pelo outro.

## Métricas do LLM

Continuam sendo medidas:

- preservação de fatos obrigatórios;
- recuperação do documento esperado no Top-K enviado ao modelo;
- citação de fonte esperada;
- ausência de alterações conhecidas;
- capacidade de recusar quando não há evidência;
- latência média;
- mediana da latência;
- p95 da latência;
- tokens de entrada e saída;
- tokens por segundo.

A checagem automática de factualidade continua sendo uma proxy determinística baseada nos casos de teste.

## Casos semânticos

`data/test_cases.json` contém paráfrases desenhadas para reduzir a dependência de palavras idênticas.

Exemplos:

```text
"Quem atua à distância tem algum apoio financeiro mensal?"
"Existe apoio psicológico disponibilizado todos os meses?"
"Recebi uma mensagem estranha pedindo credenciais. O que faço?"
"A companhia banca estudos profissionais?"
```

Esses casos ajudam a expor diferenças entre recuperação lexical e semântica.

## Observabilidade básica

O projeto usa tracing estruturado com `trace_id`.

São registrados spans como:

```text
api.query
retrieval.search
ollama.embed
ollama.chat
evaluation.query
benchmark.run
benchmark.case
benchmark.retrieval
```

Os logs são JSON e registram metadados como modo de retrieval, modelo, Top-K e duração. O conteúdo completo dos documentos e prompts não é enviado aos logs.

Se a API do OpenTelemetry estiver instalada no ambiente, os spans também usam o tracer global. O projeto não obriga um backend específico de observabilidade; exportadores/Collector/Grafana podem ser adicionados depois.

## CPU-only

A geração e os embeddings são enviados ao Ollama com `num_gpu=0`.

Quando `/api/ps` não permite confirmar o uso de VRAM, o projeto usa `None`/`desconhecido` em vez de assumir falsamente que VRAM é zero.

## Base fictícia

Os documentos estão em `data/hr_documents.json` e representam a empresa fictícia **Aurora Labs**.

No startup, `database.py` sincroniza os documentos com `data/hr_documents.db` usando SQLite.

## Pré-requisitos

- Python 3.11+
- Poetry
- Ollama

## Instalação

```bash
poetry install
```

Instale pelo menos um modelo de geração:

```bash
ollama pull qwen3:0.6b
```

Para `embedding` e `hybrid`, instale também:

```bash
ollama pull embeddinggemma
```

## Executar

```bash
poetry run cpu-llm-lab
```

Abra:

```text
http://127.0.0.1:8000
```

## Endpoints

- `GET /api/health`
- `GET /api/models`
- `GET /api/documents`
- `POST /api/query`
- `POST /api/benchmark`

### Consulta

Exemplo de body:

```json
{
  "question": "Quem atua à distância tem algum apoio financeiro mensal?",
  "models": ["qwen3:0.6b"],
  "top_k": 3,
  "retrieval_mode": "embedding",
  "embedding_model": "embeddinggemma:latest"
}
```

### Benchmark

O endpoint aceita parâmetros repetidos e controles experimentais:

```text
models=qwen3:0.6b
retrieval_modes=lexical
retrieval_modes=embedding
retrieval_modes=hybrid
top_k=3
repetitions=3
order_seed=42
```

Com 14 casos, 3 retrievers, 1 modelo e 3 repetições:

```text
14 × 3 × 1 × 3 = 126 inferências de LLM
```

Em CPU, selecione poucos modelos quando quiser uma rodada rápida.

## Como conduzir o experimento

1. Use pelo menos 3 repetições para comparar latência.
2. Mantenha a mesma `order_seed` quando quiser reproduzir a mesma ordem.
3. Compare `cold retrieval` separadamente de `warm retrieval`.
4. Use mediana e p95 junto da média.
5. Compare Recall@K e MRR antes de atribuir um erro ao LLM.
6. Compare a mudança no `factual_score` somente depois de confirmar que o contexto correto chegou.
7. Analise os padrões de falha, não apenas a maior porcentagem da tabela.

## Estrutura relevante

```text
src/cpu_llm_lab/
├── app.py
├── benchmark.py
├── config.py
├── database.py
├── evaluator.py
├── observability.py
├── ollama_client.py
├── retrieval.py
└── schemas.py
```

## Limitações atuais

- base muito pequena;
- score híbrido ainda usa peso heurístico;
- factualidade ainda usa correspondência textual determinística;
- `hallucination_free` só detecta erros previamente cadastrados;
- observabilidade ainda não possui backend persistente;
- 3 repetições ainda são poucas para inferência estatística forte;
- o benchmark controla cold/warm do **retrieval**, mas ainda não força um cold start independente do LLM em cada execução;
- resultados continuam dependentes do hardware, quantização e condições locais;
- o projeto continua task-specific.

## Testes e CI

A suíte cobre API, métricas de retrieval, similaridade de cosseno, cache cold/warm e o fluxo de benchmark repetido. O repositório também executa `pytest` no GitHub Actions a cada push e pull request.

## Documentação

- [Technical Overview](docs/TECHNICAL_OVERVIEW.md)
- [v0.3 Study Guide](docs/V03_STUDY_GUIDE.md)
- [v0.3.1 Experimental Rigor](docs/V031_EXPERIMENTAL_RIGOR.md)

## Próximas linhas de pesquisa

Depois de compreender a v0.3.1, evoluções naturais incluem:

- avaliação factual por claims estruturadas;
- source precision/source recall;
- medição do processo Ollama;
- cold/warm start controlado do modelo de geração;
- persistência de histórico;
- OpenTelemetry Collector;
- múltiplos domínios e semantic routing.

O objetivo permanece o mesmo: **descobrir quando uma solução menor é suficiente e medir exatamente o que se perde ou ganha ao aumentar a complexidade**.
