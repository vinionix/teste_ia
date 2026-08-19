# CPU LLM Lab

Laboratório local para comparar **retrieval + LLMs pequenos em CPU** em consultas corporativas fundamentadas em documentos.

A pergunta central continua sendo:

> Qual é a configuração mais barata que mantém qualidade suficiente para uma tarefa específica sem inventar ou alterar informações da base?

A versão **0.3.0** amplia o experimento: agora o laboratório compara não apenas modelos de linguagem, mas também **estratégias de recuperação de contexto**.

## v0.3 — Retrieval Lab

O projeto oferece três retrievers:

- `lexical`: baseline original baseada em palavras e pesos `5/3/1`;
- `embedding`: busca semântica com embeddings gerados pelo Ollama;
- `hybrid`: combinação normalizada do score lexical com similaridade semântica.

A baseline lexical foi mantida de propósito. O objetivo é medir se a complexidade extra do embedding realmente melhora o retrieval.

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

O `EmbeddingRetriever` usa `POST /api/embed` do Ollama.

Por padrão:

```text
embedding_model = embeddinggemma
```

Documentos são transformados em vetores e mantidos em cache em memória. A pergunta também é transformada em vetor e o ranking é calculado por **similaridade de cosseno**.

Não existe vector database na v0.3. Com apenas dez documentos, calcular o ranking diretamente deixa o experimento mais transparente.

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

Além de `retrieval_hit`, a v0.3 mede:

- `Recall@1`: quanto do conjunto de documentos esperados aparece na primeira posição;
- `Recall@3`: quanto aparece nos três primeiros;
- `Recall@5`: quanto aparece nos cinco primeiros;
- `MRR`: posição do primeiro documento relevante;
- latência média do retrieval;
- tempo gasto especificamente na geração dos embeddings.

Perguntas sem documento esperado não entram no cálculo de Recall/MRR.

## Métricas do LLM

Continuam sendo medidas:

- preservação de fatos obrigatórios;
- recuperação do documento esperado no Top-K enviado ao modelo;
- citação de fonte esperada;
- ausência de alterações conhecidas;
- capacidade de recusar quando não há evidência;
- latência;
- tokens de entrada e saída;
- tokens por segundo.

A checagem automática de factualidade continua sendo uma proxy determinística baseada nos casos de teste.

## Casos semânticos

`data/test_cases.json` agora contém paráfrases desenhadas para reduzir a dependência de palavras idênticas.

Exemplos:

```text
"Quem atua à distância tem algum apoio financeiro mensal?"
"Existe apoio psicológico disponibilizado todos os meses?"
"Recebi uma mensagem estranha pedindo credenciais. O que faço?"
"A companhia banca estudos profissionais?"
```

Esses casos ajudam a expor diferenças entre recuperação lexical e semântica.

## Observabilidade básica

A v0.3 adiciona tracing estruturado com `trace_id`.

São registrados spans como:

```text
api.query
retrieval.search
ollama.embed
ollama.chat
evaluation.query
benchmark.run
benchmark.case
```

Os logs são JSON e registram metadados como modo de retrieval, modelo, Top-K e duração. O conteúdo completo dos documentos e prompts não é enviado aos logs.

Se a API do OpenTelemetry estiver instalada no ambiente, os spans também usam o tracer global. A v0.3 não obriga um backend específico de observabilidade; exportadores/Collector/Grafana podem ser adicionados depois.

## CPU-only

A geração e os embeddings são enviados ao Ollama com `num_gpu=0`.

A verificação de VRAM foi corrigida: quando `/api/ps` não permite confirmar o valor, o projeto usa `None`/`desconhecido` em vez de assumir falsamente que VRAM é zero.

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
  "embedding_model": "embeddinggemma"
}
```

### Benchmark

O endpoint aceita parâmetros repetidos:

```text
models=qwen3:0.6b
retrieval_modes=lexical
retrieval_modes=embedding
retrieval_modes=hybrid
top_k=3
```

Comparar três retrievers executa o LLM três vezes por caso/modelo. Em CPU, selecione poucos modelos quando quiser uma rodada rápida.

## Como conduzir o experimento

1. Rode `lexical` como baseline.
2. Rode `embedding` nas mesmas perguntas.
3. Observe principalmente os casos de paráfrase.
4. Compare Recall@K e MRR.
5. Compare também a mudança no `factual_score`.
6. Rode `hybrid`.
7. Analise o ganho de qualidade em relação à latência extra.
8. Só então decida se a complexidade adicional vale a pena.

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
- embeddings ficam apenas em cache de memória;
- score híbrido ainda usa peso heurístico;
- factualidade ainda usa correspondência textual determinística;
- `hallucination_free` só detecta erros previamente cadastrados;
- observabilidade ainda não possui backend persistente;
- não há repetição estatística de cada combinação;
- o projeto continua task-specific.

## Documentação

- [Technical Overview](docs/TECHNICAL_OVERVIEW.md)
- [v0.3 Study Guide](docs/V03_STUDY_GUIDE.md)

## Próximas linhas de pesquisa

Depois de compreender a v0.3, evoluções naturais incluem:

- avaliação factual por claims estruturadas;
- source precision/source recall;
- repetições e percentis;
- medição do processo Ollama;
- persistência de histórico;
- OpenTelemetry Collector;
- múltiplos domínios e semantic routing.

O objetivo permanece o mesmo: **descobrir quando uma solução menor é suficiente e medir exatamente o que se perde ou ganha ao aumentar a complexidade**.
