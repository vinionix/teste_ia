# CPU LLM Lab v0.3.1 — Rigor Experimental

Este documento explica por que a v0.3.1 existe e como ler as novas medições.

## 1. O problema encontrado na v0.3

A v0.3 introduziu cache de embeddings dos documentos.

Isso é desejável para uso normal:

```text
primeira consulta
→ gera embeddings dos documentos
→ guarda no cache

consultas seguintes
→ reutilizam embeddings
```

Mas o primeiro benchmark comparou os retrievers na mesma execução. Se `embedding` rodasse antes de `hybrid`, o segundo poderia reutilizar o cache criado pelo primeiro.

Então uma tabela poderia mostrar algo como:

```text
embedding = 400 ms
hybrid    = 150 ms
```

sem que isso significasse que o algoritmo híbrido fosse realmente mais barato.

A diferença poderia vir simplesmente de:

```text
embedding
→ pagou o cold start

hybrid
→ recebeu cache quente
```

Esse é um exemplo de **variável de confusão experimental**.

## 2. O que a v0.3.1 controla

Para cada combinação `caso × retriever`:

```text
repetição 1
→ cold

repetição 2..N
→ warm
```

Antes da repetição cold de `embedding` e `hybrid`, o benchmark executa:

```python
retriever.clear_embedding_cache(embedding_model)
```

Isso garante que os dois modos semânticos precisam pagar seu próprio custo de criação dos vetores dos documentos.

## 3. O que significa cold

No benchmark atual, cold se refere ao **cache de embeddings dos documentos**.

Para embedding/hybrid:

```text
COLD
pergunta
↓
embedding dos documentos
+
embedding da pergunta
↓
ranking
```

Já no warm:

```text
WARM
pergunta
↓
embeddings dos documentos já existem
+
embedding novo da pergunta
↓
ranking
```

Importante: isso não significa que o modelo de geração esteja em cold start. O Ollama ainda pode manter o LLM carregado por causa de `keep_alive`.

## 4. Por que repetir

Uma única medição de tempo pode ser afetada por:

- escalonamento do sistema operacional;
- outros processos usando CPU;
- cache;
- carregamento do modelo;
- tamanho da resposta gerada;
- variação de execução do runtime.

Por isso a v0.3.1 permite:

```text
1 a 10 repetições por caso
```

O padrão é:

```text
3 repetições
```

Ainda é pouco para uma análise estatística forte, mas é muito melhor do que uma única execução.

## 5. Média, mediana e p95

### Média

```text
(tempo1 + tempo2 + ... + tempoN) / N
```

É fácil de entender, mas um valor muito alto pode puxar a média para cima.

### Mediana

Ordene os valores e pegue o centro.

Exemplo:

```text
100
110
115
120
900
```

Média:

```text
269 ms
```

Mediana:

```text
115 ms
```

A mediana representa melhor o comportamento típico quando há outliers.

### p95

O p95 tenta responder:

> Até que latência aproximadamente 95% das execuções ficaram?

Ele é útil para enxergar a cauda da distribuição.

Uma média boa pode esconder execuções ocasionalmente muito lentas.

## 6. Ordem experimental

O benchmark agora embaralha a ordem dos retrievers e dos modelos.

Mas não usa aleatoriedade impossível de reproduzir.

Existe:

```text
order_seed = 42
```

Com a mesma seed e os mesmos inputs, a ordem pseudoaleatória será a mesma.

Isso ajuda a reduzir situações como:

```text
Modelo A sempre roda primeiro
Modelo B sempre roda depois com cache quente
```

ou:

```text
Embedding sempre aquece recursos antes do Hybrid
```

## 7. Novas métricas de retrieval

Cada linha do benchmark agora possui:

```text
avg_retrieval_ms
median_retrieval_ms
p95_retrieval_ms
avg_cold_retrieval_ms
avg_warm_retrieval_ms

avg_embedding_ms
avg_cold_embedding_ms
avg_warm_embedding_ms
```

Leia assim:

```text
cold retrieval
→ custo de começar sem cache de documentos

warm retrieval
→ custo quando documentos já possuem vetores em cache

p95
→ comportamento das execuções mais lentas
```

## 8. Novas métricas do LLM

Agora também existem:

```text
avg_total_ms
median_total_ms
p95_total_ms
```

Isso melhora a comparação de latência entre modelos.

Mas existe uma limitação importante:

> A v0.3.1 não força o Ollama a descarregar o modelo entre execuções.

Portanto os números de geração ainda misturam efeitos de modelos já carregados e outros fatores do runtime.

## 9. `cases` não é igual a `executions`

Agora a linha possui:

```text
cases
successful_cases
executions
successful_executions
```

Exemplo:

```text
14 casos
3 repetições
```

para um único modelo/retriever:

```text
cases = 14
executions = 42
```

Se todos funcionarem:

```text
successful_cases = 14
successful_executions = 42
```

Isso evita confundir quantidade de perguntas com quantidade real de inferências.

## 10. Quantas inferências serão executadas?

A fórmula é:

```text
casos
× retrievers
× modelos
× repetições
```

Exemplo:

```text
14 casos
3 retrievers
4 modelos
3 repetições

14 × 3 × 4 × 3
= 504 inferências
```

Em CPU isso pode levar bastante tempo.

Para experimentação rápida, use:

```text
1 modelo
3 retrievers
3 repetições
```

Depois aumente o número de modelos quando o pipeline estiver validado.

## 11. Como interpretar a próxima tabela

Não escolha o vencedor olhando apenas uma coluna.

Primeiro pergunte:

```text
1. O retriever encontrou o contexto?
   → Recall@K / MRR / Hit Top-K

2. O LLM usou esse contexto corretamente?
   → Fatos / Fontes / Recusa / Sem alteração conhecida

3. Quanto custou o retrieval?
   → cold / warm / mediana / p95

4. Quanto custou a geração?
   → LLM mediana / p95 / tokens/s
```

A análise correta separa:

```text
qualidade de retrieval
≠
qualidade de geração
≠
custo de retrieval
≠
custo de geração
```

## 12. O que a v0.3.1 ainda não resolve

Ainda não controlamos completamente:

- cold start do modelo de geração;
- uso total de RAM pelo processo Ollama;
- consumo energético;
- significância estatística com muitas repetições;
- variações entre quantizações;
- factualidade semântica completa;
- source faithfulness.

Esses pontos devem ser tratados como próximas hipóteses, não como fatos já medidos.

## 13. Checklist de entendimento

Antes de seguir para outra feature, tente responder sem olhar:

1. Por que o resultado de latência da v0.3 estava contaminado pelo cache?
2. O que `clear_embedding_cache()` resolve?
3. Qual a diferença entre cold e warm neste projeto?
4. Por que o embedding da pergunta continua sendo calculado no warm?
5. Por que a mediana pode ser melhor que a média para latência?
6. O que p95 representa?
7. Por que embaralhar a ordem dos modelos?
8. Por que usamos uma seed?
9. Qual a diferença entre `cases` e `executions`?
10. Por que ainda não podemos afirmar que medimos cold start real do LLM?

Se essas respostas estiverem claras, você entendeu o motivo da v0.3.1.

A suíte automatizada do repositório também verifica o comportamento de cache cold/warm e os controles de repetição do benchmark.
