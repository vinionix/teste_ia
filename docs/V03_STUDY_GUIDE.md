# CPU LLM Lab v0.3 — Study Guide

Use este arquivo para estudar a atualização sem tentar entender tudo de uma vez.

## 1. Comece pelo problema

Antes da v0.3, o projeto tinha uma única forma de encontrar documentos:

```text
pergunta
→ normalização
→ tokens
→ score lexical 5/3/1
→ Top-K
```

A nova pergunta é:

> O que acontece quando a pergunta tem o mesmo significado do documento, mas usa palavras diferentes?

É por isso que a v0.3 adiciona embeddings.

## 2. Leia nesta ordem

```text
schemas.py
↓
database.py
↓
retrieval.py
↓
ollama_client.py
↓
evaluator.py
↓
benchmark.py
↓
observability.py
↓
app.py
```

Não comece pelo frontend.

## 3. `schemas.py`

Procure primeiro:

```python
RetrievalMode
RetrievedDocument
RetrievalTrace
BenchmarkRow
```

Perguntas para responder:

- Por que `score` virou `float`?
- Por que lexical, embedding e hybrid precisam do mesmo formato de saída?
- Que informação `RetrievalTrace` guarda?
- Por que `cpu_only_verified` agora pode ser `None`?

## 4. `database.py`

A função mais importante continua sendo a baseline lexical.

Leia:

```text
_normalize()
_tokens()
_lexical_score()
rank_documents_lexical()
search_documents()
```

Perceba a separação nova:

```text
rank_documents_lexical()
→ devolve o ranking

search_documents()
→ corta o ranking no Top-K
```

Essa separação é útil porque o benchmark precisa enxergar até Top-5 mesmo quando o LLM recebe apenas Top-3.

## 5. `retrieval.py`

Esse é o arquivo principal da v0.3.

### `_document_text()`

Transforma um `Document` em um único texto para embedding.

### `_cosine_similarity()`

Recebe dois vetores e retorna quão alinhados eles estão.

Estude a fórmula:

```text
cos(A,B) =
A · B
────────────
|A| × |B|
```

Você não precisa decorar. Precisa entender:

```text
mesma direção → próximo de 1
perpendiculares → próximo de 0
direções opostas → próximo de -1
```

### `_document_embeddings()`

Pergunte:

- Por que existe cache?
- O que acontece no primeiro uso?
- O que acontece na segunda pergunta?
- Por que o hash do documento entra na chave?

### `_embedding_scores()`

Fluxo:

```text
documentos
→ embeddings dos documentos

pergunta
→ embedding da pergunta

vetor da pergunta
×
cada vetor de documento
→ similaridade
```

### `_rank_from_scores()`

É o equivalente semântico da ordenação que já existia no lexical.

### `retrieve()`

Essa função oferece uma interface única:

```text
retrieve(mode="lexical")
retrieve(mode="embedding")
retrieve(mode="hybrid")
```

O restante da aplicação não precisa saber como cada algoritmo funciona internamente.

Isso é uma decisão arquitetural importante.

## 6. Entenda o Hybrid

O lexical produz números como:

```text
9
5
1
```

O embedding pode produzir:

```text
0.91
0.72
0.33
```

Não podemos simplesmente somar os valores crus.

Por isso os scores são normalizados antes da combinação.

O peso padrão 0.5/0.5 é apenas uma baseline.

Pergunta de pesquisa:

> Qual peso produz o melhor Recall@K para este conjunto de documentos?

## 7. `ollama_client.py`

A função nova é:

```python
embed_texts()
```

Ela chama:

```text
POST /api/embed
```

Entrada:

```text
modelo de embedding
+
lista de textos
```

Saída:

```text
lista de vetores
+
tempo da operação
```

Depois releia `run_grounded_query()` e perceba que ele continua fazendo geração. Embedding e geração são responsabilidades diferentes.

## 8. Retriever não é LLM

Guarde:

```text
Retriever
→ escolhe informação

LLM
→ produz resposta usando a informação
```

Na v0.3:

```text
                  ┌ lexical
pergunta → retrieve├ embedding
                  └ hybrid
                     ↓
                   Top-K
                     ↓
                    LLM
```

## 9. Métricas novas

### Recall@1

O documento esperado apareceu na primeira posição?

### Recall@3

Quanto dos documentos esperados apareceu nas três primeiras posições?

### Recall@5

Mesma ideia para cinco posições.

### MRR

Pergunta:

> Em qual posição apareceu o primeiro documento relevante?

Exemplo:

```text
ranking:
#2
#7
#4  ← correto

rank = 3
MRR desse caso = 1 / 3
```

Quanto mais cedo o documento correto aparece, maior o valor.

## 10. `benchmark.py`

Observe os dois loops novos:

```text
para cada caso
    para cada retriever
        recupera documentos
        para cada modelo
            chama LLM
            avalia
```

Isso permite comparar:

```text
retrieval_mode × model
```

Mas também aumenta muito o tempo de benchmark.

## 11. Por que recuperar `max(top_k, 5)`?

Se:

```text
top_k = 3
```

a LLM recebe três documentos.

Mas ainda queremos calcular:

```text
Recall@5
```

Então o benchmark cria um ranking de pelo menos cinco documentos e envia apenas os `top_k` primeiros ao modelo.

Essa diferença é importante.

## 12. `observability.py`

Não trate como "magia de observabilidade".

Ela faz basicamente:

```text
gera trace_id
↓
marca início
↓
executa função/bloco
↓
mede duração
↓
registra fim
```

Quando uma operação está dentro de outra, elas compartilham o mesmo `trace_id`.

Então você consegue relacionar:

```text
api.query
retrieval.search
ollama.embed
ollama.chat
```

como partes da mesma requisição.

## 13. Experimento que você deve fazer

Use a pergunta:

```text
Quem atua à distância tem algum apoio financeiro mensal?
```

Rode:

```text
lexical
embedding
hybrid
```

Antes de executar, escreva sua previsão:

- qual documento ficará em primeiro?
- qual score será maior?
- qual terá maior latência?

Depois compare com o resultado.

Faça o mesmo com:

```text
Recebi uma mensagem estranha pedindo credenciais. O que faço?
```

## 14. O que você deve conseguir explicar no final

Sem olhar o código:

1. O que é um retriever?
2. Por que retrieval acontece antes do LLM?
3. Qual a limitação da busca lexical?
4. O que é um embedding?
5. Por que usamos o mesmo modelo para pergunta e documentos?
6. O que a similaridade de cosseno compara?
7. O que é Top-K?
8. Por que manter o lexical como baseline?
9. Como funciona o hybrid?
10. O que Recall@3 mede?
11. O que MRR mede?
12. Por que melhorar retrieval pode melhorar factualidade?
13. Por que melhorar retrieval também pode aumentar latência?
14. O que o `trace_id` permite investigar?

Se você consegue explicar essas quatorze respostas, você entendeu a parte central da v0.3.
