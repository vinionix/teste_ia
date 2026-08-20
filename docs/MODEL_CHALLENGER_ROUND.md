# Model Challenger Round

## Objetivo

Esta rodada existe para desafiar o melhor modelo observado no benchmark anterior sem repetir trabalho que já foi resolvido na etapa de retrieval.

O benchmark anterior indicou que `embedding` foi o melhor retriever para a base atual, com `Recall@1 = 100%`, `Recall@3 = 100%`, `Recall@5 = 100%` e `MRR = 1.0`. Por isso, esta comparação deve manter o retriever fixo em `embedding` e variar somente o modelo de geração.

## Modelos

Baseline atual:

- `qwen3:1.7b`

Desafiantes:

- `qwen3.5:2b`
- `granite4:3b`
- `phi4-mini:3.8b`

A escolha mantém os candidatos em uma faixa pequena o suficiente para CPU e adiciona três famílias com propostas diferentes: uma geração mais nova do Qwen, um modelo voltado a instruction-following/RAG empresarial e um modelo compacto da família Phi.

## Condições que devem permanecer fixas

```text
retrieval_mode = embedding
embedding_model = embeddinggemma:latest
top_k = 3
repetitions = 3
order_seed = 42
temperature = 0
num_ctx = 4096
num_predict = 320
think = false
num_gpu = 0
```

Não altere prompt, documentos, casos ou parâmetros entre modelos durante esta rodada.

## Instalação

```bash
ollama pull qwen3:1.7b
ollama pull qwen3.5:2b
ollama pull granite4:3b
ollama pull phi4-mini:3.8b
ollama pull embeddinggemma
```

Depois:

```bash
git pull
poetry install
poetry run cpu-llm-lab
```

## Volume do experimento

A base atual possui 14 casos.

Com 1 retriever, 4 modelos e 3 repetições:

```text
14 casos × 1 retriever × 4 modelos × 3 repetições
= 168 inferências de LLM
```

Isso é suficiente para comparar os geradores sem pagar novamente o custo de testar lexical e hybrid, já que o retriever é compartilhado pelos modelos e a pergunta desta rodada é qual LLM se comporta melhor dado o mesmo contexto.

## Métricas principais

Prioridade de decisão:

1. `avg_factual_score`
2. `source_accuracy_rate`
3. `abstention_accuracy_rate`
4. `hallucination_free_rate`
5. quantidade de erros
6. mediana e p95 de latência do LLM
7. tokens por segundo

O retrieval deve continuar sendo verificado, mas espera-se que permaneça igual para todos os modelos porque ele ocorre antes da geração.

## Baseline a superar

O resultado anterior do `qwen3:1.7b` com embedding foi:

```text
Fatos:                 73.81%
Hit Top-K:            100%
Fontes:               100%
Sem alteração:        100%
Recusa:               100%
LLM médio:             10.93 s
Erros:                  0
```

Um desafiante só deve substituir a baseline se melhorar factualidade sem introduzir regressões relevantes em fontes, recusa, estabilidade ou custo de execução.

## Critério sugerido de vitória

Um candidato é claramente melhor se atingir, na mesma rodada:

```text
factual_score > 73.81%
source_accuracy = 100%
abstention_accuracy = 100%
hallucination_free = 100%
erros = 0
```

Depois disso, latência e tokens/s funcionam como desempate.

Se um modelo melhorar factualidade mas piorar uma métrica de segurança/grounding, o resultado deve ser tratado como trade-off, não como vitória automática.

## Hipóteses antes do teste

- `qwen3.5:2b`: principal candidato a superar a baseline em factualidade mantendo custo relativamente próximo.
- `granite4:3b`: candidato interessante para instruction-following e RAG; pode se destacar em aderência ao formato e às fontes.
- `phi4-mini:3.8b`: pode melhorar raciocínio e preservação de fatos, mas deve pagar um custo maior de latência em CPU.

Essas são hipóteses, não conclusões. A decisão deve ser feita somente com os resultados obtidos no mesmo protocolo.

## Resultado esperado da rodada

Ao final, a pergunta deve ser respondida de forma simples:

> Dado retrieval por embedding perfeito ou quase perfeito, qual modelo pequeno produz a melhor resposta grounded por unidade de tempo em CPU?
