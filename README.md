# CPU LLM Lab

Laboratório local para comparar modelos de linguagem pequenos e baratos em uma tarefa corporativa de **consulta fundamentada em documentos**.

A aplicação cria uma base SQLite fictícia da empresa **Aurora Labs**, recupera documentos relevantes para cada pergunta e envia somente esse contexto ao modelo executado pelo Ollama. O objetivo é descobrir qual é o menor modelo que ainda consegue responder com fidelidade suficiente, sem alterar números, prazos, valores ou regras presentes na base.

## O que o benchmark mede

- preservação de fatos obrigatórios;
- recuperação do documento esperado;
- citação da fonte esperada;
- ausência de alterações conhecidas nos fatos;
- capacidade de recusar perguntas cuja resposta não existe na base;
- latência;
- tokens de entrada e saída;
- tokens por segundo;
- execução CPU-only.

> A checagem automática de factualidade usada aqui é determinística e baseada nos casos de teste. Ela é um proxy de fidelidade, não uma prova completa de que toda frase gerada está suportada pela fonte.

## Base fictícia

Os documentos estão em `data/hr_documents.json` e incluem:

1. férias e descanso;
2. benefícios gerais;
3. plano de saúde e odontológico;
4. trabalho híbrido e auxílio home office;
5. educação e certificações;
6. bem-estar;
7. viagens e reembolsos;
8. licenças familiares;
9. mobilidade;
10. segurança da informação.

No startup, `database.py` sincroniza esses documentos com `data/hr_documents.db` usando SQLite.

## Arquitetura

```text
Pergunta
  ↓
SQLite
  ↓
recuperação lexical (top-k)
  ↓
documentos relevantes
  ↓
Ollama + modelo escolhido
  ↓
resposta estruturada + fontes
  ↓
avaliação / benchmark
```

O modelo **não recebe acesso direto ao SQL**. A aplicação recupera os documentos primeiro e entrega apenas o contexto selecionado ao LLM.

## Modelos sugeridos para comparação

O frontend mostra automaticamente todos os modelos já instalados no Ollama e também recomenda uma faixa de modelos pequenos:

- `gemma3:270m`
- `smollm2:135m`
- `smollm2:360m`
- `qwen3:0.6b`
- `gemma3:1b`
- `llama3.2:1b`
- `qwen3:1.7b`
- `smollm2:1.7b`
- `llama3.2:3b`
- `qwen3:4b`

Baixe apenas os que quiser testar, por exemplo:

```bash
ollama pull gemma3:270m
ollama pull qwen3:0.6b
ollama pull llama3.2:1b
```

Qualquer outro modelo instalado no Ollama também aparecerá automaticamente no frontend.

## Pré-requisitos

- Python 3.11+
- Poetry
- Ollama

## Instalação

```bash
poetry install
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

## Sobre o benchmark

Os casos estão em `data/test_cases.json`. Além de perguntas respondíveis, há perguntas cuja informação não existe na base. Esses casos verificam se o modelo consegue dizer que não encontrou a informação em vez de preencher a lacuna com conhecimento externo ou invenção.

O objetivo do projeto não é encontrar o maior score absoluto, mas estudar o **trade-off qualidade × custo computacional** e descobrir quando um modelo menor é suficiente para a tarefa.
