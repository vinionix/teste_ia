# CPU LLM Lab

Laboratório local para comparar **LLMs pequenos e baratos em CPU** em uma tarefa corporativa de consulta fundamentada em documentos.

A pergunta central do projeto é simples:

> Qual é o menor modelo que ainda consegue responder com fidelidade suficiente para uma tarefa específica, sem inventar ou alterar informações da base?

Em vez de comparar modelos apenas por tamanho ou benchmark genérico, a aplicação mede comportamento em um cenário controlado e reproduzível.

## Problema que o projeto investiga

Modelos maiores costumam entregar mais capacidade geral, mas também aumentam custo, memória e latência. Em vários cenários corporativos, porém, a tarefa é estreita: consultar políticas, procedimentos ou documentos internos e devolver uma resposta fundamentada.

O CPU LLM Lab foi criado para estudar esse trade-off entre:

- qualidade da resposta;
- fidelidade ao documento;
- capacidade de recusar quando não há evidência;
- latência;
- quantidade de tokens;
- throughput;
- custo computacional local.

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
contexto controlado
  ↓
Ollama + modelo escolhido
  ↓
resposta estruturada + fontes
  ↓
avaliação / benchmark
```

O modelo **não recebe acesso direto ao SQL**. A aplicação recupera os documentos primeiro e entrega apenas o contexto selecionado ao LLM. Isso mantém o experimento focado em grounded generation, e não em text-to-SQL.

## O que o benchmark mede

- preservação de fatos obrigatórios;
- recuperação do documento esperado;
- citação da fonte esperada;
- ausência de alterações conhecidas nos fatos;
- capacidade de recusar perguntas cuja resposta não existe na base;
- latência;
- tokens de entrada e saída;
- tokens por segundo;
- desempenho comparativo por modelo.

> A checagem automática de factualidade usada aqui é determinística e baseada nos casos de teste. Ela funciona como proxy de fidelidade e não como prova de que cada frase gerada é totalmente suportada pela fonte.

## Base fictícia

Os documentos estão em `data/hr_documents.json` e representam a empresa fictícia **Aurora Labs**. A base inclui temas como:

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

## Casos negativos e alucinação

O benchmark não contém apenas perguntas respondíveis. Existem casos cuja resposta **não está na base**.

Esses testes verificam se o modelo consegue dizer que não encontrou a informação em vez de preencher a lacuna com conhecimento externo ou invenção. Para este projeto, saber recusar corretamente é parte da qualidade.

## Modelos sugeridos para comparação

O frontend mostra automaticamente os modelos instalados no Ollama e também recomenda uma faixa de modelos pequenos, por exemplo:

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

Qualquer outro modelo instalado localmente também pode ser usado no laboratório.

## Pré-requisitos

- Python 3.11+
- Poetry
- Ollama

## Instalação

```bash
poetry install
```

Exemplo de download de modelos:

```bash
ollama pull gemma3:270m
ollama pull qwen3:0.6b
ollama pull llama3.2:1b
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

## Como conduzir um experimento

1. Instale os modelos que deseja comparar.
2. Execute a aplicação localmente.
3. Rode os mesmos casos de benchmark em cada modelo.
4. Compare fidelidade e métricas de desempenho.
5. Analise os padrões de falha, não apenas o score agregado.
6. Identifique o menor modelo que ainda atende o nível de qualidade desejado.

## Decisões de engenharia

Este projeto é deliberadamente simples em alguns pontos:

- recuperação lexical em vez de um pipeline vetorial completo;
- base local e fictícia para manter experimentos reproduzíveis;
- checagens determinísticas no benchmark;
- execução local via Ollama;
- foco em uma tarefa estreita em vez de avaliação geral de modelos.

Essas escolhas ajudam a isolar o comportamento que está sendo estudado e reduzem custo e complexidade experimental.

## Limitações atuais

- o benchmark de factualidade é task-specific;
- recuperação lexical pode perder relações semânticas;
- os resultados dependem do hardware local e da quantização do modelo;
- o projeto não pretende medir capacidade geral de um LLM;
- ausência de uma resposta incorreta conhecida não garante factualidade completa.

## O que este projeto demonstra

- avaliação prática de LLMs;
- grounded generation;
- execução CPU-only;
- análise de qualidade × custo computacional;
- testes de recusa e pressão contra alucinação;
- FastAPI e saída estruturada;
- experimentação reproduzível com modelos locais.

## Documentação

- [Technical Overview](docs/TECHNICAL_OVERVIEW.md) — arquitetura, metodologia de avaliação, fronteiras do experimento e ideias para próximos testes.

## Direção futura

Entre os experimentos que podem ser adicionados estão recuperação híbrida/vetorial, medição de memória e CPU, comparação entre quantizações, testes adversariais e exportação de resultados históricos.

O objetivo continua o mesmo: **descobrir quando um modelo menor é suficiente para uma tarefa realista**.
