# CPU LLM Lab

Aplicação local para comparar pequenos modelos de linguagem no Ollama em uma tarefa objetiva: receber um registro estruturado e gerar uma mensagem curta em português sem inventar dados.

## O que ela mede

- validade do JSON / schema;
- preservação de nome, valor e vencimento;
- categoria esperada nos casos de benchmark;
- tempo total e tempo de carregamento;
- tokens de entrada e saída;
- tokens por segundo;
- VRAM reportada pelo Ollama (`0` = execução CPU-only verificada pelo app).

## Modelos iniciais

- `qwen3:0.6b`
- `gemma3:1b`
- `qwen3:1.7b`

As requisições também enviam `num_gpu: 0` nas opções do Ollama.

## Pré-requisitos

- Python 3.11+
- Poetry
- Ollama instalado e em execução

## Instalação

```bash
poetry install
ollama pull qwen3:0.6b
ollama pull gemma3:1b
ollama pull qwen3:1.7b
```

Inicie o Ollama em outro terminal se ele não estiver rodando:

```bash
ollama serve
```

Depois confirme no terminal do Ollama que os modelos estão sendo executados em CPU:

```bash
ollama ps
```

A coluna `PROCESSOR` deve indicar `100% CPU`. A aplicação também consulta `/api/ps` e considera CPU-only quando `size_vram == 0`.

## Executar

```bash
poetry run cpu-llm-lab
```

Abra:

```text
http://127.0.0.1:8000
```

Também é possível executar diretamente:

```bash
poetry run uvicorn cpu_llm_lab.app:app --host 127.0.0.1 --port 8000
```

## Endpoints

- `GET /api/health`
- `GET /api/models`
- `POST /api/compare`
- `POST /api/benchmark`

## Observação sobre RAM

A métrica `process_rss_mb` mede o processo Python da aplicação, não toda a memória usada pelo processo do Ollama. Para benchmarking de memória do Ollama em nível de sistema, use também o monitor do sistema operacional. A métrica de VRAM vem diretamente de `/api/ps`.
