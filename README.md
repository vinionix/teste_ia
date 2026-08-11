# From Zero to MiniGPT

Construindo modelos de Inteligência Artificial do zero, desde a matemática básica até um pequeno modelo de linguagem baseado na arquitetura Transformer.

O objetivo deste repositório não é apenas utilizar bibliotecas prontas ou consumir APIs de modelos existentes. A proposta é compreender, implementar, testar e documentar os principais componentes envolvidos na construção de modelos modernos de IA.

## Objetivo

Responder, na prática, à seguinte pergunta:

> Como um modelo de Inteligência Artificial aprende e como podemos construir um pequeno modelo de linguagem compreendendo cada uma de suas partes?

O projeto combina quatro frentes em cada etapa:

1. **Matemática:** compreender as operações e derivar as equações principais.
2. **Implementação:** transformar os conceitos em código, começando com Python e NumPy.
3. **Experimentação:** modificar parâmetros, provocar falhas e medir o comportamento.
4. **Documentação:** registrar resultados, limitações, erros e aprendizados.

A matemática será estudada de forma incremental e aplicada. O projeto não exige dominar cálculo ou álgebra linear antes de começar: a base será reconstruída dentro do próprio roadmap.

## Roadmap

- [ ] Fundação do projeto e aprendizado de Poetry
- [ ] Matemática básica para IA: aritmética, álgebra, funções, gráficos, potências e logaritmos
- [ ] Fundamentos matemáticos computacionais: vetores, matrizes, derivadas e probabilidade
- [ ] Regressão linear do zero
- [ ] Classificação e neurônio artificial
- [ ] Rede neural multicamada e backpropagation
- [ ] Modelos de linguagem simples
- [ ] Comparação entre RNN, GRU/LSTM, convolução causal e atenção
- [ ] Embeddings
- [ ] Self-attention e máscara causal
- [ ] Bloco Transformer decoder-only
- [ ] MiniGPT
- [ ] Fine-tuning e aprendizado por preferências
- [ ] API, RAG, ferramentas, avaliações e segurança
- [ ] Assistente técnico final para redes, infraestrutura ou cibersegurança

O roadmap detalhado está em [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Fase -1 — Matemática básica para IA

Antes da álgebra linear, o projeto reconstrói a base necessária para ler e manipular fórmulas com segurança:

- aritmética, frações, razões e proporções;
- álgebra básica e equações;
- funções, plano cartesiano e leitura de gráficos;
- potências, raízes e notação científica;
- exponenciais e logaritmos;
- somatórios, índices, média e notação matemática;
- checkpoint prático aplicado a `y = wx + b`.

O código nessa fase serve para **validar a matemática**, não para substituir o raciocínio manual.

## Estrutura planejada

```text
teste_ia/
├── 00_project_foundations/
├── 01_basic_math/
├── 02_math_foundations/
├── 03_linear_regression/
├── 04_logistic_neuron/
├── 05_mlp_backpropagation/
├── 06_sequence_models/
├── 07_embeddings/
├── 08_self_attention/
├── 09_transformer/
├── 10_minigpt/
├── 11_post_training/
├── 12_llm_system/
├── docs/
├── experiments/
├── tests/
└── README.md
```

## Ambiente inicial

O ambiente Python será configurado como parte da primeira fase do projeto. A ideia é aprender Poetry enquanto ele resolve um problema real do repositório.

Dependências iniciais previstas:

- NumPy
- Matplotlib
- Jupyter Notebook
- ipykernel
- pytest

O `pyproject.toml` não será pré-configurado: criá-lo e compreender o papel do Poetry faz parte da etapa de fundação.

## Tecnologias avançadas

Serão introduzidas apenas quando o projeto justificar sua utilização:

- PyTorch
- FastAPI
- Docker
- modelos abertos e bibliotecas de tokenização
- bancos vetoriais
- ferramentas de avaliação de LLMs

As primeiras implementações evitarão frameworks que escondam o cálculo dos gradientes. PyTorch será introduzido depois que os fundamentos forem implementados manualmente.

## Critério de conclusão

Uma etapa só será considerada concluída quando for possível:

- explicar a matemática com palavras próprias;
- calcular manualmente um exemplo pequeno;
- implementar o conceito sem copiar cegamente;
- criar testes para os comportamentos importantes;
- alterar parâmetros e analisar os resultados;
- identificar ao menos uma limitação;
- documentar erros, resultados e aprendizados.

Executar código copiado não representa a conclusão de uma etapa.

## Projeto final

O projeto final será um assistente técnico voltado para redes, infraestrutura ou cibersegurança. O sistema deverá combinar um modelo de linguagem com recuperação de documentação, ferramentas restritas, avaliações automatizadas, observabilidade e controles contra ataques como prompt injection.

## Autor

Desenvolvido por Vinícius Fidelis.

- GitHub: [vinionix](https://github.com/vinionix)
- LinkedIn: [vfidelis](https://www.linkedin.com/in/vfidelis)
