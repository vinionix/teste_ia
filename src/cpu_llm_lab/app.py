from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from .benchmark import run_benchmark
from .config import (
    CASES_PATH,
    DATABASE_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DOCUMENTS_SEED_PATH,
    RECOMMENDED_MODELS,
)
from .database import init_database, list_documents
from .observability import traced
from .ollama_client import OllamaClient
from .retrieval import Retriever
from .schemas import (
    BenchmarkResponse,
    ModelResult,
    QueryRequest,
    RetrievalMode,
)

client: OllamaClient | None = None
retriever: Retriever | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global client, retriever
    init_database(DATABASE_PATH, DOCUMENTS_SEED_PATH)
    client = OllamaClient()
    retriever = Retriever(client)
    yield
    await client.close()


app = FastAPI(
    title="CPU LLM Lab",
    version="0.3.1",
    lifespan=lifespan,
)


def get_client() -> OllamaClient:
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Cliente Ollama ainda não inicializado.",
        )
    return client


def get_retriever() -> Retriever:
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Retriever ainda não inicializado.",
        )
    return retriever


async def _installed_or_503() -> set[str]:
    ollama = get_client()
    if not await ollama.health():
        raise HTTPException(
            status_code=503,
            detail="Ollama não está acessível em localhost:11434.",
        )
    return await ollama.installed_models()


def _validate_generation_models(
    selected: list[str],
    installed: set[str],
) -> None:
    if not selected:
        raise HTTPException(
            status_code=400,
            detail="Selecione pelo menos um modelo instalado.",
        )
    missing = [model for model in selected if model not in installed]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Modelos não instalados: {', '.join(missing)}",
        )


def _validate_embedding_model(
    retrieval_modes: list[RetrievalMode],
    embedding_model: str,
    installed: set[str],
) -> None:
    needs_embedding = any(
        mode in {"embedding", "hybrid"}
        for mode in retrieval_modes
    )
    if needs_embedding and embedding_model not in installed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Modelo de embedding não instalado: {embedding_model}. "
                f"Execute: ollama pull {embedding_model}"
            ),
        )


@app.get("/api/health")
async def health():
    ollama = get_client()
    return {
        "app": "ok",
        "ollama": await ollama.health(),
        "database": DATABASE_PATH.exists(),
    }


@app.get("/api/models")
async def models():
    ollama = get_client()
    if not await ollama.health():
        return {
            "recommended": RECOMMENDED_MODELS,
            "installed": [],
            "ollama": False,
            "embedding": {
                "model": DEFAULT_EMBEDDING_MODEL,
                "installed": False,
            },
        }

    installed = await ollama.installed_models()
    generation_models = sorted(
        model
        for model in installed
        if model != DEFAULT_EMBEDDING_MODEL
    )
    return {
        "recommended": RECOMMENDED_MODELS,
        "installed": generation_models,
        "ollama": True,
        "embedding": {
            "model": DEFAULT_EMBEDDING_MODEL,
            "installed": DEFAULT_EMBEDDING_MODEL in installed,
        },
    }


@app.get("/api/documents")
async def documents():
    return [
        {
            "id": document.id,
            "title": document.title,
            "category": document.category,
        }
        for document in list_documents(DATABASE_PATH)
    ]


@app.post("/api/query", response_model=list[ModelResult])
async def query(payload: QueryRequest):
    ollama = get_client()
    searcher = get_retriever()
    installed = await _installed_or_503()
    _validate_generation_models(payload.models, installed)

    embedding_model = (
        payload.embedding_model
        or DEFAULT_EMBEDDING_MODEL
    )
    _validate_embedding_model(
        [payload.retrieval_mode],
        embedding_model,
        installed,
    )

    with traced(
        "api.query",
        retrieval_mode=payload.retrieval_mode,
        top_k=payload.top_k,
        model_count=len(payload.models),
    ):
        try:
            retrieved, retrieval_trace = await searcher.retrieve(
                DATABASE_PATH,
                payload.question,
                mode=payload.retrieval_mode,
                limit=payload.top_k,
                embedding_model=embedding_model,
            )
        except RuntimeError as exc:
            raise HTTPException(
                status_code=502,
                detail=str(exc),
            ) from exc

        results = []
        for model in payload.models:
            results.append(
                await ollama.run_grounded_query(
                    model,
                    payload.question,
                    retrieved,
                    retrieval=retrieval_trace,
                )
            )
        return results


@app.post(
    "/api/benchmark",
    response_model=BenchmarkResponse,
)
async def benchmark(
    models: list[str] | None = Query(default=None),
    retrieval_modes: list[RetrievalMode] | None = Query(default=None),
    top_k: int = Query(default=3, ge=1, le=10),
    embedding_model: str = Query(default=DEFAULT_EMBEDDING_MODEL),
    repetitions: int = Query(default=3, ge=1, le=10),
    order_seed: int = Query(default=42, ge=0, le=2_147_483_647),
):
    ollama = get_client()
    searcher = get_retriever()
    installed = await _installed_or_503()

    selected = models or []
    selected_modes = retrieval_modes or ["lexical"]
    _validate_generation_models(selected, installed)
    _validate_embedding_model(
        selected_modes,
        embedding_model,
        installed,
    )

    try:
        return await run_benchmark(
            ollama,
            searcher,
            selected,
            selected_modes,
            CASES_PATH,
            DATABASE_PATH,
            top_k=top_k,
            embedding_model=embedding_model,
            repetitions=repetitions,
            order_seed=order_seed,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


def run() -> None:
    uvicorn.run(
        "cpu_llm_lab.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


HTML = r'''<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CPU LLM Lab</title>
<style>
:root{font-family:Inter,system-ui,sans-serif;color-scheme:dark;background:#0c0f14;color:#edf2f7}
body{max-width:1280px;margin:auto;padding:28px}h1{margin-bottom:4px}.muted{color:#9aa4b2}.ok{color:#66e3a4}.bad{color:#ff7a90}
.panel,.card{background:#151a22;border:1px solid #2b3442;border-radius:14px;padding:18px;margin:16px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
textarea,input,select{box-sizing:border-box;width:100%;padding:11px;border-radius:8px;border:1px solid #394454;background:#0f141b;color:#fff}textarea{min-height:100px;resize:vertical}
button{background:#edf2f7;color:#111827;border:0;border-radius:9px;padding:11px 15px;font-weight:700;cursor:pointer;margin:10px 8px 0 0}button:disabled{opacity:.5;cursor:wait}
.models,.retrievers{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:8px;margin:12px 0}.model,.retriever{display:flex;gap:7px;align-items:center;border:1px solid #2b3442;border-radius:9px;padding:9px}.model input,.retriever input{width:auto}.disabled{opacity:.55}
.output{white-space:pre-wrap;background:#0f141b;padding:12px;border-radius:8px;min-height:70px}.metric{display:grid;grid-template-columns:1fr auto;gap:5px;font-size:14px}.pill{display:inline-block;border:1px solid #394454;border-radius:999px;padding:3px 8px;font-size:12px;margin:2px}
table{width:100%;border-collapse:collapse;overflow:auto}th,td{text-align:left;padding:8px;border-bottom:1px solid #2b3442;font-size:12px;white-space:nowrap}.docs{display:flex;flex-wrap:wrap;gap:7px}.docs span{border:1px solid #394454;border-radius:999px;padding:5px 9px;font-size:12px}
.control-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
</style>
</head>
<body>
<h1>CPU LLM Lab <small class="muted">v0.3.1</small></h1>
<div class="muted">Retrieval Lab com cache controlado, repetições, mediana, p95 e ordem experimental reproduzível.</div>
<div id="status" class="muted">Verificando ambiente…</div>

<section class="panel">
<h2>Base fictícia da Aurora Labs</h2>
<p class="muted">O SQLite é criado automaticamente a partir dos documentos versionados no repositório.</p>
<div class="docs" id="documents"></div>
</section>

<section class="panel">
<h2>Modelos para geração</h2>
<div class="models" id="models"></div>
<p id="embeddingStatus" class="muted"></p>
</section>

<section class="panel">
<h2>1. Consultar os documentos</h2>
<label for="question">Pergunta do funcionário</label>
<textarea id="question">A empresa ajuda financeiramente quem trabalha de casa?</textarea>
<div class="control-grid">
<div><label for="topk">Top-K</label><input id="topk" type="number" min="1" max="10" value="3"></div>
<div><label for="retrievalMode">Retriever</label>
<select id="retrievalMode">
<option value="lexical">Lexical — baseline 5/3/1</option>
<option value="embedding">Embedding — similaridade semântica</option>
<option value="hybrid">Hybrid — lexical + embedding</option>
</select></div>
</div>
<button id="queryBtn" onclick="queryModels()">Comparar resposta dos modelos</button>
<div class="grid" id="results"></div>
</section>

<section class="panel">
<h2>2. Benchmark</h2>
<p class="muted">A primeira repetição de cada retriever é tratada como cold. Para embedding/hybrid o cache de documentos é limpo antes dessa repetição; as demais são warm. A ordem de retrievers e modelos é embaralhada de forma reproduzível pela seed.</p>
<div class="retrievers" id="retrievers">
<label class="retriever"><input type="checkbox" value="lexical" checked> Lexical</label>
<label class="retriever" id="embeddingChoice"><input type="checkbox" value="embedding" checked> Embedding</label>
<label class="retriever" id="hybridChoice"><input type="checkbox" value="hybrid" checked> Hybrid</label>
</div>
<div class="control-grid">
<div><label for="repetitions">Repetições por caso</label><input id="repetitions" type="number" min="1" max="10" value="3"></div>
<div><label for="orderSeed">Seed da ordem</label><input id="orderSeed" type="number" min="0" max="2147483647" value="42"></div>
</div>
<button id="benchBtn" onclick="runBenchmark()">Rodar benchmark</button>
<div id="benchmark"></div>
</section>

<script>
let installedModels=[];
let embeddingModel='embeddinggemma:latest';
let embeddingReady=false;

function esc(s){
 return String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))
}
function selectedModels(){
 return [...document.querySelectorAll('#models input:checked:not(:disabled)')].map(x=>x.value)
}
function selectedRetrievers(){
 return [...document.querySelectorAll('#retrievers input:checked:not(:disabled)')].map(x=>x.value)
}
function selectedDefault(name){
 return ['gemma3:270m','smollm2:360m','qwen3:0.6b','gemma3:1b','llama3.2:1b'].includes(name)
}
async function init(){
 try{
  const [mr,dr]=await Promise.all([fetch('/api/models'),fetch('/api/documents')]);
  const m=await mr.json(),docs=await dr.json();
  installedModels=m.installed||[];
  embeddingModel=(m.embedding||{}).model||'embeddinggemma:latest';
  embeddingReady=Boolean((m.embedding||{}).installed);
  document.getElementById('status').innerHTML=m.ollama?'<span class="ok">● Ollama online</span> · banco carregado':'<span class="bad">● Ollama offline</span> · banco carregado';
  document.getElementById('embeddingStatus').innerHTML=embeddingReady
    ? `<span class="ok">Embedding:</span> ${esc(embeddingModel)} instalado`
    : `<span class="bad">Embedding:</span> ${esc(embeddingModel)} não instalado — execute <code>ollama pull ${esc(embeddingModel)}</code>`;
  document.getElementById('documents').innerHTML=docs.map(d=>`<span>#${d.id} ${esc(d.title)}</span>`).join('');
  const all=[...new Set([...(m.recommended||[]),...installedModels])];
  document.getElementById('models').innerHTML=all.map(name=>{
    const ok=installedModels.includes(name);
    return `<label class="model ${ok?'':'disabled'}"><input type="checkbox" value="${esc(name)}" ${ok?'':'disabled'} ${ok&&selectedDefault(name)?'checked':''}> <span>${esc(name)}<br><small class="${ok?'ok':'bad'}">${ok?'instalado':'não instalado'}</small></span></label>`
  }).join('');

  for(const id of ['embeddingChoice','hybridChoice']){
    const el=document.getElementById(id);
    const input=el.querySelector('input');
    input.disabled=!embeddingReady;
    input.checked=embeddingReady;
    el.classList.toggle('disabled',!embeddingReady);
  }
  for(const option of document.querySelectorAll('#retrievalMode option')){
    if(option.value!=='lexical') option.disabled=!embeddingReady;
  }
 }catch(e){
  document.getElementById('status').innerHTML='<span class="bad">Falha ao carregar aplicação</span>'
 }
}
async function queryModels(){
 const models=selectedModels();
 if(!models.length){alert('Selecione pelo menos um modelo instalado.');return}
 const btn=document.getElementById('queryBtn');
 btn.disabled=true;
 btn.textContent='Consultando…';
 document.getElementById('results').innerHTML='';
 const payload={
   question:document.getElementById('question').value,
   models,
   top_k:Number(document.getElementById('topk').value),
   retrieval_mode:document.getElementById('retrievalMode').value,
   embedding_model:embeddingModel
 };
 try{
  const r=await fetch('/api/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const d=await r.json();
  if(!r.ok)throw new Error(d.detail||'Erro');
  document.getElementById('results').innerHTML=d.map(card).join('')
 }catch(e){
  document.getElementById('results').innerHTML=`<div class="bad">${esc(e.message)}</div>`
 }
 btn.disabled=false;
 btn.textContent='Comparar resposta dos modelos';
}
function card(x){
 if(!x.ok)return `<article class="card"><h3>${esc(x.model)}</h3><div class="bad">${esc(x.error)}</div></article>`;
 const a=x.answer,m=x.metrics,r=x.retrieval,docs=x.retrieved_documents||[];
 const ranking=(r?.ranked_document_ids||[]).map((id,i)=>`#${id} (${r.ranked_scores[i]})`).join(' · ');
 const cpu=m.cpu_only_verified===true?'sim':m.cpu_only_verified===false?'não':'desconhecido';
 return `<article class="card"><h3>${esc(x.model)}</h3><div class="output">${esc(a.resposta)}</div><p><strong>Retriever:</strong> ${esc(r?.mode||'-')} · ${r?.latency_ms??0} ms<br><strong>Embedding:</strong> ${r?.embedding_ms??0} ms<br><strong>Ranking:</strong> ${esc(ranking||'nenhum')}<br><strong>Encontrado:</strong> ${a.encontrado?'sim':'não'}<br><strong>Fontes:</strong> ${(a.fontes||[]).map(id=>'#'+id).join(', ')||'nenhuma'}<br><strong>Contexto:</strong> ${docs.map(d=>'#'+d.id+' '+esc(d.title)).join(' · ')||'nenhum'}<br><strong>Trace:</strong> ${esc(x.trace_id||'-')}</p><div class="metric"><span>Tempo LLM</span><b>${m.total_ms} ms</b><span>Tokens/s</span><b>${m.tokens_per_second}</b><span>Tokens entrada</span><b>${m.prompt_tokens}</b><span>Tokens saída</span><b>${m.output_tokens}</b><span>CPU verificada</span><b>${cpu}</b></div></article>`
}
async function runBenchmark(){
 const models=selectedModels();
 const modes=selectedRetrievers();
 const repetitions=Number(document.getElementById('repetitions').value);
 const orderSeed=Number(document.getElementById('orderSeed').value);
 if(!models.length){alert('Selecione pelo menos um modelo instalado.');return}
 if(!modes.length){alert('Selecione pelo menos um retriever.');return}
 const btn=document.getElementById('benchBtn');
 btn.disabled=true;
 btn.textContent='Executando benchmark…';
 const perCase=models.length*modes.length*repetitions;
 document.getElementById('benchmark').innerHTML=`<p class="muted">Executando ${repetitions} repetição(ões). Cada caso fará ${perCase} inferência(s) de LLM.</p>`;
 try{
  const qs=[
    ...models.map(m=>'models='+encodeURIComponent(m)),
    ...modes.map(m=>'retrieval_modes='+encodeURIComponent(m)),
    'top_k='+encodeURIComponent(document.getElementById('topk').value),
    'embedding_model='+encodeURIComponent(embeddingModel),
    'repetitions='+encodeURIComponent(repetitions),
    'order_seed='+encodeURIComponent(orderSeed)
  ].join('&');
  const r=await fetch('/api/benchmark?'+qs,{method:'POST'});
  const d=await r.json();
  if(!r.ok)throw new Error(d.detail||'Erro');
  const cpu=x=>x===true?'sim':x===false?'não':'?';
  document.getElementById('benchmark').innerHTML=`<p class="muted">RAM: ${d.environment.ram_gb} GB · ${d.environment.logical_cpus} threads · top_k=${d.environment.top_k} · repetições=${d.environment.repetitions} · seed=${d.environment.order_seed} · inferências planejadas=${d.environment.planned_llm_executions}</p><div style="overflow:auto"><table><thead><tr><th>Retriever</th><th>Modelo</th><th>R@1</th><th>R@3</th><th>R@5</th><th>MRR</th><th>Ret avg</th><th>Ret med</th><th>Ret p95</th><th>Cold ret</th><th>Warm ret</th><th>Emb cold</th><th>Emb warm</th><th>Fatos</th><th>Hit Top-K</th><th>Fontes</th><th>Sem alteração conhecida</th><th>Recusa</th><th>LLM avg</th><th>LLM med</th><th>LLM p95</th><th>tokens/s</th><th>CPU</th><th>Erros</th></tr></thead><tbody>${d.rows.map(x=>`<tr><td>${esc(x.retrieval_mode)}</td><td>${esc(x.model)}</td><td>${x.recall_at_1}%</td><td>${x.recall_at_3}%</td><td>${x.recall_at_5}%</td><td>${x.mrr}</td><td>${x.avg_retrieval_ms} ms</td><td>${x.median_retrieval_ms} ms</td><td>${x.p95_retrieval_ms} ms</td><td>${x.avg_cold_retrieval_ms} ms</td><td>${x.avg_warm_retrieval_ms} ms</td><td>${x.avg_cold_embedding_ms} ms</td><td>${x.avg_warm_embedding_ms} ms</td><td>${x.avg_factual_score}%</td><td>${x.retrieval_hit_rate}%</td><td>${x.source_accuracy_rate}%</td><td>${x.hallucination_free_rate}%</td><td>${x.abstention_accuracy}%</td><td>${x.avg_total_ms} ms</td><td>${x.median_total_ms} ms</td><td>${x.p95_total_ms} ms</td><td>${x.avg_tokens_per_second}</td><td>${cpu(x.cpu_only_all_runs)}</td><td>${x.errors}</td></tr>`).join('')}</tbody></table></div>`
 }catch(e){
  document.getElementById('benchmark').innerHTML=`<p class="bad">${esc(e.message)}</p>`
 }
 btn.disabled=false;
 btn.textContent='Rodar benchmark';
}
init();
</script>
</body>
</html>'''
