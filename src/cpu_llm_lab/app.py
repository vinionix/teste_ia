from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from .benchmark import run_benchmark
from .config import (
    CASES_PATH,
    DATABASE_PATH,
    DOCUMENTS_SEED_PATH,
    RECOMMENDED_MODELS,
)
from .database import init_database, list_documents, search_documents
from .ollama_client import OllamaClient
from .schemas import BenchmarkResponse, ModelResult, QueryRequest

client: OllamaClient | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global client
    init_database(DATABASE_PATH, DOCUMENTS_SEED_PATH)
    client = OllamaClient()
    yield
    await client.close()


app = FastAPI(title="CPU LLM Lab", version="0.2.0", lifespan=lifespan)


def get_client() -> OllamaClient:
    if client is None:
        raise HTTPException(status_code=503, detail="Cliente Ollama ainda não inicializado.")
    return client


@app.get("/api/health")
async def health():
    ollama = get_client()
    return {"app": "ok", "ollama": await ollama.health(), "database": DATABASE_PATH.exists()}


@app.get("/api/models")
async def models():
    ollama = get_client()
    if not await ollama.health():
        return {"recommended": RECOMMENDED_MODELS, "installed": [], "ollama": False}
    installed = sorted(await ollama.installed_models())
    return {"recommended": RECOMMENDED_MODELS, "installed": installed, "ollama": True}


@app.get("/api/documents")
async def documents():
    return [
        {"id": document.id, "title": document.title, "category": document.category}
        for document in list_documents(DATABASE_PATH)
    ]


@app.post("/api/query", response_model=list[ModelResult])
async def query(payload: QueryRequest):
    ollama = get_client()
    if not await ollama.health():
        raise HTTPException(status_code=503, detail="Ollama não está acessível em localhost:11434.")

    selected = payload.models
    if not selected:
        raise HTTPException(status_code=400, detail="Selecione pelo menos um modelo instalado.")

    installed = await ollama.installed_models()
    missing = [model for model in selected if model not in installed]
    if missing:
        raise HTTPException(status_code=400, detail=f"Modelos não instalados: {', '.join(missing)}")

    retrieved = search_documents(DATABASE_PATH, payload.question, limit=payload.top_k)
    results = []
    for model in selected:
        results.append(await ollama.run_grounded_query(model, payload.question, retrieved))
    return results


@app.post("/api/benchmark", response_model=BenchmarkResponse)
async def benchmark(
    models: list[str] | None = Query(default=None),
    top_k: int = Query(default=3, ge=1, le=10),
):
    ollama = get_client()
    if not await ollama.health():
        raise HTTPException(status_code=503, detail="Ollama não está acessível em localhost:11434.")

    selected = models or []
    if not selected:
        raise HTTPException(status_code=400, detail="Selecione pelo menos um modelo instalado.")

    installed = await ollama.installed_models()
    missing = [model for model in selected if model not in installed]
    if missing:
        raise HTTPException(status_code=400, detail=f"Modelos não instalados: {', '.join(missing)}")

    return await run_benchmark(
        ollama,
        selected,
        CASES_PATH,
        DATABASE_PATH,
        top_k=top_k,
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


def run() -> None:
    uvicorn.run("cpu_llm_lab.app:app", host="127.0.0.1", port=8000, reload=False)


HTML = r'''<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CPU LLM Lab</title>
<style>
:root{font-family:Inter,system-ui,sans-serif;color-scheme:dark;background:#0c0f14;color:#edf2f7}
body{max-width:1200px;margin:auto;padding:28px}h1{margin-bottom:4px}.muted{color:#9aa4b2}.ok{color:#66e3a4}.bad{color:#ff7a90}
.panel,.card{background:#151a22;border:1px solid #2b3442;border-radius:14px;padding:18px;margin:16px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
textarea,input{box-sizing:border-box;width:100%;padding:11px;border-radius:8px;border:1px solid #394454;background:#0f141b;color:#fff}textarea{min-height:100px;resize:vertical}
button{background:#edf2f7;color:#111827;border:0;border-radius:9px;padding:11px 15px;font-weight:700;cursor:pointer;margin:10px 8px 0 0}button:disabled{opacity:.5;cursor:wait}
.models{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:8px;margin:12px 0}.model{display:flex;gap:7px;align-items:center;border:1px solid #2b3442;border-radius:9px;padding:9px}.model input{width:auto}.model.disabled{opacity:.55}
.output{white-space:pre-wrap;background:#0f141b;padding:12px;border-radius:8px;min-height:70px}.metric{display:grid;grid-template-columns:1fr auto;gap:5px;font-size:14px}.pill{display:inline-block;border:1px solid #394454;border-radius:999px;padding:3px 8px;font-size:12px;margin:2px}
table{width:100%;border-collapse:collapse;overflow:auto}th,td{text-align:left;padding:8px;border-bottom:1px solid #2b3442;font-size:12px}.docs{display:flex;flex-wrap:wrap;gap:7px}.docs span{border:1px solid #394454;border-radius:999px;padding:5px 9px;font-size:12px}
</style>
</head>
<body>
<h1>CPU LLM Lab</h1>
<div class="muted">Benchmark de modelos locais baratos para consultas corporativas fundamentadas em documentos — CPU-only.</div>
<div id="status" class="muted">Verificando ambiente…</div>

<section class="panel">
<h2>Base fictícia da Aurora Labs</h2>
<p class="muted">O SQLite é criado automaticamente a partir dos documentos versionados no repositório.</p>
<div class="docs" id="documents"></div>
</section>

<section class="panel">
<h2>Modelos para teste</h2>
<p class="muted">Todos os modelos já instalados no seu Ollama aparecem aqui. Os modelos recomendados que ainda não foram baixados ficam visíveis, mas desabilitados.</p>
<div class="models" id="models"></div>
</section>

<section class="panel">
<h2>1. Consultar os documentos</h2>
<label for="question">Pergunta do funcionário</label>
<textarea id="question">Com quanta antecedência eu preciso pedir minhas férias e posso dividir o período?</textarea>
<div style="max-width:180px"><label for="topk">Documentos recuperados</label><input id="topk" type="number" min="1" max="10" value="3"></div>
<button id="queryBtn" onclick="queryModels()">Comparar resposta dos modelos</button>
<div class="grid" id="results"></div>
</section>

<section class="panel">
<h2>2. Benchmark de fidelidade</h2>
<p class="muted">Executa perguntas com fatos conhecidos e casos sem resposta para medir preservação, fontes, recusa, latência e tokens/s.</p>
<button id="benchBtn" onclick="runBenchmark()">Rodar benchmark</button>
<div id="benchmark"></div>
</section>

<script>
let installedModels=[];
function esc(s){return String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function selectedModels(){return [...document.querySelectorAll('#models input:checked:not(:disabled)')].map(x=>x.value)}
async function init(){
 try{
  const [mr,dr]=await Promise.all([fetch('/api/models'),fetch('/api/documents')]);
  const m=await mr.json(),docs=await dr.json();installedModels=m.installed||[];
  document.getElementById('status').innerHTML=m.ollama?'<span class="ok">● Ollama online</span> · banco carregado':'<span class="bad">● Ollama offline</span> · banco carregado';
  document.getElementById('documents').innerHTML=docs.map(d=>`<span>#${d.id} ${esc(d.title)}</span>`).join('');
  const all=[...new Set([...(m.recommended||[]),...installedModels])];
  document.getElementById('models').innerHTML=all.map(name=>{const ok=installedModels.includes(name);return `<label class="model ${ok?'':'disabled'}"><input type="checkbox" value="${esc(name)}" ${ok?'':'disabled'} ${ok&&selectedDefault(name)?'checked':''}> <span>${esc(name)}<br><small class="${ok?'ok':'bad'}">${ok?'instalado':'não instalado'}</small></span></label>`}).join('');
 }catch(e){document.getElementById('status').innerHTML='<span class="bad">Falha ao carregar aplicação</span>'}
}
function selectedDefault(name){return ['gemma3:270m','smollm2:360m','qwen3:0.6b','gemma3:1b','llama3.2:1b'].includes(name)}
async function queryModels(){
 const models=selectedModels();if(!models.length){alert('Selecione pelo menos um modelo instalado.');return}
 const btn=document.getElementById('queryBtn');btn.disabled=true;btn.textContent='Consultando…';document.getElementById('results').innerHTML='';
 const payload={question:document.getElementById('question').value,models,top_k:Number(document.getElementById('topk').value)};
 try{const r=await fetch('/api/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Erro');document.getElementById('results').innerHTML=d.map(card).join('')}
 catch(e){document.getElementById('results').innerHTML=`<div class="bad">${esc(e.message)}</div>`}
 btn.disabled=false;btn.textContent='Comparar resposta dos modelos';
}
function card(x){
 if(!x.ok)return `<article class="card"><h3>${esc(x.model)}</h3><div class="bad">${esc(x.error)}</div></article>`;
 const a=x.answer,m=x.metrics,docs=x.retrieved_documents||[];
 return `<article class="card"><h3>${esc(x.model)}</h3><div class="output">${esc(a.resposta)}</div><p><strong>Encontrado:</strong> ${a.encontrado?'sim':'não'}<br><strong>Fontes declaradas:</strong> ${(a.fontes||[]).map(id=>'#'+id).join(', ')||'nenhuma'}<br><strong>Contexto recuperado:</strong> ${docs.map(d=>'#'+d.id+' '+esc(d.title)).join(' · ')||'nenhum'}</p><div class="metric"><span>Tempo total</span><b>${m.total_ms} ms</b><span>Tokens/s</span><b>${m.tokens_per_second}</b><span>Tokens entrada</span><b>${m.prompt_tokens}</b><span>Tokens saída</span><b>${m.output_tokens}</b><span>VRAM</span><b class="${m.cpu_only_verified?'ok':'bad'}">${m.vram_bytes===0?'0 — CPU OK':m.vram_bytes}</b></div></article>`
}
async function runBenchmark(){
 const models=selectedModels();if(!models.length){alert('Selecione pelo menos um modelo instalado.');return}
 const btn=document.getElementById('benchBtn');btn.disabled=true;btn.textContent='Executando benchmark…';document.getElementById('benchmark').innerHTML='<p class="muted">Executando casos sequencialmente…</p>';
 try{const qs=[...models.map(m=>'models='+encodeURIComponent(m)),'top_k='+encodeURIComponent(document.getElementById('topk').value)].join('&');const r=await fetch('/api/benchmark?'+qs,{method:'POST'});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Erro');document.getElementById('benchmark').innerHTML=`<p class="muted">RAM: ${d.environment.ram_gb} GB · ${d.environment.logical_cpus} threads · top_k=${d.environment.top_k}</p><div style="overflow:auto"><table><thead><tr><th>Modelo</th><th>Fatos</th><th>Retrieval</th><th>Fontes</th><th>Sem alteração conhecida</th><th>Recusa correta</th><th>Tempo</th><th>tokens/s</th><th>CPU</th><th>Erros</th></tr></thead><tbody>${d.rows.map(x=>`<tr><td>${esc(x.model)}</td><td>${x.avg_factual_score}%</td><td>${x.retrieval_hit_rate}%</td><td>${x.source_accuracy_rate}%</td><td>${x.hallucination_free_rate}%</td><td>${x.abstention_accuracy}%</td><td>${x.avg_total_ms} ms</td><td>${x.avg_tokens_per_second}</td><td>${x.cpu_only_all_runs?'sim':'não'}</td><td>${x.errors}</td></tr>`).join('')}</tbody></table></div>`}
 catch(e){document.getElementById('benchmark').innerHTML=`<p class="bad">${esc(e.message)}</p>`}
 btn.disabled=false;btn.textContent='Rodar benchmark';
}
init();
</script>
</body></html>'''
