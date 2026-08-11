from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .benchmark import run_benchmark
from .config import DEFAULT_MODELS
from .ollama_client import OllamaClient
from .schemas import BenchmarkResponse, CompareRequest, ModelResult

BASE_DIR = Path(__file__).resolve().parents[2]
CASES_PATH = BASE_DIR / "data" / "test_cases.json"

client: OllamaClient | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global client
    client = OllamaClient()
    yield
    await client.close()


app = FastAPI(title="CPU LLM Lab", version="0.1.0", lifespan=lifespan)


def get_client() -> OllamaClient:
    if client is None:
        raise HTTPException(status_code=503, detail="Cliente Ollama ainda não inicializado.")
    return client


@app.get("/api/health")
async def health():
    ollama = get_client()
    return {"app": "ok", "ollama": await ollama.health()}


@app.get("/api/models")
async def models():
    ollama = get_client()
    if not await ollama.health():
        return {"recommended": DEFAULT_MODELS, "installed": [], "ollama": False}
    installed = sorted(await ollama.installed_models())
    return {"recommended": DEFAULT_MODELS, "installed": installed, "ollama": True}


@app.post("/api/compare", response_model=list[ModelResult])
async def compare(payload: CompareRequest):
    ollama = get_client()
    if not await ollama.health():
        raise HTTPException(status_code=503, detail="Ollama não está acessível em localhost:11434.")
    selected = payload.models or DEFAULT_MODELS
    installed = await ollama.installed_models()
    missing = [m for m in selected if m not in installed]
    if missing:
        raise HTTPException(status_code=400, detail=f"Modelos não instalados: {', '.join(missing)}")

    results = []
    for model in selected:
        results.append(await ollama.run_model(model, payload.record))
    return results


@app.post("/api/benchmark", response_model=BenchmarkResponse)
async def benchmark(models: list[str] | None = None):
    ollama = get_client()
    if not await ollama.health():
        raise HTTPException(status_code=503, detail="Ollama não está acessível em localhost:11434.")
    selected = models or DEFAULT_MODELS
    installed = await ollama.installed_models()
    missing = [m for m in selected if m not in installed]
    if missing:
        raise HTTPException(status_code=400, detail=f"Modelos não instalados: {', '.join(missing)}")
    return await run_benchmark(ollama, selected, CASES_PATH)


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
body{max-width:1180px;margin:auto;padding:28px} h1{margin-bottom:4px} .muted{color:#9aa4b2}
.panel,.card{background:#151a22;border:1px solid #2b3442;border-radius:14px;padding:18px;margin:16px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:14px}
label{display:block;font-size:13px;color:#b8c1cc;margin-top:10px} input,select{box-sizing:border-box;width:100%;padding:10px;border-radius:8px;border:1px solid #394454;background:#0f141b;color:#fff}
button{background:#edf2f7;color:#111827;border:0;border-radius:9px;padding:11px 15px;font-weight:700;cursor:pointer;margin-top:14px} button:disabled{opacity:.5;cursor:wait}
.models{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0}.models label{display:flex;gap:6px;align-items:center;margin:0}.models input{width:auto}
.ok{color:#66e3a4}.bad{color:#ff7a90}.metric{display:grid;grid-template-columns:1fr auto;gap:5px;font-size:14px}.output{white-space:pre-wrap;background:#0f141b;padding:12px;border-radius:8px;min-height:80px}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:9px;border-bottom:1px solid #2b3442;font-size:13px}
#status{margin-top:12px}.pill{display:inline-block;border:1px solid #394454;border-radius:999px;padding:3px 8px;font-size:12px;margin-right:5px}
</style>
</head>
<body>
<h1>CPU LLM Lab</h1>
<div class="muted">Compare modelos locais pequenos para formatação de texto — sem GPU.</div>
<div id="status" class="muted">Verificando Ollama…</div>

<section class="panel">
<h2>1. Comparar uma entrada</h2>
<div class="grid">
<div><label>Cliente</label><input id="cliente" value="João Silva"></div>
<div><label>Plano</label><input id="plano" value="Premium"></div>
<div><label>Valor</label><input id="valor" type="number" step="0.01" value="149.90"></div>
<div><label>Status</label><select id="situacao"><option>pagamento_pendente</option><option>pagamento_confirmado</option><option>suporte_solicitado</option></select></div>
<div><label>Vencimento</label><input id="vencimento" value="15/08/2026"></div>
</div>
<div class="models" id="models"></div>
<button id="compareBtn" onclick="compareModels()">Comparar modelos</button>
<div class="grid" id="results"></div>
</section>

<section class="panel">
<h2>2. Benchmark</h2>
<p class="muted">Executa 5 casos nos modelos selecionados e agrega fidelidade, schema, latência e tokens/s.</p>
<button id="benchBtn" onclick="runBenchmark()">Rodar benchmark</button>
<div id="benchmark"></div>
</section>

<script>
const recommended=['qwen3:0.6b','gemma3:1b','qwen3:1.7b'];
async function init(){
 try{
  const r=await fetch('/api/models'); const d=await r.json();
  document.getElementById('status').innerHTML=d.ollama?'<span class="ok">● Ollama online</span>':'<span class="bad">● Ollama offline</span>';
  document.getElementById('models').innerHTML=recommended.map(m=>`<label><input type="checkbox" value="${m}" ${d.installed.includes(m)?'checked':''}> ${m} ${d.installed.includes(m)?'<span class="ok">instalado</span>':'<span class="bad">ausente</span>'}</label>`).join('');
 }catch(e){document.getElementById('status').innerHTML='<span class="bad">Falha ao consultar backend</span>'}
}
function selectedModels(){return [...document.querySelectorAll('#models input:checked')].map(x=>x.value)}
function esc(s){return String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
async function compareModels(){
 const btn=document.getElementById('compareBtn');btn.disabled=true;btn.textContent='Executando…';document.getElementById('results').innerHTML='';
 const payload={record:{cliente:cliente.value,plano:plano.value,valor:Number(valor.value),status:situacao.value,vencimento:vencimento.value},models:selectedModels()};
 try{
  const r=await fetch('/api/compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}); const d=await r.json();
  if(!r.ok) throw new Error(d.detail||'Erro');
  document.getElementById('results').innerHTML=d.map(card).join('');
 }catch(e){document.getElementById('results').innerHTML=`<div class="bad">${esc(e.message)}</div>`}
 btn.disabled=false;btn.textContent='Comparar modelos';
}
function card(x){
 if(!x.ok)return `<article class="card"><h3>${esc(x.model)}</h3><div class="bad">${esc(x.error)}</div></article>`;
 const m=x.metrics,e=x.evaluation,o=x.output;
 return `<article class="card"><h3>${esc(x.model)}</h3><div class="output"><strong>${esc(o.titulo)}</strong><br><br>${esc(o.mensagem)}<br><br><span class="pill">${esc(o.categoria)}</span></div><p><strong>Fidelidade:</strong> ${e.fidelity_score}%</p><div class="metric"><span>Tempo total</span><b>${m.total_ms} ms</b><span>Tokens/s</span><b>${m.tokens_per_second}</b><span>Tokens saída</span><b>${m.output_tokens}</b><span>VRAM</span><b class="${m.cpu_only_verified?'ok':'bad'}">${m.vram_bytes===0?'0 — CPU OK':m.vram_bytes}</b></div>${e.notes.length?`<p class="bad">${e.notes.map(esc).join('<br>')}</p>`:''}</article>`
}
async function runBenchmark(){
 const btn=document.getElementById('benchBtn');btn.disabled=true;btn.textContent='Executando benchmark…';document.getElementById('benchmark').innerHTML='<p class="muted">Processando casos sequencialmente…</p>';
 try{
  const models=selectedModels();const qs=models.map(m=>'models='+encodeURIComponent(m)).join('&');
  const r=await fetch('/api/benchmark'+(qs?'?'+qs:''),{method:'POST'}); const d=await r.json();if(!r.ok)throw new Error(d.detail||'Erro');
  document.getElementById('benchmark').innerHTML=`<p class="muted">CPU: ${esc(d.environment.processor||'não identificado')} · RAM: ${d.environment.ram_gb} GB · ${d.environment.logical_cpus} threads</p><table><thead><tr><th>Modelo</th><th>Schema</th><th>Fidelidade</th><th>Tempo médio</th><th>tokens/s</th><th>CPU-only</th><th>Erros</th></tr></thead><tbody>${d.rows.map(x=>`<tr><td>${esc(x.model)}</td><td>${x.schema_success_rate}%</td><td>${x.avg_fidelity}%</td><td>${x.avg_total_ms} ms</td><td>${x.avg_tokens_per_second}</td><td class="${x.cpu_only_all_runs?'ok':'bad'}">${x.cpu_only_all_runs?'sim':'não verificado'}</td><td>${x.errors}</td></tr>`).join('')}</tbody></table>`;
 }catch(e){document.getElementById('benchmark').innerHTML=`<p class="bad">${esc(e.message)}</p>`}
 btn.disabled=false;btn.textContent='Rodar benchmark';
}
init();
</script>
</body></html>'''
