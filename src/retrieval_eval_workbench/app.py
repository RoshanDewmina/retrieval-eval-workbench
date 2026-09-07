from __future__ import annotations

import json
import threading
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .answering import answer_query
from .data import ROOT, load_documents, load_frozen_settings
from .retrieval import LexicalRetriever, Retriever, SemanticRetriever


app = FastAPI(title="Retrieval Eval Workbench", version="0.2.0")
_semantic_initialization_lock = threading.Lock()


class QueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    strategy: str = Field(default="semantic", pattern="^(lexical|semantic)$")
    top_k: int | None = Field(default=None, ge=1, le=5)


@lru_cache(maxsize=1)
def _documents():
    return load_documents()


@lru_cache(maxsize=1)
def _settings() -> dict:
    return load_frozen_settings()


@lru_cache(maxsize=1)
def _lexical() -> Retriever:
    return LexicalRetriever(_documents())


@lru_cache(maxsize=1)
def _semantic_initialized() -> Retriever:
    return SemanticRetriever(_documents())


def _semantic() -> Retriever:
    """Single-flight admission: concurrent cold requests share one model construction."""
    with _semantic_initialization_lock:
        return _semantic_initialized()


def _retriever(strategy: str) -> Retriever:
    try:
        return _lexical() if strategy == "lexical" else _semantic()
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"{strategy} retriever unavailable: {error}") from error


def _effective_settings(strategy: str, requested_top_k: int | None) -> dict:
    frozen = _settings()
    configured = frozen["retrievers"][strategy]
    return {
        "strategy": strategy,
        "retriever": configured["name"],
        "minimum_score": configured["minimum_score"],
        "top_k": requested_top_k if requested_top_k is not None else frozen["top_k"],
        "frozen_config_fingerprint": frozen["fingerprint"],
        "question_set_version": frozen["question_set_version"],
    }


@app.get("/health")
def health() -> dict:
    try:
        _lexical()
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"lexical retriever unavailable: {error}") from error
    return {"status": "ok", "service": "retrieval-eval-workbench", "version": "0.2.0"}


@app.post("/api/query")
def query(payload: QueryRequest) -> dict:
    settings = _effective_settings(payload.strategy, payload.top_k)
    response = answer_query(_retriever(payload.strategy), payload.query, limit=settings["top_k"], minimum_score=settings["minimum_score"])
    return {
        "strategy": payload.strategy,
        "effective_settings": settings,
        "answer": response.answer,
        "answerable": response.answerable,
        "citations": [citation.__dict__ for citation in response.citations],
        "hits": [{"document_id": hit.document.id, "title": hit.document.title, "score": round(hit.score, 4)} for hit in response.hits],
    }


@app.get("/api/results")
def results() -> dict:
    path = ROOT / "evidence" / "benchmarks" / "latest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No historical benchmark receipt is available.")
    return json.loads(path.read_text())


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>Retrieval Eval Workbench</title>
<style>body{max-width:900px;margin:3rem auto;padding:0 1rem;background:#101827;color:#e7edf7;font:16px system-ui}.grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}textarea,select,button{font:inherit;padding:.65rem;border-radius:6px}textarea{width:100%;box-sizing:border-box}.card{background:#192337;padding:1rem;border-radius:8px;margin-top:1rem}.muted{color:#b3c1d5}button{background:#1d8cc8;color:#fff;border:0;margin-top:.6rem}button:disabled{opacity:.6}@media(max-width:600px){body{margin:1rem auto}.grid{grid-template-columns:1fr}}</style></head><body>
<h1>Retrieval Eval Workbench</h1><p class='muted'>Compare TF-IDF with a pinned local MiniLM encoder. Answers are extractive corpus quotes with physical line citations.</p>
<div class='grid'><label>Retriever<select id='strategy'><option value='semantic'>Semantic</option><option value='lexical'>Lexical TF-IDF</option></select></label><label>Top K (optional override)<select id='topk'><option value=''>Frozen default</option><option value='1'>1</option><option value='3'>3</option><option value='5'>5</option></select></label></div>
<p><label for='query'>Question</label><textarea id='query' rows='3'>What happens to a case if no routing rule matches?</textarea><br><button id='run'>Run query</button> <button id='results'>Show historical benchmark</button></p><div id='output' class='card' role='status' aria-live='polite'>Run a query to inspect retrieved documents, answer, and line-level citation.</div>
<script>const out=document.querySelector('#output'),runButton=document.querySelector('#run');function node(tag,text){const x=document.createElement(tag);x.textContent=text;return x}function clear(){out.replaceChildren()}function list(title,items){out.append(node('h3',title));const l=document.createElement('ul');items.forEach(i=>l.append(node('li',i)));out.append(l)}async function run(){runButton.disabled=true;clear();out.append(node('p','Running query…'));try{const top=document.querySelector('#topk').value;const body={query:document.querySelector('#query').value,strategy:document.querySelector('#strategy').value};if(top)body.top_k=Number(top);const r=await fetch('/api/query',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)}),d=await r.json();clear();if(!r.ok){out.append(node('p',d.detail||'Query failed.'));return}out.append(node('h2',d.answerable?'Grounded extract':'No supported answer'));out.append(node('p',d.answer));list('Citations',d.citations.length?d.citations.map(x=>`${x.title}, line ${x.line_start}: “${x.quote}”`):['No citation.']);list('Retrieved documents',d.hits.map(x=>`${x.title} (${x.score})`));out.append(node('p',`Settings: ${d.effective_settings.retriever}; top K ${d.effective_settings.top_k}; floor ${d.effective_settings.minimum_score}.`))}catch(e){clear();out.append(node('p',`Network error: ${e.message}`))}finally{runButton.disabled=false}}async function showResults(){clear();out.append(node('p','Loading historical receipt…'));try{const r=await fetch('/api/results'),d=await r.json();clear();if(!r.ok){out.append(node('p',d.detail||'No receipt.'));return}out.append(node('h2','Historical development benchmark'));out.append(node('p','This receipt predates frozen model provenance and is not an independent held-out result.'));list('Recorded retrievers',Object.keys(d.measured_results.results).map(x=>x))}catch(e){clear();out.append(node('p',`Network error: ${e.message}`))}}document.querySelector('#run').onclick=run;document.querySelector('#results').onclick=showResults;</script></body></html>"""
