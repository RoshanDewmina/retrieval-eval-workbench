from __future__ import annotations

import json
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .answering import answer_query
from .data import ROOT, load_documents
from .retrieval import LexicalRetriever, Retriever, SemanticRetriever


app = FastAPI(title="Retrieval Eval Workbench", version="0.1.0")


class QueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    strategy: str = Field(default="semantic", pattern="^(lexical|semantic)$")
    top_k: int = Field(default=3, ge=1, le=5)


@lru_cache(maxsize=1)
def _documents():
    return load_documents()


@lru_cache(maxsize=1)
def _lexical() -> Retriever:
    return LexicalRetriever(_documents())


@lru_cache(maxsize=1)
def _semantic() -> Retriever:
    return SemanticRetriever(_documents())


def _retriever(strategy: str) -> Retriever:
    try:
        return _lexical() if strategy == "lexical" else _semantic()
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"{strategy} retriever unavailable: {error}") from error


@app.get("/health")
def health() -> dict:
    try:
        _lexical()
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"lexical retriever unavailable: {error}") from error
    return {"status": "ok", "service": "retrieval-eval-workbench", "version": "0.1.0"}


@app.post("/api/query")
def query(payload: QueryRequest) -> dict:
    response = answer_query(_retriever(payload.strategy), payload.query, limit=payload.top_k)
    return {
        "strategy": payload.strategy,
        "answer": response.answer,
        "answerable": response.answerable,
        "citations": [citation.__dict__ for citation in response.citations],
        "hits": [{"document_id": hit.document.id, "title": hit.document.title, "score": round(hit.score, 4)} for hit in response.hits],
    }


@app.get("/api/results")
def results() -> dict:
    path = ROOT / "evidence" / "benchmarks" / "latest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No benchmark receipt has been generated. Run make benchmark.")
    return json.loads(path.read_text())


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html><html><head><meta charset='utf-8'><title>Retrieval Eval Workbench</title>
<style>body{max-width:900px;margin:3rem auto;padding:0 1rem;background:#101827;color:#e7edf7;font:16px system-ui}h1{color:#88d4ff}textarea,select,button{font:inherit;padding:.65rem;border-radius:6px}textarea{width:100%;box-sizing:border-box}button{background:#1d8cc8;color:white;border:0;margin-top:.6rem}.grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.card{background:#192337;padding:1rem;border-radius:8px;margin-top:1rem}.muted{color:#b3c1d5}code{color:#b6e3ff}</style></head><body>
<h1>Retrieval Eval Workbench</h1><p class='muted'>Compare TF-IDF with a local <code>all-MiniLM-L6-v2</code> dense retriever. Answers are deterministic extractive quotes, not generated prose.</p>
<div class='grid'><label>Retriever<select id='strategy'><option value='semantic'>Semantic</option><option value='lexical'>Lexical TF-IDF</option></select></label><label>Top K<select id='topk'><option>3</option><option>1</option><option>5</option></select></label></div>
<p><textarea id='query' rows='3'>What happens to a case if no routing rule matches?</textarea><br><button onclick='run()'>Run query</button></p><div id='output' class='card'>Run a query to inspect retrieved documents, answer, and line-level citation.</div>
<script>async function run(){let payload={query:query.value,strategy:strategy.value,top_k:+topk.value};let r=await fetch('/api/query',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(payload)});let d=await r.json();if(!r.ok){output.textContent=d.detail;return}let c=d.citations.map(x=>`<li><b>${x.title}</b>, line ${x.line_start}: “${x.quote}”</li>`).join('')||'<li>No citation: corpus evidence was not found.</li>';let h=d.hits.map(x=>`<li>${x.title} <span class='muted'>(${x.score})</span></li>`).join('');output.innerHTML=`<h2>${d.answerable?'Grounded extract':'No supported answer'}</h2><p>${d.answer}</p><h3>Citations</h3><ul>${c}</ul><h3>Retrieved documents</h3><ol>${h}</ol>`}</script></body></html>"""
