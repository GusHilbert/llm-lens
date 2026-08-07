"""
app.py — API do LLM Lens

Expõe os internals do modelo (tokenização, embeddings, atenção, geração)
como endpoints REST simples para o frontend consumir.

Rodar com: uvicorn app:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model_utils import get_lens, DEFAULT_MODEL_NAME

app = FastAPI(
    title="LLM Lens API",
    description="Introspecção passo a passo de um modelo de linguagem pequeno.",
    version="0.1.0",
)

# Libera CORS para o frontend local (ajuste em produção)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str
    model_name: str = DEFAULT_MODEL_NAME


class GenerateStepRequest(BaseModel):
    text: str
    chosen_token_id: int | None = None
    top_k: int = 10
    temperature: float = 1.0
    model_name: str = DEFAULT_MODEL_NAME


@app.get("/")
def health_check():
    return {"status": "ok", "message": "LLM Lens API rodando."}


@app.post("/api/tokenize")
def tokenize(req: TextRequest):
    lens = get_lens(req.model_name)
    return {"tokens": lens.tokenize(req.text)}


@app.post("/api/embeddings")
def embeddings(req: TextRequest):
    lens = get_lens(req.model_name)
    return {"points": lens.embeddings_2d(req.text)}


@app.post("/api/attention")
def attention(req: TextRequest):
    lens = get_lens(req.model_name)
    return lens.attention_maps(req.text)


@app.post("/api/next-token")
def next_token(req: TextRequest):
    lens = get_lens(req.model_name)
    return {"distribution": lens.next_token_distribution(req.text)}


@app.post("/api/generate-step")
def generate_step(req: GenerateStepRequest):
    lens = get_lens(req.model_name)
    return lens.generate_step(
        req.text,
        chosen_token_id=req.chosen_token_id,
        top_k=req.top_k,
        temperature=req.temperature,
    )
