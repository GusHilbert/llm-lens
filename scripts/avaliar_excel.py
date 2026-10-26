"""
Experimento RAG × fine-tuning com as fichas do Excel (seções 8 e 9 do site).
Protocolo em avaliacao/README.md.

Uso:
  python scripts/avaliar_excel.py baixar              # baixa o modelo do site (precisa de acesso ao huggingface.co)
  python scripts/avaliar_excel.py busca               # só recuperação: BM25 × hidden states do GPT-2
  python scripts/avaliar_excel.py original            # GPT-2 PT sem contexto
  python scripts/avaliar_excel.py rag [bm25|gpt2|oraculo]

"busca" com BM25 roda sem o modelo. As outras configurações usam o MESMO modelo
quantizado do navegador (GusHil/gpt2-small-portuguese-onnx-v2), via ONNX Runtime.
Os resultados vão para avaliacao/resultados/<config>.json.
"""
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FICHAS = REPO / "frontend" / "data" / "fichas-excel.json"
PERGUNTAS = REPO / "avaliacao" / "perguntas-teste-excel.json"
RESULTADOS = REPO / "avaliacao" / "resultados"
MODEL_REPO = "GusHil/gpt2-small-portuguese-onnx-v2"
MODEL_DIR = REPO / "onnx_model_v2_quant"  # mesmo caminho do calibrar_simulador.py

TOP_K = 3  # fichas que entram no prompt do RAG
# Mesma escolha do simulador (simulator.js): média dos tokens sem o 1º, camadas 4–12.
LAYERS = range(4, 13)


def carregar():
    fichas = json.loads(FICHAS.read_text(encoding="utf-8"))["fichas"]
    dados = json.loads(PERGUNTAS.read_text(encoding="utf-8"))
    return fichas, dados["perguntas"], dados["modelo_prompt"]


def wilson(acertos, n, z=1.96):
    # Intervalo de confiança de 95% para uma proporção — com 46 perguntas ele é largo,
    # e a monografia precisa mostrar isso junto com a taxa de acerto.
    if n == 0:
        return 0.0, 0.0
    p = acertos / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    margem = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centro - margem, centro + margem


def fmt_taxa(acertos, n):
    lo, hi = wilson(acertos, n)
    return f"{acertos}/{n} = {acertos / n:.0%} (IC95% {lo:.0%}–{hi:.0%})"


def salvar(nome, conteudo):
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    caminho = RESULTADOS / f"{nome}.json"
    caminho.write_text(json.dumps(conteudo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nDetalhes salvos em {caminho.relative_to(REPO)}")


# ---------------------------------------------------------------------------
# Recuperação
# ---------------------------------------------------------------------------
STOPWORDS = set("""
a o as os um uma uns umas de do da dos das no na nos nas em por para com sem que se e ou
eu meu minha voce qual quais como onde quando quero preciso uso usar tenho ao aos
""".split())


def palavras(texto):
    # Sem acento e sem pontuação: "CONT.SE" vira "cont se", "Média" vira "media".
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return [w for w in re.split(r"[^a-z0-9]+", t) if w and w not in STOPWORDS]


class BM25:
    """BM25 clássico (k1=1,5; b=0,75) — a busca por palavras usada antes dos embeddings."""

    def __init__(self, textos, k1=1.5, b=0.75):
        self.docs = [Counter(palavras(t)) for t in textos]
        self.tam = [sum(d.values()) for d in self.docs]
        self.media = sum(self.tam) / len(self.docs)
        df = Counter(w for d in self.docs for w in d)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.k1, self.b = k1, b

    def scores(self, consulta):
        q = palavras(consulta)
        out = []
        for d, tam in zip(self.docs, self.tam):
            s = 0.0
            for w in q:
                f = d.get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * tam / self.media))
            out.append(s)
        return np.array(out)


class BuscaGPT2:
    """Mesma ideia do roteador do simulador, agora comparando pergunta × ficha."""

    def __init__(self, modelo, textos):
        vecs = [modelo.vetor(t) for t in textos]
        self.mu = np.mean(vecs, axis=0)
        self.banco = np.array([self._norm(v) for v in vecs])

    def _norm(self, v):
        c = v - self.mu
        return c / (np.linalg.norm(c, axis=-1, keepdims=True) + 1e-9)

    def scores(self, consulta, modelo):
        q = self._norm(modelo.vetor(consulta))
        return np.einsum("nld,ld->n", self.banco, q) / len(LAYERS)


def ranking(scores):
    return list(np.argsort(-scores, kind="stable"))


def avaliar_busca(fichas, perguntas, nome, score_fn):
    ids = [f["id"] for f in fichas]
    top1 = top3 = 0
    detalhes = []
    for p in perguntas:
        ordem = [ids[i] for i in ranking(score_fn(p["pergunta"]))]
        rel = set(p["fichas_relevantes"])
        h1, h3 = ordem[0] in rel, bool(rel & set(ordem[:TOP_K]))
        top1 += h1
        top3 += h3
        detalhes.append({"id": p["id"], "recuperadas": ordem[:TOP_K], "acerto_top1": h1, "acerto_top3": h3})
    n = len(perguntas)
    print(f"{nome}:")
    print(f"  ficha certa em 1º lugar:    {fmt_taxa(top1, n)}")
    print(f"  ficha certa entre as {TOP_K}:    {fmt_taxa(top3, n)}")
    erros = [d for d in detalhes if not d["acerto_top3"]]
    if erros:
        print(f"  não encontradas: {', '.join(d['id'] for d in erros)}")
    return {"top1": top1, "top3": top3, "n": n, "perguntas": detalhes}


# ---------------------------------------------------------------------------
# Modelo (ONNX quantizado, o mesmo do navegador)
# ---------------------------------------------------------------------------
class Modelo:
    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        onnx = MODEL_DIR / "onnx" / "model_quantized.onnx"
        if not onnx.exists():
            sys.exit(f"Modelo não encontrado em {MODEL_DIR}. Rode primeiro: python scripts/avaliar_excel.py baixar")
        self.tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        self.sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
        self.max_len = 1024

    def ids(self, texto):
        return self.tok.encode(texto, add_special_tokens=False).ids

    def _feed(self, ids):
        return {"input_ids": np.array([ids], dtype=np.int64), "attention_mask": np.ones((1, len(ids)), dtype=np.int64)}

    def logprobs(self, ids):
        logits = self.sess.run(["logits"], self._feed(ids))[0][0].astype(np.float64)
        logits -= logits.max(axis=-1, keepdims=True)
        return logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))

    def logprob_continuacao(self, contexto, continuacao):
        # Contexto e continuação tokenizados separados (como no lm-evaluation-harness):
        # a continuação começa com espaço, que o BPE do GPT-2 cola no primeiro token dela.
        ctx, cont = self.ids(contexto), self.ids(continuacao)
        todos = ctx + cont
        if len(todos) > self.max_len:
            raise ValueError(f"prompt com {len(todos)} tokens passa do limite de {self.max_len}")
        lp = self.logprobs(todos)
        total = sum(lp[len(ctx) + i - 1, t] for i, t in enumerate(cont))
        return float(total), len(cont)

    def vetor(self, texto):
        nomes = [f"hidden_states.{i}" for i in LAYERS]
        hs = self.sess.run(nomes, self._feed(self.ids(texto)))
        return np.array([h[0, 1:].mean(0) if h.shape[1] > 1 else h[0, 0] for h in hs], dtype=np.float64)


# ---------------------------------------------------------------------------
# Múltipla escolha por verossimilhança
# ---------------------------------------------------------------------------
def montar_prompt(modelo_prompt, p, contexto_fichas):
    base = modelo_prompt.split("{opcao}")[0].rstrip()  # "...Resposta: {inicio_resposta}"
    prompt = base.format(pergunta=p["pergunta"], inicio_resposta=p["inicio_resposta"])
    if contexto_fichas:
        # Ficha mais parecida por último, colada na pergunta: modelos pequenos
        # dão mais peso ao que está perto do fim do texto.
        blocos = "\n\n".join(f["texto"] for f in reversed(contexto_fichas))
        prompt = f"{blocos}\n\n{prompt}"
    return prompt


def avaliar_lm(modelo, fichas, perguntas, modelo_prompt, nome, recuperar=None):
    por_id = {f["id"]: f for f in fichas}
    acertos = acertos_norm = 0
    detalhes = []
    for p in perguntas:
        contexto = [por_id[i] for i in recuperar(p)] if recuperar else []
        prompt = montar_prompt(modelo_prompt, p, contexto)
        opcoes = [p["resposta"]] + p["distratores"]
        lps = [modelo.logprob_continuacao(prompt, " " + o) for o in opcoes]
        soma = [lp for lp, _ in lps]
        norm = [lp / n for lp, n in lps]
        ok, ok_norm = int(np.argmax(soma)) == 0, int(np.argmax(norm)) == 0
        acertos += ok
        acertos_norm += ok_norm
        # Probabilidades relativas entre as 4 opções — é o que o site vai desenhar como barras.
        rel = np.exp(np.array(soma) - max(soma))
        rel /= rel.sum()
        detalhes.append({
            "id": p["id"],
            "fichas_no_prompt": [f["id"] for f in contexto],
            "escolhida": opcoes[int(np.argmax(soma))],
            "acerto": ok,
            "acerto_normalizado": ok_norm,
            "prob_relativa": {o: round(float(x), 4) for o, x in zip(opcoes, rel)},
        })
    n = len(perguntas)
    print(f"{nome}:")
    print(f"  acerto (soma das log-probs):       {fmt_taxa(acertos, n)}")
    print(f"  acerto (normalizado por token):    {fmt_taxa(acertos_norm, n)}")
    print("  (escolher ao acaso acerta 25%)")
    return {"acertos": acertos, "acertos_normalizado": acertos_norm, "n": n, "perguntas": detalhes}


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "busca"
    fichas, perguntas, modelo_prompt = carregar()
    textos = [f["texto"] for f in fichas]
    ids = [f["id"] for f in fichas]

    if cmd == "baixar":
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_REPO, local_dir=MODEL_DIR)
        print(f"Modelo salvo em {MODEL_DIR.relative_to(REPO)}")
        return

    bm25 = BM25(textos)

    if cmd == "busca":
        res = {"bm25": avaliar_busca(fichas, perguntas, "BM25 (palavras)", bm25.scores)}
        if (MODEL_DIR / "onnx" / "model_quantized.onnx").exists():
            modelo = Modelo()
            busca = BuscaGPT2(modelo, textos)
            res["gpt2"] = avaliar_busca(fichas, perguntas, "Hidden states do GPT-2", lambda q: busca.scores(q, modelo))
        else:
            print("\n(Modelo não baixado: pulando a busca por hidden states do GPT-2.)")
        salvar("busca", res)
        return

    modelo = Modelo()
    if cmd == "original":
        salvar("original", avaliar_lm(modelo, fichas, perguntas, modelo_prompt, "GPT-2 PT original (sem contexto)"))
    elif cmd == "rag":
        tipo = sys.argv[2] if len(sys.argv) > 2 else "bm25"
        if tipo == "oraculo":
            # Limite superior: as fichas certas sempre entram no prompt.
            recuperar = lambda p: p["fichas_relevantes"]
        elif tipo == "gpt2":
            busca = BuscaGPT2(modelo, textos)
            recuperar = lambda p: [ids[i] for i in ranking(busca.scores(p["pergunta"], modelo))[:TOP_K]]
        else:
            recuperar = lambda p: [ids[i] for i in ranking(bm25.scores(p["pergunta"]))[:TOP_K]]
        salvar(f"rag-{tipo}", avaliar_lm(modelo, fichas, perguntas, modelo_prompt, f"RAG ({tipo}, top-{TOP_K})", recuperar))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
