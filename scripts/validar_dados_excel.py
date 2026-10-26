"""
Confere a base de fichas do Excel (frontend/data/fichas-excel.json) e o
conjunto de teste (avaliacao/perguntas-teste-excel.json) antes de usá-los no
RAG, no fine-tuning ou no experimento. Só usa a biblioteca padrão.

Uso:
  python scripts/validar_dados_excel.py

Checagens:
  - ids únicos e campos obrigatórios presentes;
  - fichas curtas o bastante para caber 3 delas no contexto do GPT-2;
  - cada pergunta tem 3 distratores, sem repetir a resposta;
  - fichas_relevantes existem e contêm a resposta (senão o RAG não teria
    como ajudar);
  - o conhecimento do cenário "atualizacao" (PROCX) não vaza para as fichas
    que entram no fine-tuning — senão a demonstração RAG × fine-tuning perde
    o sentido.
"""
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FICHAS = REPO / "frontend" / "data" / "fichas-excel.json"
PERGUNTAS = REPO / "avaliacao" / "perguntas-teste-excel.json"

MIN_PALAVRAS, MAX_PALAVRAS = 30, 90
# Termos que só podem aparecer em fichas do cenário "atualizacao".
TERMOS_ATUALIZACAO = ["PROCX"]


def main():
    fichas = json.loads(FICHAS.read_text(encoding="utf-8"))["fichas"]
    perguntas = json.loads(PERGUNTAS.read_text(encoding="utf-8"))["perguntas"]
    erros = []

    ids = [f["id"] for f in fichas]
    for dup, n in Counter(ids).items():
        if n > 1:
            erros.append(f"ficha duplicada: {dup}")
    por_id = {f["id"]: f for f in fichas}

    for f in fichas:
        for campo in ("id", "categoria", "titulo", "texto"):
            if not f.get(campo):
                erros.append(f"ficha {f.get('id')}: falta '{campo}'")
        palavras = len(f["texto"].split())
        if not MIN_PALAVRAS <= palavras <= MAX_PALAVRAS:
            erros.append(f"ficha {f['id']}: {palavras} palavras (esperado {MIN_PALAVRAS}–{MAX_PALAVRAS})")
        if f.get("cenario") != "atualizacao":
            for termo in TERMOS_ATUALIZACAO:
                if termo in f["texto"]:
                    erros.append(f"ficha {f['id']}: menciona {termo}, que deve ficar fora do fine-tuning")

    ids_p = [p["id"] for p in perguntas]
    for dup, n in Counter(ids_p).items():
        if n > 1:
            erros.append(f"pergunta duplicada: {dup}")

    for p in perguntas:
        for campo in ("id", "pergunta", "inicio_resposta", "resposta", "distratores", "fichas_relevantes"):
            if not p.get(campo):
                erros.append(f"pergunta {p.get('id')}: falta '{campo}'")
        if len(p["distratores"]) != 3 or len(set(p["distratores"])) != 3:
            erros.append(f"pergunta {p['id']}: precisa de 3 distratores diferentes")
        if p["resposta"] in p["distratores"]:
            erros.append(f"pergunta {p['id']}: a resposta aparece entre os distratores")
        relevantes = [por_id.get(fid) for fid in p["fichas_relevantes"]]
        if None in relevantes:
            erros.append(f"pergunta {p['id']}: ficha relevante inexistente em {p['fichas_relevantes']}")
        elif not any(p["resposta"] in f["texto"] for f in relevantes):
            erros.append(f"pergunta {p['id']}: nenhuma ficha relevante contém a resposta '{p['resposta']}'")

    cobertas = {fid for p in perguntas for fid in p["fichas_relevantes"]}
    a_verificar = [f["id"] for f in fichas if f.get("verificar")]

    print(f"Fichas: {len(fichas)} ({len(fichas) - sum(f.get('cenario') == 'atualizacao' for f in fichas)} entram no fine-tuning)")
    for cat, n in sorted(Counter(f["categoria"] for f in fichas).items()):
        print(f"  {cat}: {n}")
    print(f"Perguntas de teste: {len(perguntas)} (cobrem {len(cobertas)} fichas)")
    print(f"Fichas com detalhe a confirmar no Excel: {len(a_verificar)} -> {', '.join(a_verificar)}")

    if erros:
        print(f"\n{len(erros)} problema(s):")
        for e in erros:
            print(f"  - {e}")
        sys.exit(1)
    print("\nTudo certo.")


if __name__ == "__main__":
    main()
