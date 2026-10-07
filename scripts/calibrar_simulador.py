"""
Calibração do roteador de comandos do simulador "Construa um jogo com a IA"
(seção 7 do site). Roda o MESMO modelo quantizado que o navegador usa, via
ONNX Runtime, e lê banco de exemplos/palavras-chave direto do
frontend/simulator-data.js (precisa de Node) — uma fonte só de verdade.

Uso:
  python scripts/calibrar_simulador.py avaliar [pt|en]   # avaliação final (mesma lógica do simulator.js)
  python scripts/calibrar_simulador.py fewshot           # experimento 1: few-shot prompting (PT)
  python scripts/calibrar_simulador.py camadas [pt|en]   # experimento 2: varredura de camadas/pooling

Resultados registrados (modelos q8 do site):
  - few-shot (PT: prever o nome do comando como próximo token): 36% no conjunto de ajuste
    (calibração contextual de Zhao et al. 2021 não ajudou: 33%) → descartado.
  - embeddings: "mix" (último token + média) deu 85% no conjunto novo PT, mas só 90% e
    2 execuções erradas no EN → trocado por "mean1" (média dos tokens sem o 1º).
  - config final (mean1, camadas 4–12, 2 vizinhos, limiares por idioma):
      PT conjunto novo: 19/20 (95%) top-1, 19 executados direto, 1 errado
         ("conta quantas maçãs eu comi" → comida; o site oferece a 2ª opção)
      EN conjunto novo: 20/20 (100%) top-1, 14 executados direto, 0 errados
"""
import json, re, subprocess, sys, time, unicodedata
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parent.parent
MODELS = {
    "pt": (REPO / "onnx_model_v2_quant", REPO / "onnx_model_v2_quant" / "onnx" / "model_quantized.onnx"),
    "en": (REPO / "onnx_model_v2_en_quant", REPO / "onnx_model_v2_en_quant" / "onnx" / "model_quantized.onnx"),
}

# Mesma configuração do simulator.js — mantenha em sincronia.
LAYERS = range(4, 13)
TOP_K = 2
KEYWORD_BONUS = 0.15
# Limiares por idioma (modelos diferentes → escalas de similaridade diferentes),
# escolhidos com o experimento "limiares". Executa se ≥ exec_min e com folga
# ≥ margin sobre o 2º; empatado, pergunta; abaixo de exec_min, recusa.
THRESHOLDS = {"pt": (0.30, 0.05), "en": (0.45, 0.05)}
LANG = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in THRESHOLDS else "pt"
EXEC_MIN, EXEC_MARGIN = THRESHOLDS[LANG]
ASK_MIN = 0.30


def load_sim_data():
    url = (REPO / "frontend" / "simulator-data.js").as_uri()
    js = f"import('{url}').then(m => console.log(JSON.stringify({{COMMANDS: m.COMMANDS, BANK: m.BANK, KEYWORDS: m.KEYWORDS}})))"
    out = subprocess.run(["node", "--input-type=module", "-e", js], capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(out.stdout)


DATA = load_sim_data()
LABELS = DATA["COMMANDS"]


class Model:
    def __init__(self, lang):
        tok_dir, onnx_path = MODELS[lang]
        self.tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.hs_names = [f"hidden_states.{i}" for i in range(13)]

    def _feed(self, ids):
        return {"input_ids": np.array([ids], dtype=np.int64), "attention_mask": np.ones((1, len(ids)), dtype=np.int64)}

    def logits_last(self, text):
        ids = self.tok.encode(text).ids
        return self.sess.run(["logits"], self._feed(ids))[0][0, -1].astype(np.float64)

    def hidden(self, text):
        ids = self.tok.encode(clean(text)).ids
        return [o[0].astype(np.float64) for o in self.sess.run(self.hs_names, self._feed(ids))]  # 13 x (seq, 768)


# ---------------------------------------------------------------------------
# Pré-processamento (espelha simulator.js)
# ---------------------------------------------------------------------------
def clean(text):
    # O vetor do último token é sensível a pontuação final: "(...morrer)" termina
    # em ")" e muda tudo. Tira parênteses/aspas e a pontuação do fim.
    out = re.sub(r"[()\[\]{}\"«»“”]", " ", text)
    out = re.sub(r"[\s.!?…,;:]+$", "", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out or text.strip()


def fold(s):
    # " texto sem acento e sem pontuação " — palavra-chave precisa começar no
    # início de uma palavra (senão "tema" casaria com "sistema").
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " " + re.sub(r"[^a-z0-9]+", " ", s).strip() + " "


# ---------------------------------------------------------------------------
# Roteador final (mesma lógica do simulator.js)
# ---------------------------------------------------------------------------
POOL = sys.argv[3] if len(sys.argv) > 3 else "mean1"

def mix_vectors(h):
    # Por camada: média dos tokens sem o 1º ("mean1"), opcionalmente junto com o último token ("mix")
    out = []
    for layer in LAYERS:
        m = h[layer]
        mean1 = m[1:].mean(0) if len(m) > 1 else m[0]
        out.append(np.concatenate([m[-1], mean1]) if POOL == "mix" else mean1)
    return out


class Router:
    def __init__(self, model, lang):
        self.model = model
        self.keywords = DATA["KEYWORDS"][lang]
        self.items = [(lab, ex) for lab in LABELS for ex in DATA["BANK"][lang][lab]]
        vecs = [mix_vectors(model.hidden(ex)) for _, ex in self.items]
        self.mu = [np.mean([v[i] for v in vecs], axis=0) for i in range(len(LAYERS))]
        self.bank = [self._center(v) for v in vecs]

    def _center(self, v):
        return [(x - mu) / np.linalg.norm(x - mu) for x, mu in zip(v, self.mu)]

    def scores(self, text, use_keywords=True):
        v = self._center(mix_vectors(self.model.hidden(text)))
        sims = np.array([np.mean([b[i] @ v[i] for i in range(len(LAYERS))]) for b in self.bank])
        folded = fold(text)
        result = {}
        for lab in LABELS:
            own = sorted([sims[j] for j, (l, _) in enumerate(self.items) if l == lab], reverse=True)
            hit = use_keywords and any(" " + k in folded for k in self.keywords[lab])
            result[lab] = float(np.mean(own[:TOP_K])) + (KEYWORD_BONUS if hit else 0.0)
        return result


def decide(s):
    srt = sorted(s.items(), key=lambda kv: -kv[1])
    (l1, s1), (_, s2) = srt[0], srt[1]
    if s1 >= EXEC_MIN and s1 - s2 >= EXEC_MARGIN:
        return "exec", l1
    if s1 >= ASK_MIN:
        return "ask", l1
    return "reject", None


# ---------------------------------------------------------------------------
# Conjuntos de teste
# ---------------------------------------------------------------------------
SETS = {
    "pt": {
        # Usado pra escolher camadas/pooling e ajustar o banco.
        "ajuste": [
            ("cria o jogo da cobra pra mim", "criar"), ("construa um jogo da cobrinha", "criar"),
            ("quero jogar snake, programa um", "criar"), ("bota uma comidinha pra cobra pegar", "comida"),
            ("adicione frutas no mapa", "comida"), ("a cobra precisa ter o que comer", "comida"),
            ("acrescente a comida", "comida"), ("bater na parede deveria matar a cobra", "parede"),
            ("faz ela morrer na borda da tela", "parede"), ("adiciona o sistema de batida na parede", "parede"),
            ("game over quando encostar no canto", "parede"), ("quando sair pela direita aparece na esquerda", "portal"),
            ("ela devia passar pelas paredes", "portal"), ("faz a tela dar a volta", "portal"),
            ("se a cobra se morder ela perde", "corpo"), ("colisão com o próprio corpo", "corpo"),
            ("encostar no rabo tem que matar", "corpo"), ("coloca uma pontuação", "placar"),
            ("quero ver o score", "placar"), ("conta os pontos e guarda o recorde", "placar"),
            ("aumenta a velocidade conforme ela come", "velocidade"), ("deixa mais difícil com o tempo", "velocidade"),
            ("tá muito lento", "velocidade"), ("botão de pausar", "pausa"),
            ("quero parar o jogo apertando espaço", "pausa"), ("dá pra congelar o jogo?", "pausa"),
            ("melhora o visual", "cores"), ("coloca umas cores mais legais", "cores"),
            ("tá muito feio, capricha no design", "cores"), ("corrige os erros", "corrigir"),
            ("o jogo tá bugado", "corrigir"), ("a cobra tá deixando rastro, arruma isso", "corrigir"),
            ("conserta os bugs", "corrigir"),
            # segundo lote (inclui a frase da demonstração)
            ("COnstrua um jogo da cobrinha", "criar"), ("bora fazer um snake", "criar"), ("gera o jogo", "criar"),
            ("agora acrescente a comida", "comida"), ("põe uma maçã pra ela comer", "comida"),
            ("o sistema de batida (pra quando a cobra bater na parede morrer)", "parede"),
            ("quero que bater no muro seja fatal", "parede"), ("faz ela reaparecer do outro lado da tela", "portal"),
            ("sem paredes, ela passa direto", "portal"), ("se ela comer o próprio corpo morre", "corpo"),
            ("quero ver minha pontuação", "placar"), ("adicione pontos", "placar"),
            ("fica mais rápido a cada maçã", "velocidade"), ("aumenta a dificuldade", "velocidade"),
            ("aperta espaço pra pausar", "pausa"), ("deixa colorido", "cores"),
            ("dá uma cara mais profissional", "cores"), ("corrija o jogo", "corrigir"), ("tá cheio de bug", "corrigir"),
        ],
        # Escrito DEPOIS de fechar banco/palavras-chave/limiares — o número honesto.
        "novo": [
            ("faz pra mim aquele jogo da cobra", "criar"), ("inicia um projeto de snake", "criar"),
            ("coloca umas maçãs vermelhas", "comida"), ("a cobra tá com fome", "comida"),
            ("quando tocar a parede, fim de jogo", "parede"), ("as bordas têm que matar", "parede"),
            ("deixa sair pela esquerda e entrar pela direita", "portal"), ("tira as paredes", "portal"),
            ("ela não pode se enrolar em si mesma", "corpo"), ("morder a si mesma deveria acabar o jogo", "corpo"),
            ("conta quantas maçãs eu comi", "placar"), ("exibe a pontuação no canto", "placar"),
            ("cada vez mais rápido", "velocidade"), ("muito fácil, deixa mais difícil", "velocidade"),
            ("preciso pausar pra atender o telefone", "pausa"), ("quero um botão de parar", "pausa"),
            ("põe um fundo mais bonito", "cores"), ("muda a cor da cobra pra verde", "cores"),
            ("tem coisa quebrada", "corrigir"), ("arruma esses problemas", "corrigir"),
        ],
        "fora": ["qual é a capital da França?", "me conta uma piada", "oi, tudo bem?", "quanto é dois mais dois",
                 "faz um jogo de xadrez", "obrigado!", "quem é você?", "escreva um poema"],
    },
    "en": {
        # O banco EN foi escrito sem olhar estes resultados — todos são "novos" na 1ª rodada.
        "ajuste": [
            ("Build a snake game", "criar"), ("make me that snake game", "criar"), ("code a snake game", "criar"),
            ("Now add the food", "comida"), ("put some apples on the board", "comida"), ("the snake is hungry", "comida"),
            ("add the food and the crash system", "comida"),
            ("Make it die when it hits the wall", "parede"), ("the edges should kill it", "parede"),
            ("game over when it touches the border", "parede"),
            ("Let it go through the walls", "portal"), ("leave on the left and come back on the right", "portal"),
            ("remove the walls", "portal"),
            ("It should die if it bites its own body", "corpo"), ("it can't run into itself", "corpo"),
            ("biting its own tail ends the game", "corpo"),
            ("Show the score", "placar"), ("count how many apples I ate", "placar"), ("display the points", "placar"),
            ("Make the snake faster over time", "velocidade"), ("it's too easy, make it harder", "velocidade"),
            ("faster and faster", "velocidade"),
            ("I want to be able to pause", "pausa"), ("I need to pause to answer the phone", "pausa"),
            ("add a stop button", "pausa"),
            ("Make the game prettier", "cores"), ("give it a nicer background", "cores"), ("make the snake green", "cores"),
            ("Fix the bugs", "corrigir"), ("something is broken", "corrigir"), ("clean up these problems", "corrigir"),
        ],
        # Escrito depois do ajuste do banco EN (troca de 2 exemplos) — o número honesto.
        "novo": [
            ("create the snake game please", "criar"), ("let's start a new snake project", "criar"),
            ("add some red apples", "comida"), ("give the snake something to eat", "comida"),
            ("touching the wall should end the game", "parede"), ("walls are deadly", "parede"),
            ("let it wrap around the screen", "portal"), ("it should come out on the other side", "portal"),
            ("don't let it cross its own body", "corpo"), ("eating itself means game over", "corpo"),
            ("keep track of my points", "placar"), ("show the high score", "placar"),
            ("speed it up a bit each time", "velocidade"), ("the game is too slow", "velocidade"),
            ("let me freeze the game", "pausa"), ("pause with the P key", "pausa"),
            ("use nicer colors", "cores"), ("make it look like a real game", "cores"),
            ("there's a bug, fix it", "corrigir"), ("repair the broken stuff", "corrigir"),
        ],
        "fora": ["what is the capital of France?", "tell me a joke", "hi, how are you?", "what's two plus two",
                 "make a chess game", "thanks!", "who are you?", "write a poem"],
    },
}


def report(router, name, data, use_keywords=True, verbose=True):
    ok = execd = wrong_exec = 0
    for x, gold in data:
        s = router.scores(x, use_keywords)
        srt = sorted(s.items(), key=lambda kv: -kv[1])
        action, lab = decide(s)
        ok += srt[0][0] == gold
        execd += action == "exec"
        wrong_exec += action == "exec" and lab != gold
        if verbose and (srt[0][0] != gold or action != "exec"):
            mark = "ok" if srt[0][0] == gold else "X "
            print(f"  {mark} {action:6} {x!r:64} -> {srt[0][0]:10} {srt[0][1]:.3f} (2º {srt[1][0]} {srt[1][1]:.3f})")
    print(f"{name}: top-1 {ok}/{len(data)} = {ok/len(data):.0%} | executou direto {execd}, "
          f"dos quais ERRADOS {wrong_exec} | perguntou/recusou {len(data)-execd}\n")


def cmd_avaliar(lang):
    model = Model(lang)
    t0 = time.time()
    router = Router(model, lang)
    print(f"[{lang}] banco: {len(router.items)} exemplos em {time.time()-t0:.1f}s\n")
    sets = SETS[lang]
    report(router, "AJUSTE", sets["ajuste"])
    if sets["novo"]:
        report(router, "NOVO sem palavras-chave", sets["novo"], use_keywords=False, verbose=False)
        report(router, "NOVO com palavras-chave", sets["novo"])
    print("Fora do escopo:")
    for x in sets["fora"]:
        s = router.scores(x)
        srt = sorted(s.items(), key=lambda kv: -kv[1])
        print(f"  {decide(s)[0]:6} {x!r:40} -> {srt[0][0]:10} {srt[0][1]:.3f}")


# ---------------------------------------------------------------------------
# Experimento 1: few-shot prompting (descartado — 36%)
# ---------------------------------------------------------------------------
FEWSHOT = [
    ("faça um jogo da cobrinha", "criar"), ("quero que a cobra coma maçãs e cresça", "comida"),
    ("se ela bater na parede tem que morrer", "parede"), ("deixa a cobra atravessar a borda e sair do outro lado", "portal"),
    ("ela deve morrer quando morder o próprio rabo", "corpo"), ("mostra quantos pontos eu fiz", "placar"),
    ("o jogo podia ficar mais rápido com o tempo", "velocidade"), ("quero poder parar o jogo um pouco", "pausa"),
    ("deixa o jogo mais bonito e colorido", "cores"), ("tem uns bugs estranhos, conserta aí", "corrigir"),
]


def cmd_fewshot():
    model = Model("pt")
    label_ids = [model.tok.encode(" " + l).ids[0] for l in LABELS]  # todos os rótulos são 1 token só

    def probs(user):
        lines = ["Pedidos de um usuário e o comando que o assistente executa.", ""]
        for p, c in FEWSHOT:
            lines += [f"Pedido: {p}", f"Comando: {c}", ""]
        lines += [f"Pedido: {user}", "Comando:"]
        sel = model.logits_last("\n".join(lines))[label_ids]
        p = np.exp(sel - sel.max())
        return p / p.sum()

    data = SETS["pt"]["ajuste"][:33]
    raw_ok = sum(LABELS[int(np.argmax(probs(x)))] == g for x, g in data)
    # Calibração contextual (Zhao et al., 2021): divide pelo viés de rótulo com entrada "vazia".
    base = np.mean([probs(x) for x in ["N/A", "", "[MASK]"]], axis=0)
    cal_ok = sum(LABELS[int(np.argmax(probs(x) / base))] == g for x, g in data)
    print(f"few-shot bruto: {raw_ok}/{len(data)} | com calibração contextual: {cal_ok}/{len(data)}")


# ---------------------------------------------------------------------------
# Experimento 2: qual camada / pooling usar
# ---------------------------------------------------------------------------
def cmd_camadas(lang):
    model = Model(lang)
    items = [(lab, ex) for lab in LABELS for ex in DATA["BANK"][lang][lab]]
    bank_h = [model.hidden(ex) for _, ex in items]
    data = SETS[lang]["ajuste"]
    test_h = [model.hidden(x) for x, _ in data]
    pools = {
        "last": lambda m: m[-1],
        "mean": lambda m: m.mean(0),  # ruim: o 1º token do GPT-2 domina a média
        "mean1": lambda m: m[1:].mean(0) if len(m) > 1 else m[0],
        "mix": lambda m: np.concatenate([m[-1], m[1:].mean(0) if len(m) > 1 else m[0]]),
    }
    for pname, pool in pools.items():
        for layers in [[l] for l in range(13)] + [list(range(4, 13)), list(range(6, 12))]:
            sims = np.zeros((len(data), len(items)))
            for layer in layers:
                B = np.array([pool(h[layer]) for h in bank_h])
                mu = B.mean(0)
                Bc = B - mu
                Bc /= np.linalg.norm(Bc, axis=1, keepdims=True)
                T = np.array([pool(h[layer]) for h in test_h]) - mu
                T /= np.linalg.norm(T, axis=1, keepdims=True)
                sims += T @ Bc.T
            ok = 0
            for i, (_, gold) in enumerate(data):
                sc = [np.mean(sorted([sims[i, j] for j, (l, _) in enumerate(items) if l == lab], reverse=True)[:TOP_K]) for lab in LABELS]
                ok += LABELS[int(np.argmax(sc))] == gold
            name = f"camada {layers[0]}" if len(layers) == 1 else f"camadas {layers[0]}-{layers[-1]}"
            print(f"{pname:5} {name:14} {ok}/{len(data)}")


# ---------------------------------------------------------------------------
# Experimento 3: limiares de decisão (executar / perguntar / recusar)
# ---------------------------------------------------------------------------
def cmd_limiares():
    cache = {}
    for lang in ["pt", "en"]:
        router = Router(Model(lang), lang)
        sets = SETS[lang]
        labeled = sets["ajuste"] + sets["novo"]
        cache[lang] = ([(router.scores(x), g) for x, g in labeled], [router.scores(x) for x in sets["fora"]])
    print("exec_min margem ask_min | por idioma: certos executados / ERRADOS executados / fora executados / fora recusados")
    for exec_min in [0.30, 0.35, 0.40, 0.45, 0.50]:
        for margin in [0.05, 0.08]:
            for ask_min in [0.20, 0.25, 0.30]:
                if ask_min > exec_min:
                    continue
                row = []
                for lang in ["pt", "en"]:
                    labeled, oos = cache[lang]
                    def dec(s):
                        srt = sorted(s.items(), key=lambda kv: -kv[1])
                        if srt[0][1] >= exec_min and srt[0][1] - srt[1][1] >= margin: return "exec", srt[0][0]
                        return ("ask", None) if srt[0][1] >= ask_min else ("reject", None)
                    good = sum(1 for s, g in labeled if dec(s) == ("exec", g))
                    bad = sum(1 for s, g in labeled if dec(s)[0] == "exec" and dec(s)[1] != g)
                    oos_exec = sum(1 for s in oos if dec(s)[0] == "exec")
                    oos_rej = sum(1 for s in oos if dec(s)[0] == "reject")
                    row.append(f"{lang}: {good:2d}/{len(labeled)} err {bad} | fora exec {oos_exec} rec {oos_rej}/{len(oos)}")
                print(f"{exec_min:.2f} {margin:.2f} {ask_min:.2f} | " + "   ".join(row))


# ---------------------------------------------------------------------------
# Checagem: toda sugestão clicável do chat tem que disparar o próprio comando
# (os botões mandam o texto pelo modelo, igual a um pedido digitado).
# ---------------------------------------------------------------------------
DEMO = {
    "pt": ["COnstrua um jogo da cobrinha", "agora acrescente a comida",
           "o sistema de batida (pra quando a cobra bater na parede morrer)", "fica mais rápido a cada comida"],
    "en": ["Build a snake game", "now add the food", "the crash system (so the snake dies when it hits the wall)",
           "get faster with every bite"],
}


def cmd_sugestoes(lang):
    url = (REPO / "frontend" / "simulator-data.js").as_uri()
    js = f"import('{url}').then(m => console.log(JSON.stringify(m.SIM_TEXT['{lang}'].suggestions)))"
    out = subprocess.run(["node", "--input-type=module", "-e", js], capture_output=True, text=True, encoding="utf-8", check=True)
    router = Router(Model(lang), lang)
    for cmd, text in list(json.loads(out.stdout).items()) + [("?", x) for x in DEMO[lang]]:
        s = router.scores(text)
        srt = sorted(s.items(), key=lambda kv: -kv[1])[:2]
        d = decide(s)
        flag = "ok" if d == ("exec", cmd) else "  " if cmd == "?" else "XX"
        print(f"{flag} {d[0]:6} {srt[0][0]:10} {srt[0][1]:.3f} (2º {srt[1][0]} {srt[1][1]:.3f})  {text}")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "avaliar"
    lang = sys.argv[2] if len(sys.argv) > 2 else "pt"
    {"avaliar": lambda: cmd_avaliar(lang), "fewshot": cmd_fewshot, "camadas": lambda: cmd_camadas(lang),
     "limiares": cmd_limiares, "sugestoes": lambda: cmd_sugestoes(lang)}[action]()
