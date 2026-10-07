// ============================================================================
// SEÇÃO 7 — "Construa um jogo com a IA"
// ----------------------------------------------------------------------------
// A pessoa conversa com o modelo e ele "programa" um jogo da cobrinha. Na
// verdade o modelo não escreve nada: ele transforma o pedido num vetor (os
// hidden states — os mesmos da seção 3), compara com os vetores de 100
// pedidos de exemplo e dispara o comando mais parecido. Cada comando liga uma
// peça que já está pronta aqui embaixo. É o mesmo princípio de "tool calling"
// dos agentes de verdade: o modelo escolhe a ferramenta, código comum executa.
// ============================================================================
import { COMMANDS, BUGS_FROM, BANK, KEYWORDS, CODE, FIX_CODE, SIM_TEXT } from "./simulator-data.js";

// Configuração calibrada em scripts/calibrar_simulador.py — mantenha em sincronia.
const LAYERS = [4, 5, 6, 7, 8, 9, 10, 11, 12];
const TOP_K = 2;
const KEYWORD_BONUS = 0.15;
// Limiares por idioma (modelos diferentes → escalas de similaridade diferentes).
// Executa se ≥ execMin e com folga ≥ margin sobre o 2º; entre ASK_MIN e execMin
// (ou empatado), pergunta; abaixo de ASK_MIN, recusa.
const THRESHOLDS = { pt: { execMin: 0.3, margin: 0.05 }, en: { execMin: 0.45, margin: 0.05 } };
const ASK_MIN = 0.3;

const FEATURE_ORDER = ["comida", "parede", "corpo", "placar", "velocidade", "pausa", "cores", "portal"];

// Nome "de API" de cada comando, mostrado como a chamada de ferramenta gerada.
const TOOL_CALL = {
  criar: { tool: "create_game", game: "snake" },
  comida: { tool: "add_feature", feature: "food" },
  parede: { tool: "add_feature", feature: "wall_collision" },
  portal: { tool: "add_feature", feature: "wrap_around" },
  corpo: { tool: "add_feature", feature: "self_collision" },
  placar: { tool: "add_feature", feature: "score" },
  velocidade: { tool: "add_feature", feature: "speed_up" },
  pausa: { tool: "add_feature", feature: "pause" },
  cores: { tool: "add_feature", feature: "visual_theme" },
  corrigir: { tool: "fix_bugs" },
};

// " texto sem acento e sem pontuação " — palavras-chave precisam começar no
// início de uma palavra (senão "tema" casaria com "sistema").
const fold = (s) =>
  " " + s.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "").replace(/[^a-z0-9]+/g, " ").trim() + " ";

// O vetor do último token é sensível a pontuação final: "(...morrer)" termina
// em ")" e muda tudo. Tira parênteses/aspas e a pontuação do fim.
const clean = (s) => {
  const out = s.replace(/[()[\]{}"«»“”]/g, " ").replace(/[\s.!?…,;:]+$/u, "").replace(/\s+/g, " ").trim();
  return out || s.trim();
};

// ============================================================================
// ROTEADOR — escolhe o comando comparando hidden states
// ============================================================================
function createRouter({ tokenizer, model, lang }) {
  const items = [];
  COMMANDS.forEach((cmd) => BANK[lang][cmd].forEach((text) => items.push({ cmd, text })));
  const keywords = KEYWORDS[lang];
  const { execMin, margin } = THRESHOLDS[lang];

  let mean = null; // vetor médio do banco, por camada (pra centralizar)
  let bankVecs = null;
  let readyPromise = null;
  const progressListeners = new Set();

  // Pra cada camada: média dos vetores dos tokens, sem o 1º — no GPT-2 o 1º
  // token tem ativações gigantes que dominariam uma média simples.
  async function rawVectors(text) {
    const inputs = await tokenizer(clean(text));
    const outputs = await model({ ...inputs });
    const ids = Array.from(inputs.input_ids.data).map(Number);
    const vecs = LAYERS.map((layer) => {
      const tensor = outputs[`hidden_states.${layer}`];
      const [, seqLen, dim] = tensor.dims;
      const data = tensor.data;
      const v = new Float64Array(dim);
      const from = seqLen > 1 ? 1 : 0;
      for (let s = from; s < seqLen; s++) {
        for (let d = 0; d < dim; d++) v[d] += data[s * dim + d];
      }
      for (let d = 0; d < dim; d++) v[d] /= seqLen - from;
      return v;
    });
    return { vecs, ids };
  }

  function centerAndNormalize(vecs) {
    return vecs.map((v, li) => {
      const out = new Float64Array(v.length);
      let norm = 0;
      for (let d = 0; d < v.length; d++) {
        out[d] = v[d] - mean[li][d];
        norm += out[d] * out[d];
      }
      norm = Math.sqrt(norm) || 1;
      for (let d = 0; d < v.length; d++) out[d] /= norm;
      return out;
    });
  }

  function ensureReady() {
    if (!readyPromise) {
      readyPromise = (async () => {
        const raw = [];
        for (let i = 0; i < items.length; i++) {
          raw.push((await rawVectors(items[i].text)).vecs);
          progressListeners.forEach((fn) => fn(i + 1, items.length));
        }
        mean = LAYERS.map((_, li) => {
          const m = new Float64Array(raw[0][li].length);
          raw.forEach((vecs) => vecs[li].forEach((x, d) => { m[d] += x; }));
          return m.map((x) => x / raw.length);
        });
        bankVecs = raw.map(centerAndNormalize);
      })();
      readyPromise.catch(() => { readyPromise = null; });
    }
    return readyPromise;
  }

  async function score(text) {
    await ensureReady();
    const { vecs, ids } = await rawVectors(text);
    const v = centerAndNormalize(vecs);
    const sims = bankVecs.map((b) => {
      let total = 0;
      for (let li = 0; li < LAYERS.length; li++) {
        let dot = 0;
        for (let d = 0; d < v[li].length; d++) dot += b[li][d] * v[li][d];
        total += dot;
      }
      return total / LAYERS.length;
    });

    const folded = fold(text);
    const ranking = COMMANDS.map((cmd) => {
      const own = items.map((it, i) => ({ ...it, sim: sims[i] })).filter((it) => it.cmd === cmd);
      own.sort((a, b) => b.sim - a.sim);
      const sim = own.slice(0, TOP_K).reduce((acc, it) => acc + it.sim, 0) / TOP_K;
      const keyword = keywords[cmd].some((k) => folded.includes(" " + k));
      return { cmd, sim, keyword, score: sim + (keyword ? KEYWORD_BONUS : 0), nearest: own[0].text };
    }).sort((a, b) => b.score - a.score);

    const [first, second] = ranking;
    const decision =
      first.score >= execMin && first.score - second.score >= margin ? "exec"
      : first.score >= ASK_MIN ? "ask"
      : "reject";

    return { text, tokens: ids.map((id) => tokenizer.decode([id])), ranking, decision };
  }

  return {
    size: items.length,
    ensureReady,
    score,
    onProgress: (fn) => progressListeners.add(fn),
  };
}

// "acrescente a comida e o sistema de batida" → duas partes, um comando cada.
const CLAUSE_SPLIT = {
  pt: /\s*(?:[,;.]|\s+(?:e|depois|também|tambem|além disso|alem disso)\s+)\s*/i,
  en: /\s*(?:[,;.]|\s+(?:and|then|also)\s+)\s*/i,
};

async function interpret(router, text, lang) {
  const whole = await router.score(text);
  const parts = text.split(CLAUSE_SPLIT[lang]).map((p) => p.trim()).filter((p) => p.replace(/[^\p{L}]/gu, "").length >= 3);
  if (parts.length >= 2) {
    const analyses = [];
    for (const part of parts) analyses.push(await router.score(part)); // sequencial: o ONNX não roda em paralelo
    const commands = [...new Set(analyses.filter((a) => a.decision === "exec").map((a) => a.ranking[0].cmd))];
    // Partes em dúvida viram pergunta — a não ser que apontem pra algo que já vai rodar.
    const pending = analyses.filter((a) => a.decision === "ask" && !commands.includes(a.ranking[0].cmd));
    const useParts = commands.length >= 2 || (commands.length === 1 && (pending.length > 0 || whole.decision !== "exec"));
    if (useParts) return { kind: "exec", commands, analyses, pending, split: true };
  }
  return {
    kind: whole.decision,
    commands: whole.decision === "exec" ? [whole.ranking[0].cmd] : [],
    analyses: [whole],
    pending: [],
    split: false,
  };
}

// ============================================================================
// O JOGO — um único motor; cada recurso/bug é uma opção liga/desliga
// ============================================================================
const GRID = 20;
const SIZE = 400;
const START_SNAKE = () => [{ x: 10, y: 10 }, { x: 9, y: 10 }, { x: 8, y: 10 }, { x: 7, y: 10 }];
const KEY_DIRS = {
  ArrowUp: { x: 0, y: -1 }, ArrowDown: { x: 0, y: 1 }, ArrowLeft: { x: -1, y: 0 }, ArrowRight: { x: 1, y: 0 },
  w: { x: 0, y: -1 }, s: { x: 0, y: 1 }, a: { x: -1, y: 0 }, d: { x: 1, y: 0 },
};

class SnakeGame {
  constructor(els, text) {
    this.els = els;
    this.text = text;
    this.cell = SIZE / GRID;
    const dpr = window.devicePixelRatio || 1;
    els.canvas.width = SIZE * dpr;
    els.canvas.height = SIZE * dpr;
    this.ctx = els.canvas.getContext("2d");
    this.ctx.scale(dpr, dpr);
    this.cfg = { created: false, features: new Set(), bugs: new Set() };
    this.best = 0;
    this.score = 0;
    this.focused = false;
    this.timer = null;
    this.toastTimer = null;
    this.bindInput();
    this.reset(true);
  }

  has(feature) { return this.cfg.features.has(feature); }
  bug(id) { return this.cfg.bugs.has(id); }

  configure(cfg) {
    this.cfg = { created: cfg.created, features: new Set(cfg.features), bugs: new Set(cfg.bugs) };
    this.reset(true);
  }

  reset(full) {
    clearTimeout(this.timer);
    this.snake = START_SNAKE();
    this.dir = { x: 1, y: 0 };
    this.queue = [];
    this.interval = 150;
    if (full || !this.bug("scoreSticky")) this.score = 0;
    this.offscreenTicks = 0;
    this.food = this.has("comida") ? this.spawnFood() : null;
    this.status = this.cfg.created ? "ready" : "empty";
    this.clear(true);
    this.draw();
    this.updateOverlay();
    this.updateHud();
    this.els.pauseBtn.style.display = this.has("pausa") ? "" : "none";
  }

  spawnFood() {
    const onSnake = (c) => this.snake.some((p) => p.x === c.x && p.y === c.y);
    if (this.bug("foodInSnake") && Math.random() < 0.35) {
      // Bug: nunca checa a cobra, e aqui a gente "ajuda" o azar a aparecer.
      const body = this.snake.slice(1).filter((p) => p.x >= 0 && p.y >= 0 && p.x < GRID && p.y < GRID);
      if (body.length) return { ...body[Math.floor(Math.random() * body.length)] };
    }
    let cell;
    do {
      cell = { x: Math.floor(Math.random() * GRID), y: Math.floor(Math.random() * GRID) };
    } while (!this.bug("foodInSnake") && onSnake(cell));
    return cell;
  }

  start() {
    this.status = "running";
    this.updateOverlay();
    this.schedule();
  }

  schedule() {
    clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      if (this.status !== "running") return;
      this.step();
      if (this.status === "running") this.schedule();
    }, this.interval);
  }

  step() {
    if (this.queue.length) this.dir = this.queue.shift();
    const head = { x: this.snake[0].x + this.dir.x, y: this.snake[0].y + this.dir.y };

    if (this.has("portal")) {
      if (head.x >= GRID) {
        head.x = 0;
        if (this.bug("portalShift")) head.y = (head.y + 1) % GRID;
      } else if (head.x < 0) head.x = GRID - 1;
      if (head.y >= GRID) head.y = 0;
      else if (head.y < 0) head.y = GRID - 1;
    }

    if (this.has("parede")) {
      const limit = this.bug("wallEarly") ? GRID - 1 : GRID;
      if (head.x < 0 || head.y < 0 || head.x >= limit || head.y >= limit) return this.gameOver();
    }

    const eats = this.food && head.x === this.food.x && head.y === this.food.y;
    if (this.has("corpo")) {
      const body = eats ? this.snake : this.snake.slice(0, -1);
      if (body.some((p) => p.x === head.x && p.y === head.y)) return this.gameOver();
    }

    this.snake.unshift(head);
    if (eats) {
      this.score += 1;
      this.best = Math.max(this.best, this.score);
      if (this.has("velocidade")) {
        this.interval = this.bug("turbo") ? Math.max(25, this.interval * 0.7) : Math.max(70, this.interval * 0.95);
      }
      this.food = this.spawnFood();
    } else {
      this.snake.pop();
    }

    // Sem parede nem portal, a cobra simplesmente some da tela.
    if (!this.has("parede") && !this.has("portal")) {
      const visible = this.snake.some((p) => p.x >= 0 && p.y >= 0 && p.x < GRID && p.y < GRID);
      if (!visible) {
        this.offscreenTicks++;
        if (this.offscreenTicks === 1) this.toast(this.text.fled);
        if (this.offscreenTicks > 10) {
          this.snake = START_SNAKE();
          this.dir = { x: 1, y: 0 };
          this.queue = [];
          this.offscreenTicks = 0;
          this.clear(true);
        }
      }
    }

    this.draw();
    this.updateHud();
  }

  turn(next) {
    if (this.status === "empty" || this.status === "over" || this.status === "paused") return;
    const last = this.queue.length ? this.queue[this.queue.length - 1] : this.dir;
    const isReverse = next.x === -last.x && next.y === -last.y;
    if (isReverse && !this.bug("reverse")) return;
    if (next.x !== last.x || next.y !== last.y) {
      if (this.queue.length < 3) this.queue.push(next);
    }
    if (this.status === "ready") this.start();
  }

  togglePause() {
    if (!this.has("pausa")) return;
    if (this.status === "running") {
      this.status = "paused";
      this.pausedAt = performance.now();
      clearTimeout(this.timer);
    } else if (this.status === "paused") {
      this.status = "running";
      if (this.bug("pauseJump")) {
        // Bug: "recupera" os passos que teriam acontecido durante a pausa.
        const missed = Math.min(6, Math.floor((performance.now() - this.pausedAt) / this.interval));
        for (let i = 0; i < missed && this.status === "running"; i++) this.step();
      }
      if (this.status === "running") this.schedule();
    }
    this.updateOverlay();
  }

  gameOver() {
    this.status = "over";
    clearTimeout(this.timer);
    this.draw();
    this.updateOverlay();
    this.updateHud();
  }

  restart() {
    this.reset(false);
  }

  // ---------- Desenho ----------
  clear(force) {
    if (!force && this.bug("trail")) return; // Bug: nunca apaga o quadro anterior
    const ctx = this.ctx;
    const fancy = this.has("cores");
    ctx.shadowBlur = 0;
    ctx.fillStyle = fancy ? "#12161F" : "#050505";
    ctx.fillRect(0, 0, SIZE, SIZE);
    if (fancy) {
      ctx.strokeStyle = "#1B2130";
      ctx.lineWidth = 1;
      for (let i = 1; i < GRID; i++) {
        ctx.beginPath(); ctx.moveTo(i * this.cell + 0.5, 0); ctx.lineTo(i * this.cell + 0.5, SIZE); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, i * this.cell + 0.5); ctx.lineTo(SIZE, i * this.cell + 0.5); ctx.stroke();
      }
    }
  }

  draw() {
    this.clear(false);
    if (!this.cfg.created) return;
    const ctx = this.ctx;
    const c = this.cell;
    const off = this.bug("offset") ? c / 2 : 0;

    if (this.has("cores")) {
      if (this.food) {
        ctx.save();
        ctx.fillStyle = ctx.shadowColor = "#F0A868";
        ctx.shadowBlur = 12;
        ctx.beginPath();
        ctx.arc(this.food.x * c + c / 2 + off, this.food.y * c + c / 2 + off, c / 2.6, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
      const n = this.snake.length;
      // Desenha do rabo pra cabeça, pra cabeça ficar por cima quando se sobrepõe.
      for (let i = n - 1; i >= 0; i--) {
        const p = this.snake[i];
        ctx.fillStyle = mixColor("#4FD1C5", "#2C7A7B", n > 1 ? i / (n - 1) : 0);
        roundRect(ctx, p.x * c + 1 + off, p.y * c + 1 + off, c - 2, c - 2, 6);
      }
      this.drawEyes(this.snake[0], off);
    } else {
      if (this.food) {
        ctx.fillStyle = "red";
        ctx.fillRect(this.food.x * c + off, this.food.y * c + off, c, c);
      }
      ctx.fillStyle = "lime";
      this.snake.forEach((p) => ctx.fillRect(p.x * c + off, p.y * c + off, c, c));
    }
  }

  drawEyes(head, off) {
    const ctx = this.ctx;
    const c = this.cell;
    const cx = head.x * c + c / 2 + off;
    const cy = head.y * c + c / 2 + off;
    const { x: dx, y: dy } = this.dir;
    const side = { x: -dy, y: dx };
    [-1, 1].forEach((s) => {
      const ex = cx + dx * 3 + side.x * s * 4.5;
      const ey = cy + dy * 3 + side.y * s * 4.5;
      ctx.fillStyle = "#0A0E14";
      ctx.beginPath(); ctx.arc(ex, ey, 2.6, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#E4E7EC";
      ctx.beginPath(); ctx.arc(ex + dx, ey + dy, 1.1, 0, Math.PI * 2); ctx.fill();
    });
  }

  // ---------- HUD / overlays (HTML por cima do canvas — não sujam com o bug do rastro) ----------
  updateHud() {
    const hud = this.els.hud;
    if (!this.cfg.created || !this.has("placar")) {
      hud.style.display = "none";
      return;
    }
    hud.style.display = "block";
    hud.textContent = `${this.text.score} ${this.score} · ${this.text.best} ${this.best}`;
  }

  updateOverlay() {
    const o = this.els.overlay;
    let main = "";
    let sub = "";
    if (this.status === "empty") main = this.text.empty;
    else if (this.status === "ready") main = this.focused ? this.text.pressArrow : this.text.clickToPlay;
    else if (this.status === "paused") main = this.text.paused;
    else if (this.status === "over") {
      main = this.text.gameOver;
      sub = this.has("placar") ? `${this.text.score}: ${this.score}` : "";
      sub = sub ? `${sub} — ${this.text.restartHint}` : this.text.restartHint;
    }
    o.style.display = main ? "flex" : "none";
    o.classList.toggle("dim", this.status === "over" || this.status === "paused" || this.status === "empty");
    o.innerHTML = "";
    const strong = document.createElement("strong");
    strong.textContent = main;
    o.appendChild(strong);
    if (sub) {
      const span = document.createElement("span");
      span.textContent = sub;
      o.appendChild(span);
    }
  }

  toast(message) {
    const el = this.els.toast;
    el.textContent = message;
    el.classList.add("show");
    clearTimeout(this.toastTimer);
    this.toastTimer = setTimeout(() => el.classList.remove("show"), 2600);
  }

  // ---------- Entrada (teclado só com o jogo em foco, pra não rolar a página) ----------
  bindInput() {
    const wrap = this.els.wrap;
    wrap.addEventListener("focus", () => { this.focused = true; this.updateOverlay(); });
    wrap.addEventListener("blur", () => { this.focused = false; this.updateOverlay(); });
    wrap.addEventListener("click", () => {
      wrap.focus();
      if (this.status === "over") this.restart();
    });
    wrap.addEventListener("keydown", (e) => {
      const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
      if (KEY_DIRS[key]) {
        e.preventDefault();
        this.turn(KEY_DIRS[key]);
      } else if (key === " " || key === "p") {
        e.preventDefault();
        if (this.status === "over") this.restart();
        else this.togglePause();
      } else if (key === "Enter") {
        e.preventDefault();
        if (this.status === "over") this.restart();
      }
    });

    let touchStart = null;
    wrap.addEventListener("touchstart", (e) => {
      const t = e.touches[0];
      touchStart = { x: t.clientX, y: t.clientY };
    }, { passive: true });
    wrap.addEventListener("touchend", (e) => {
      if (!touchStart) return;
      const t = e.changedTouches[0];
      const dx = t.clientX - touchStart.x;
      const dy = t.clientY - touchStart.y;
      touchStart = null;
      if (Math.max(Math.abs(dx), Math.abs(dy)) < 24) return;
      this.turn(Math.abs(dx) > Math.abs(dy) ? { x: Math.sign(dx), y: 0 } : { x: 0, y: Math.sign(dy) });
    });

    this.els.dpad.querySelectorAll("[data-dir]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const [x, y] = btn.dataset.dir.split(",").map(Number);
        if (this.status === "over") this.restart();
        else this.turn({ x, y });
      });
    });
    this.els.pauseBtn.addEventListener("click", () => this.togglePause());
  }
}

function mixColor(hexA, hexB, t) {
  const a = parseInt(hexA.slice(1), 16);
  const b = parseInt(hexB.slice(1), 16);
  const ch = (shift) => Math.round(((a >> shift) & 255) + ((((b >> shift) & 255) - ((a >> shift) & 255)) * t));
  return `rgb(${ch(16)}, ${ch(8)}, ${ch(0)})`;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
  ctx.fill();
}

// ============================================================================
// INTERFACE — chat, painel "por baixo do capô", código e versões
// ============================================================================
export function initSimulator({ tokenizer, model, lang, mode }) {
  const T = SIM_TEXT[lang];
  const $ = (id) => document.getElementById(id);
  const section = $("section-sim");
  const messages = $("sim-messages");
  const input = $("sim-input");
  const sendBtn = $("sim-send");
  const status = $("sim-status");
  const codeEl = $("sim-code");
  const codeTitle = $("sim-code-title");
  const versionsEl = $("sim-versions");
  const insightEl = $("sim-insight");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  input.placeholder = T.placeholder;
  sendBtn.textContent = T.send;
  status.textContent = T.bankIdle;
  $("sim-versions-label").textContent = T.versions;
  codeTitle.textContent = T.codeTitle;
  codeEl.textContent = T.codeEmpty;
  codeEl.classList.add("empty");

  const router = createRouter({ tokenizer, model, lang });
  router.onProgress((n, total) => {
    status.textContent = n < total ? T.bankLoading(n, total) : T.bankReady;
  });

  const game = new SnakeGame({
    wrap: $("sim-game"),
    canvas: $("sim-canvas"),
    hud: $("sim-hud"),
    overlay: $("sim-overlay"),
    toast: $("sim-toast"),
    dpad: $("sim-dpad"),
    pauseBtn: $("sim-pause-btn"),
  }, T.game);

  let cfg = { created: false, features: [], bugs: [] };
  const versions = [];
  let activeVersion = -1;
  let busy = false;

  // Começa a "ler" os exemplos assim que a seção aparece na tela (em vez de
  // atrasar a primeira resposta). O modelo é serializado em script.js, então
  // isso não colide com as outras seções.
  const warmUp = () => router.ensureReady().catch((err) => console.error(err));
  new IntersectionObserver((entries, obs) => {
    if (entries.some((e) => e.isIntersecting)) {
      obs.disconnect();
      warmUp();
    }
  }).observe(section);
  input.addEventListener("focus", warmUp, { once: true });

  // ---------- Mensagens ----------
  function scrollChat() {
    messages.scrollTop = messages.scrollHeight;
  }

  function addMessage(who, text) {
    const msg = document.createElement("div");
    msg.className = `sim-msg ${who}`;
    if (text) {
      const p = document.createElement("p");
      p.textContent = text;
      msg.appendChild(p);
    }
    messages.appendChild(msg);
    scrollChat();
    return msg;
  }

  function addParagraph(msg, text) {
    const p = document.createElement("p");
    p.textContent = text;
    msg.appendChild(p);
    scrollChat();
    return p;
  }

  function addList(msg, lines) {
    const ul = document.createElement("ul");
    lines.forEach((line) => {
      const li = document.createElement("li");
      li.textContent = line;
      ul.appendChild(li);
    });
    msg.appendChild(ul);
    scrollChat();
  }

  function addChips(msg, chips) {
    const row = document.createElement("div");
    row.className = "sim-chips";
    chips.forEach(({ label, onClick }) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "example-chip";
      b.textContent = label;
      b.addEventListener("click", () => {
        if (busy) return;
        row.querySelectorAll("button").forEach((x) => { x.disabled = true; });
        onClick();
      });
      row.appendChild(b);
    });
    msg.appendChild(row);
    scrollChat();
  }

  const sendChip = (cmd) => ({ label: T.suggestions[cmd], onClick: () => handleRequest(T.suggestions[cmd]) });

  // Empate (folga pequena entre 1º e 2º) ou pontuação baixa demais pra executar?
  const isTie = (a) => a.ranking[0].score - a.ranking[1].score < THRESHOLDS[lang].margin;

  function askText(analysis, part) {
    const [a, b] = analysis.ranking;
    return isTie(analysis)
      ? T.askTie(T.commandLabels[a.cmd], T.commandLabels[b.cmd], part)
      : T.askLow(T.commandLabels[a.cmd], part);
  }

  // ---------- Quadro de insight: explica a última decisão com os números reais ----------
  let lastResult = null;
  function renderInsight() {
    if (!lastResult) return;
    const mode = section.classList.contains("sim-tech") ? "technical" : "simple";
    const texts = T.insight[mode];
    const { execMin, margin } = THRESHOLDS[lang];
    const base = { margin: T.num(margin), execMin: T.num(execMin), askMin: T.num(ASK_MIN) };
    let text;
    if (lastResult.split) {
      const parts = lastResult.analyses
        .filter((a) => a.decision === "exec")
        .map((a) => ({ text: a.text, label: T.commandLabels[a.ranking[0].cmd], score: T.num(a.ranking[0].score) }));
      text = texts.split({ parts });
    } else {
      const a = lastResult.analyses[0];
      const [first, second] = a.ranking;
      const gap = first.score - second.score;
      const data = {
        ...base,
        top: T.commandLabels[first.cmd], s1: T.num(first.score),
        second: T.commandLabels[second.cmd], s2: T.num(second.score),
        gap: T.num(gap),
      };
      if (a.decision === "exec") text = gap >= 0.2 ? texts.clear(data) : texts.close(data);
      else if (a.decision === "ask") text = isTie(a) ? texts.tie(data) : texts.low(data);
      else text = texts.reject(data);
    }
    insightEl.textContent = text;
    insightEl.style.display = "block";
  }

  // Quando o modelo fica em dúvida, a pessoa escolhe entre os 3 primeiros.
  function addChoiceChips(msg, analysis) {
    addCommandChips(msg, analysis.ranking.slice(0, 3).map(({ cmd }) => cmd));
  }

  // Escolheu algo que já está no jogo? Oferece as próximas opções do ranking
  // (ex: "mais rápido a cada comida" → 🍎 ganhou, mas ⚡ era o que se queria).
  function offerAlternatives(msg, analysis) {
    const unavailable = (cmd) =>
      cfg.features.includes(cmd) || (cmd === "criar" && cfg.created) || (cmd === "corrigir" && !cfg.bugs.length);
    const alternatives = analysis.ranking.slice(1, 3).filter((r) => r.score >= ASK_MIN && !unavailable(r.cmd));
    if (!alternatives.length) return false;
    addParagraph(msg, T.alreadyAlt);
    addCommandChips(msg, alternatives.map((r) => r.cmd));
    return true;
  }

  function addCommandChips(msg, cmds) {
    addChips(msg, cmds.map((cmd) => ({
      label: T.commandLabels[cmd],
      onClick: async () => {
        busy = true;
        sendBtn.disabled = true;
        const reply = addMessage("bot");
        await runCommand(cmd, reply);
        addChips(reply, nextSuggestions());
        busy = false;
        sendBtn.disabled = false;
      },
    })));
  }

  function nextSuggestions() {
    if (!cfg.created) return [sendChip("criar")];
    const missing = FEATURE_ORDER.filter((f) => !cfg.features.includes(f) && !(f === "portal" && cfg.features.includes("parede")));
    const chips = missing.slice(0, 2).map(sendChip);
    if (cfg.bugs.length) chips.push(sendChip("corrigir"));
    return chips;
  }

  // ---------- Painel "por baixo do capô" ----------
  function renderAnalysis(container, analysis, chosenCmd) {
    const block = document.createElement("div");
    block.className = "sim-analysis";

    if (analysis.part) {
      const q = document.createElement("p");
      q.className = "sim-hood-part";
      q.textContent = `«${analysis.text}»`;
      block.appendChild(q);
    }

    const tokLabel = document.createElement("p");
    tokLabel.className = "sim-hood-label";
    tokLabel.textContent = T.hoodTokens;
    const toks = document.createElement("div");
    toks.className = "sim-tokens";
    analysis.tokens.forEach((tk) => {
      const chip = document.createElement("span");
      chip.className = "sim-token";
      chip.textContent = tk;
      toks.appendChild(chip);
    });
    block.append(tokLabel, toks);

    const scoresLabel = document.createElement("p");
    scoresLabel.className = "sim-hood-label";
    scoresLabel.textContent = T.hoodScores;
    block.appendChild(scoresLabel);
    const maxScore = Math.max(0.01, analysis.ranking[0].score);
    analysis.ranking.forEach((r, i) => {
      const row = document.createElement("div");
      row.className = "sim-score-row" + (i >= 3 ? " tech-only" : "") + (r.cmd === chosenCmd ? " chosen" : "");
      const label = document.createElement("span");
      label.className = "sim-score-label";
      label.textContent = T.commandLabels[r.cmd];
      const barBg = document.createElement("div");
      barBg.className = "prob-bar-bg";
      const fill = document.createElement("div");
      fill.className = "prob-bar-fill";
      fill.style.width = `${Math.max(0, r.score / maxScore) * 100}%`;
      barBg.appendChild(fill);
      // Iniciante: só a ★; Técnico: similaridade + bônus, decompostos.
      const val = document.createElement("span");
      val.className = "sim-score-val";
      const simpleVal = document.createElement("span");
      simpleVal.className = "simple-only";
      simpleVal.textContent = r.keyword ? "★" : "";
      const techVal = document.createElement("span");
      techVal.className = "tech-only";
      techVal.textContent = r.keyword ? `${T.num(r.sim)} + ${T.num(KEYWORD_BONUS)}★` : T.num(r.sim);
      val.append(simpleVal, techVal);
      row.append(label, barBg, val);
      block.appendChild(row);
    });

    if (analysis.ranking.slice(0, 3).some((r) => r.keyword) && !container.querySelector(".sim-star-legend")) {
      const legend = document.createElement("p");
      legend.className = "sim-hood-note simple-only sim-star-legend";
      legend.textContent = T.starLegend;
      block.appendChild(legend);
    }

    const nearest = document.createElement("p");
    nearest.className = "sim-hood-note";
    nearest.textContent = T.hoodNearest(analysis.ranking[0].nearest);
    block.appendChild(nearest);

    const boosted = analysis.ranking.filter((r) => r.keyword).map((r) => T.commandLabels[r.cmd]);
    if (boosted.length) {
      const kw = document.createElement("p");
      kw.className = "sim-hood-note tech-only";
      kw.textContent = `★ ${T.hoodKeyword(boosted.join(", "))} (+${T.num(KEYWORD_BONUS)})`;
      block.appendChild(kw);
    }

    const callLabel = document.createElement("p");
    callLabel.className = "sim-hood-label";
    callLabel.textContent = T.hoodCall;
    const call = document.createElement("code");
    call.className = "sim-call";
    call.textContent = chosenCmd ? JSON.stringify(TOOL_CALL[chosenCmd]) : analysis.decision === "ask" ? T.hoodUnsure : T.hoodNone;
    block.append(callLabel, call);

    container.appendChild(block);
  }

  function addHood(msg, result, chosen) {
    const details = document.createElement("details");
    details.className = "sim-hood";
    details.open = section.classList.contains("sim-tech");
    const summary = document.createElement("summary");
    summary.textContent = T.hood;
    details.appendChild(summary);
    if (result.split) {
      const note = document.createElement("p");
      note.className = "sim-hood-note";
      note.textContent = T.hoodSplit(result.analyses.length);
      details.appendChild(note);
    }
    result.analyses.forEach((a) => {
      const cmd = a.decision === "exec" ? a.ranking[0].cmd : chosen && !result.split ? chosen : null;
      renderAnalysis(details, { ...a, part: result.split }, cmd);
    });
    const rule = document.createElement("p");
    rule.className = "sim-hood-note tech-only";
    rule.textContent = T.hoodRule({ ...THRESHOLDS[lang], askMin: ASK_MIN });
    details.appendChild(rule);
    msg.appendChild(details);
    scrollChat();
    return details;
  }

  // ---------- Código ----------
  async function showCode(title, diff) {
    codeTitle.textContent = title;
    codeEl.classList.remove("empty");
    codeEl.innerHTML = "";
    const lines = diff.split("\n");
    for (const line of lines) {
      const div = document.createElement("div");
      const mark = line[0];
      div.className = "sim-code-line " + (mark === "+" ? "add" : mark === "-" ? "del" : "ctx");
      div.textContent = line;
      codeEl.appendChild(div);
      if (!reducedMotion) {
        codeEl.scrollTop = codeEl.scrollHeight;
        await new Promise((r) => setTimeout(r, 28));
      }
    }
    codeEl.scrollTop = 0;
  }

  // ---------- Versões ----------
  function renderVersions() {
    versionsEl.innerHTML = "";
    versions.forEach((v, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "sim-version" + (i === activeVersion ? " active" : "");
      b.textContent = v.name;
      b.title = v.label;
      b.addEventListener("click", () => {
        if (busy) return;
        activeVersion = i;
        cfg = structuredClone(v.cfg);
        game.configure(cfg);
        renderVersions();
        showCode(`${T.codeTitle} — ${v.name}`, v.diff);
      });
      versionsEl.appendChild(b);
    });
  }

  function pushVersion(label, diff) {
    const name = `v0.${versions.length + 1}`;
    versions.push({ name, label, cfg: structuredClone(cfg), diff });
    activeVersion = versions.length - 1;
    renderVersions();
    return name;
  }

  // ---------- Executar um comando (só liga opções que já existem) ----------
  async function runCommand(cmd, msg) {
    const has = (f) => cfg.features.includes(f);

    if (cmd !== "criar" && !cfg.created) {
      addParagraph(msg, T.needCreate);
      return false;
    }
    if (cmd === "criar" && cfg.created) {
      addParagraph(msg, T.alreadyCreated);
      return "already";
    }
    if (cmd !== "criar" && cmd !== "corrigir" && has(cmd)) {
      addParagraph(msg, T.already(T.commandLabels[cmd]));
      return "already";
    }

    if (cmd === "corrigir") {
      const fixed = [...cfg.bugs];
      if (!fixed.length) {
        addParagraph(msg, T.fixNone);
        return false;
      }
      cfg = { ...cfg, bugs: [] };
      const diff = fixed.map((b) => `  // ── ${T.bugs[b]}\n${FIX_CODE[b]}`).join("\n\n");
      const name = pushVersion(T.commandLabels[cmd], diff);
      await showCode(`${T.codeTitle} — ${name} (${T.codeFixTitle})`, diff);
      game.configure(cfg);
      addParagraph(msg, T.fixDone(fixed.length));
      addList(msg, fixed.map((b) => T.bugs[b]));
      return true;
    }

    const ctx = {
      hadPortal: cmd === "parede" && has("portal"),
      hadWall: cmd === "portal" && has("parede"),
      hasReverse: cfg.bugs.includes("reverse"),
      hasFood: has("comida"),
      fixedOffset: cmd === "cores" && cfg.bugs.includes("offset"),
    };
    let features = cmd === "criar" ? [] : [...cfg.features, cmd];
    let bugs = [...cfg.bugs, ...BUGS_FROM[cmd]];
    if (cmd === "parede") features = features.filter((f) => f !== "portal");
    if (cmd === "portal") features = features.filter((f) => f !== "parede");
    if (ctx.hadPortal) bugs = bugs.filter((b) => b !== "portalShift");
    if (ctx.hadWall) bugs = bugs.filter((b) => b !== "wallEarly");
    if (cmd === "cores") bugs = bugs.filter((b) => b !== "offset"); // o desenho novo não tem o deslocamento
    cfg = { created: true, features, bugs: [...new Set(bugs)] };

    const name = pushVersion(T.commandLabels[cmd], CODE[cmd]);
    await showCode(`${T.codeTitle} — ${name}`, CODE[cmd]);
    game.configure(cfg);
    addParagraph(msg, T.replies[cmd](ctx));
    return true;
  }

  // ---------- Fluxo principal ----------
  async function handleRequest(text) {
    if (busy || !text.trim()) return;
    busy = true;
    sendBtn.disabled = true;
    addMessage("user", text);
    const msg = addMessage("bot");
    const thinking = addParagraph(msg, T.thinking);
    thinking.classList.add("sim-thinking");

    try {
      const result = await interpret(router, text, lang);
      thinking.remove();
      lastResult = result;
      renderInsight();

      if (result.kind === "exec") {
        addHood(msg, result, result.commands[0]);
        let offered = false;
        for (const cmd of result.commands) {
          const outcome = await runCommand(cmd, msg);
          if (outcome === "already" && !result.split) offered = offerAlternatives(msg, result.analyses[0]);
        }
        result.pending.forEach((a) => {
          addParagraph(msg, askText(a, a.text));
          addChoiceChips(msg, a);
        });
        if (!result.pending.length && !offered) addChips(msg, nextSuggestions());
      } else if (result.kind === "ask") {
        addHood(msg, result, null);
        addParagraph(msg, askText(result.analyses[0]));
        addChoiceChips(msg, result.analyses[0]);
      } else {
        addHood(msg, result, null);
        addParagraph(msg, T.reject);
        addChips(msg, nextSuggestions());
      }
    } catch (err) {
      console.error(err);
      thinking.textContent = "⚠️ " + (lang === "pt" ? "Algo deu errado ao rodar o modelo." : "Something went wrong running the model.");
    } finally {
      busy = false;
      sendBtn.disabled = false;
    }
  }

  $("sim-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const text = input.value;
    input.value = "";
    handleRequest(text);
  });

  const intro = addMessage("bot", T.intro);
  addChips(intro, nextSuggestions());

  function setMode(m) {
    section.classList.toggle("sim-tech", m === "technical");
    renderInsight();
  }
  setMode(mode);

  return { setMode };
}
