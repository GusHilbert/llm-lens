// ============================================================================
// CONFIGURAÇÃO
// ============================================================================
// Os dois idiomas apontam para o MESMO modelo por enquanto (o em português).
// TROQUE "MODEL_IDS.en" pelo modelo em inglês assim que ele for exportado.
const MODEL_IDS = {
  pt: "GusHil/gpt2-small-portuguese-onnx",
  en: "GusHil/gpt2-small-portuguese-onnx", // TODO: trocar pelo modelo em inglês
};

import {
  AutoTokenizer,
  AutoModelForCausalLM,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0";

env.allowLocalModels = false;

// ============================================================================
// IDIOMA ATUAL (via parâmetro da URL, ex: app.html?lang=pt)
// ============================================================================
const params = new URLSearchParams(window.location.search);
const LANG = params.get("lang") === "en" ? "en" : "pt";
document.getElementById("html-root").lang = LANG === "pt" ? "pt-br" : "en";

// ============================================================================
// TEXTOS ESTÁTICOS DA INTERFACE
// ============================================================================
const UI = {
  pt: {
    loadingTitle: "Preparando seu modelo…",
    loadingNote: "Isso só acontece na primeira vez — depois fica salvo no seu navegador. Tudo roda localmente, nenhum dado sai da sua máquina.",
    inputEyebrow: "DIGITE UMA FRASE",
    analyzeBtn: "Analisar",
    defaultPrompt: "O rato que o gato perseguiu fugiu",
    defaultWrite: "Era uma vez",
    pipelineText: "Texto",
    pipelineToken: "Token",
    pipelineEmbedding: "Embedding",
    pipelineAttention: "Atenção",
    pipelinePrediction: "Previsão",
    examples: [
      "O rato que o gato perseguiu fugiu",
      "A inteligência artificial no Brasil",
      "Ela disse que ele nunca voltaria",
    ],
    tokensTitle: "Tokenização",
    tokensHint: "Cada bloco é um token — a unidade mínima que o modelo enxerga.",
    attentionTitle: "Atenção",
    attentionHint: "Escolha uma camada e uma cabeça para ver para onde cada token \"olha\".",
    attentionBreakdownTitle: "Para onde cada token olha (top 3)",
    layerLabel: "Camada",
    headLabel: "Cabeça",
    nextTitle: "Próxima palavra",
    nextHint: "A distribuição de probabilidade que o modelo calculou para a próxima palavra. Clique numa barra para continuar a frase com ela.",
    writeTitle: "Escreva junto com o modelo",
    writeHint: "Continue uma frase e deixe o modelo sugerir a próxima palavra, uma de cada vez.",
    writeStart: "Começar",
    writeReset: "Recomeçar",
    footer: "Projeto de código aberto — TCC Pós-Graduação IA Generativa & LLMs (PUC-Rio / ICA).",
    sobreLink: "sobre.html?lang=pt",
    errorMsg: "Algo deu errado ao carregar ou rodar o modelo. Abra o Console (F12) para detalhes.",
    aboutRole: "TCC — Pós-Graduação em Inteligência Artificial Generativa & LLMs, PUC-Rio / Laboratório ICA",
    aboutProjectHeading: "Sobre o projeto",
    aboutProjectText: "O LLM Lens é uma ferramenta interativa de código aberto que mostra, passo a passo, como um modelo de linguagem processa e gera texto — tokenização, atenção e previsão de próxima palavra — rodando inteiramente no navegador, sem nenhum servidor por trás.",
    aboutStackHeading: "Stack técnica",
    lessons: {
      tokens: {
        simple: "Um modelo de linguagem não lê palavras inteiras como nós — ele quebra o texto em pedacinhos chamados tokens antes de processar qualquer coisa. Às vezes um token é uma palavra inteira, às vezes é só um pedaço dela, dependendo do quão comum ela foi durante o treinamento. É como se o modelo enxergasse tudo através de um vocabulário fixo de dezenas de milhares de \"peças de Lego\", e qualquer texto precisa ser remontado a partir delas.",
        technical: "A tokenização usa Byte-Pair Encoding (BPE): um vocabulário fixo de subpalavras (aqui, ~50 mil unidades) construído a partir da frequência estatística de pares de caracteres no corpus de treinamento. Palavras frequentes viram um único token; palavras raras ou compostas são fragmentadas em múltiplas subpalavras. É a primeira etapa do pipeline — o modelo nunca vê caracteres soltos, só os IDs desses tokens.",
      },
      attention: {
        simple: "Atenção é o mecanismo que permite ao modelo \"olhar para trás\" na frase enquanto processa cada palavra, decidindo quais outras são relevantes. É parecido com o que você faz ao ler \"ela disse que ele nunca voltaria\" — pra saber quem é \"ele\", seu cérebro busca automaticamente lá atrás por um nome que combine. O modelo faz algo parecido, várias vezes, em paralelo, em cada camada.",
        technical: "Self-attention calcula, para cada token, uma distribuição de pesos sobre todos os outros tokens da sequência, via projeções de Query, Key e Value seguidas de um softmax escalado. Cada camada tem múltiplas \"cabeças\" de atenção, cada uma podendo capturar um tipo diferente de relação (sintática, correferencial, posicional). Empilhando camadas, o modelo constrói representações cada vez mais contextuais.",
      },
      next: {
        simple: "Depois de processar a frase inteira, o modelo não \"escolhe\" a próxima palavra direto — ele calcula uma probabilidade pra cada palavra possível do vocabulário (todas as dezenas de milhares!) e normalmente escolhe uma das mais prováveis. É como se apostasse em várias palavras ao mesmo tempo, com fichas diferentes em cada uma.",
        technical: "A camada final produz um vetor de logits do tamanho do vocabulário, convertido em probabilidades via softmax. A escolha do próximo token pode ser determinística (greedy: sempre o de maior probabilidade) ou estocástica (sampling com temperature, top-k ou top-p) — o que afeta diretamente a criatividade vs. previsibilidade do texto gerado.",
      },
      write: {
        simple: "Aqui você usa exatamente o mesmo mecanismo da seção anterior, só que repetido várias vezes seguidas — cada palavra que você aceita vira parte do contexto pra calcular a próxima. É basicamente como funciona o autocomplete do seu celular, só que agora você está vendo por dentro.",
        technical: "Cada clique dispara um novo forward pass com o texto acumulado como entrada, recalculando toda a distribuição do zero (sem cache incremental nessa implementação). É a mesma geração autoregressiva usada em produção, só exposta passo a passo em vez de rodar automaticamente até um token de parada.",
      },
    },
    glossaryBasic: {
      token: "Pedaço de texto — pode ser uma palavra inteira ou um fragmento dela.",
      atenção: "Mecanismo que deixa o modelo \"olhar\" para outras palavras da frase ao processar uma palavra.",
      camada: "Um bloco de processamento (atenção + uma pequena rede neural) que o modelo repete várias vezes em sequência, refinando o entendimento da frase a cada passagem.",
      cabeças: "Uma das \"atenções\" que rodam em paralelo dentro da mesma camada — cada cabeça pode aprender a focar num tipo diferente de relação entre as palavras.",
      probabilidade: "O quão \"confiante\" o modelo está de que aquela é a próxima palavra certa.",
    },
    glossaryTechnical: {
      token: "Pedaço de texto — pode ser uma palavra inteira ou um fragmento dela.",
      atenção: "Mecanismo que deixa o modelo \"olhar\" para outras palavras da frase ao processar uma palavra.",
      camada: "Um bloco de processamento (atenção + uma pequena rede neural) que o modelo repete várias vezes em sequência, refinando o entendimento da frase a cada passagem.",
      cabeças: "Uma das \"atenções\" que rodam em paralelo dentro da mesma camada — cada cabeça pode aprender a focar num tipo diferente de relação entre as palavras.",
      probabilidade: "O quão \"confiante\" o modelo está de que aquela é a próxima palavra certa.",
      "tokenização": "Processo de quebrar o texto em tokens antes de qualquer processamento.",
      "Byte-Pair Encoding": "Algoritmo que constrói o vocabulário juntando repetidamente os pares de caracteres mais frequentes do corpus.",
      "subpalavras": "Pedaços de uma palavra menores que ela inteira, usados quando a palavra completa não está no vocabulário do tokenizador.",
      "corpus de treinamento": "O conjunto de textos usado para treinar o modelo.",
      "Self-attention": "Mecanismo que faz cada token se comparar com todos os outros da sequência para decidir onde prestar atenção.",
      "Query, Key e Value": "Três vetores calculados para cada token, usados para decidir o quanto ele deve prestar atenção nos demais.",
      "softmax": "Função que transforma uma lista de números em probabilidades que somam 100%.",
      "logits": "Os números brutos que o modelo calcula antes de virarem probabilidades.",
      "determinística": "Estratégia de geração em que o resultado é sempre o mesmo para a mesma entrada.",
      "greedy": "Estratégia que escolhe sempre a palavra mais provável, sem aleatoriedade.",
      "estocástica": "Estratégia de geração que envolve algum sorteio aleatório entre as opções mais prováveis.",
      "temperature": "Parâmetro que controla o quanto a escolha da próxima palavra é aleatória.",
      "top-k": "Estratégia que sorteia a próxima palavra só entre as k opções mais prováveis.",
      "top-p": "Estratégia que sorteia a próxima palavra entre as opções que juntas somam p% de probabilidade.",
      "forward pass": "Uma passada completa dos dados de entrada pelo modelo, do início ao fim, até gerar uma saída.",
      "cache incremental": "Técnica de guardar cálculos já feitos para não precisar refazê-los a cada novo token gerado.",
      "geração autoregressiva": "Gerar texto uma palavra de cada vez, sempre usando as palavras já geradas como parte da entrada.",
      "token de parada": "Um token especial que sinaliza ao modelo que ele deve parar de gerar texto.",
    },
  },
  en: {
    loadingTitle: "Getting your model ready…",
    loadingNote: "This only happens the first time — after that it's cached in your browser. Everything runs locally, no data ever leaves your machine.",
    inputEyebrow: "TYPE A SENTENCE",
    analyzeBtn: "Analyze",
    defaultPrompt: "The mouse that the cat chased ran away",
    defaultWrite: "Once upon a time",
    pipelineText: "Text",
    pipelineToken: "Token",
    pipelineEmbedding: "Embedding",
    pipelineAttention: "Attention",
    pipelinePrediction: "Prediction",
    examples: [
      "The mouse that the cat chased ran away",
      "Artificial intelligence in Brazil",
      "She said he would never come back",
    ],
    tokensTitle: "Tokenization",
    tokensHint: "Each block is a token — the smallest unit the model sees.",
    attentionTitle: "Attention",
    attentionHint: "Pick a layer and a head to see where each token \"looks\".",
    attentionBreakdownTitle: "Where each token looks (top 3)",
    layerLabel: "Layer",
    headLabel: "Head",
    nextTitle: "Next word",
    nextHint: "The probability distribution the model computed for the next word. Click a bar to continue the sentence with it.",
    writeTitle: "Write together with the model",
    writeHint: "Continue a sentence and let the model suggest the next word, one at a time.",
    writeStart: "Start",
    writeReset: "Reset",
    footer: "Open-source project — Capstone, Generative AI & LLMs Postgrad (PUC-Rio / ICA).",
    sobreLink: "sobre.html?lang=en",
    errorMsg: "Something went wrong loading or running the model. Open the Console (F12) for details.",
    aboutRole: "Capstone Project — Postgraduate in Generative AI & LLMs, PUC-Rio / ICA Lab",
    aboutProjectHeading: "About the project",
    aboutProjectText: "LLM Lens is an open-source interactive tool that shows, step by step, how a language model processes and generates text — tokenization, attention, and next-word prediction — running entirely in the browser, with no server behind it.",
    aboutStackHeading: "Tech stack",
    lessons: {
      tokens: {
        simple: "A language model doesn't read whole words like we do — it breaks text into small pieces called tokens before processing anything. Sometimes a token is a whole word, sometimes just a fragment of one, depending on how common it was during training. It's as if the model saw everything through a fixed vocabulary of tens of thousands of \"Lego pieces\", and any text has to be rebuilt from them.",
        technical: "Tokenization uses Byte-Pair Encoding (BPE): a fixed subword vocabulary (here, ~50k units) built from the statistical frequency of character pairs in the training corpus. Frequent words become a single token; rare or compound words get fragmented into multiple subwords. This is the first pipeline stage — the model never sees raw characters, only these token IDs.",
      },
      attention: {
        simple: "Attention is the mechanism that lets the model \"look back\" at the sentence while processing each word, deciding which others are relevant. It's similar to what you do reading \"she said he would never come back\" — to know who \"he\" is, your brain automatically searches earlier for a matching name. The model does something similar, many times, in parallel, at every layer.",
        technical: "Self-attention computes, for each token, a weighted distribution over every other token in the sequence, via Query, Key, and Value projections followed by a scaled softmax. Each layer has multiple attention \"heads\", each potentially capturing a different kind of relationship (syntactic, coreferential, positional). Stacking layers builds increasingly contextual representations.",
      },
      next: {
        simple: "After processing the whole sentence, the model doesn't \"pick\" the next word directly — it computes a probability for every possible word in the vocabulary (tens of thousands of them!) and usually picks one of the most likely ones. It's as if it were betting on many words at once, with different amounts on each.",
        technical: "The final layer produces a logits vector the size of the vocabulary, converted into probabilities via softmax. Next-token selection can be deterministic (greedy: always the highest probability) or stochastic (sampling with temperature, top-k, or top-p) — directly affecting the creativity vs. predictability of the generated text.",
      },
      write: {
        simple: "Here you're using the exact same mechanism from the previous section, just repeated over and over — each word you accept becomes part of the context for computing the next one. It's basically how your phone's autocomplete works, except now you're seeing inside it.",
        technical: "Each click triggers a new forward pass with the accumulated text as input, recomputing the full distribution from scratch (no incremental cache in this implementation). This is the same autoregressive generation used in production, just exposed step-by-step instead of run automatically until a stop token.",
      },
    },
    glossaryBasic: {
      token: "A chunk of text — can be a whole word or a fragment of one.",
      attention: "The mechanism that lets the model \"look\" at other words in the sentence while processing one word.",
      layer: "A processing block (attention + a small neural network) that the model repeats several times in sequence, refining its understanding of the sentence at each pass.",
      heads: "One of several \"attentions\" running in parallel within the same layer — each head can learn to focus on a different kind of relationship between words.",
      probability: "How \"confident\" the model is that this is the right next word.",
    },
    glossaryTechnical: {
      token: "A chunk of text — can be a whole word or a fragment of one.",
      attention: "The mechanism that lets the model \"look\" at other words in the sentence while processing one word.",
      layer: "A processing block (attention + a small neural network) that the model repeats several times in sequence, refining its understanding of the sentence at each pass.",
      heads: "One of several \"attentions\" running in parallel within the same layer — each head can learn to focus on a different kind of relationship between words.",
      probability: "How \"confident\" the model is that this is the right next word.",
      "Tokenization": "The process of breaking text into tokens before any processing happens.",
      "Byte-Pair Encoding": "An algorithm that builds the vocabulary by repeatedly merging the most frequent character pairs in the corpus.",
      "subwords": "Pieces of a word smaller than the whole word, used when the full word isn't in the tokenizer's vocabulary.",
      "training corpus": "The collection of text used to train the model.",
      "Self-attention": "The mechanism that makes each token compare itself to every other token in the sequence to decide where to focus.",
      "Query, Key, and Value": "Three vectors computed for each token, used to decide how much attention it should pay to the others.",
      "softmax": "A function that turns a list of numbers into probabilities that add up to 100%.",
      "logits": "The raw numbers the model computes before they become probabilities.",
      "deterministic": "A generation strategy where the result is always the same for the same input.",
      "greedy": "A strategy that always picks the most probable word, with no randomness.",
      "stochastic": "A generation strategy that involves some random draw among the most likely options.",
      "temperature": "A parameter that controls how random the next-word choice is.",
      "top-k": "A strategy that samples the next word only from the k most likely options.",
      "top-p": "A strategy that samples the next word from the smallest set of options whose probabilities add up to p%.",
      "forward pass": "One full pass of the input data through the model, start to end, until it produces an output.",
      "incremental cache": "A technique for storing already-computed results so they don't need to be recalculated for every new token.",
      "autoregressive generation": "Generating text one word at a time, always using the words generated so far as part of the input.",
      "stop token": "A special token that signals to the model that it should stop generating text.",
    },
  },
};

const t = UI[LANG];

// ============================================================================
// EXPLICAÇÕES DINÂMICAS — geradas a partir de números reais da análise,
// não de texto genérico fixo. Cada função recebe as estatísticas calculadas
// e devolve uma frase em PT ou EN, no modo Simples ou Técnico.
// ============================================================================
let explanationMode = "simple"; // "simple" | "technical"

const EXPLAIN = {
  tokens(lang, mode, { text, tokenCount, wordCount }) {
    const split = tokenCount > wordCount;
    if (lang === "pt") {
      if (mode === "simple") {
        return split
          ? `Sua frase tem ${wordCount} palavras, mas virou ${tokenCount} tokens — o modelo quebrou pelo menos uma palavra em pedaços menores, porque não a "decorou" inteira durante o treinamento.`
          : `Sua frase virou ${tokenCount} tokens, um para cada palavra — nenhuma precisou ser quebrada em pedaços.`;
      }
      return split
        ? `Contagem de tokens (${tokenCount}) maior que a de palavras (${wordCount}): o tokenizador BPE encontrou ao menos um termo fora do vocabulário de subpalavras mais frequentes, exigindo fragmentação.`
        : `Contagem de tokens igual à de palavras (${tokenCount}): todas as palavras corresponderam a uma unidade única do vocabulário do tokenizador.`;
    }
    if (mode === "simple") {
      return split
        ? `Your sentence has ${wordCount} words, but became ${tokenCount} tokens — the model split at least one word into smaller pieces, since it wasn't "memorized" whole during training.`
        : `Your sentence became ${tokenCount} tokens, one per word — none needed to be split.`;
    }
    return split
      ? `Token count (${tokenCount}) exceeds word count (${wordCount}): the BPE tokenizer found at least one term outside the most frequent subword vocabulary, requiring fragmentation.`
      : `Token count matches word count (${tokenCount}): every word mapped to a single tokenizer vocabulary unit.`;
  },

  attention(lang, mode, { focusedToken, focusedScore, spreadToken, mostAttendedToken, secondaryToken, secondaryScore }) {
    const pct = Math.round(focusedScore * 100);
    const secondaryPct = Math.round(secondaryScore * 100);
    const concentrated = focusedScore > 0 && secondaryScore / focusedScore < 0.15;
    if (lang === "pt") {
      if (mode === "simple") {
        const breakdownNote = concentrated
          ? ` No quadro abaixo dá pra ver isso de perto: quase toda a atenção de "${focusedToken}" vai para um único token — o segundo colocado, "${secondaryToken}", fica bem atrás, com só ${secondaryPct}%.`
          : ` No quadro abaixo dá pra ver isso de perto: a atenção de "${focusedToken}" se divide entre poucos tokens — o segundo colocado, "${secondaryToken}", ainda fica com ${secondaryPct}%, disputando espaço com o primeiro.`;
        return `Nessa combinação de camada/cabeça, o token "${focusedToken}" está bem concentrado — cerca de ${pct}% da atenção dele vai para um único outro token. Já "${spreadToken}" está mais "distraído", espalhando atenção entre vários. No geral, "${mostAttendedToken}" é quem mais recebe atenção dos outros — parece ser o centro das atenções dessa frase.${breakdownNote}`;
      }
      const breakdownNote = ` A quebra abaixo detalha essa linha: a segunda maior atenção de "${focusedToken}" vai para "${secondaryToken}" (${secondaryPct}%), ${concentrated ? "uma queda acentuada em relação ao primeiro colocado — atenção fortemente concentrada" : "relativamente próxima do primeiro colocado — atenção mais distribuída"}.`;
      return `Pico de atenção máximo observado: ${pct}% (token de origem "${focusedToken}"). O token "${spreadToken}" apresenta a distribuição mais próxima de uniforme entre as posições. Somando a atenção recebida por coluna, "${mostAttendedToken}" concentra o maior total — indicando maior relevância posicional nessa cabeça específica.${breakdownNote}`;
    }
    if (mode === "simple") {
      const breakdownNote = concentrated
        ? ` The breakdown below makes this concrete: almost all of "${focusedToken}"'s attention goes to a single token — the runner-up, "${secondaryToken}", trails far behind at just ${secondaryPct}%.`
        : ` The breakdown below makes this concrete: "${focusedToken}"'s attention splits across a few tokens — the runner-up, "${secondaryToken}", still holds ${secondaryPct}%, competing with the top pick.`;
      return `In this layer/head combination, the token "${focusedToken}" is quite focused — about ${pct}% of its attention goes to a single other token. Meanwhile "${spreadToken}" is more "distracted", spreading attention across several. Overall, "${mostAttendedToken}" receives the most attention from others — it seems to be the center of attention in this sentence.${breakdownNote}`;
    }
    const breakdownNote = ` The breakdown below details this row: the second-highest weight for "${focusedToken}" goes to "${secondaryToken}" (${secondaryPct}%), ${concentrated ? "a sharp drop from the top pick — indicating strongly concentrated attention" : "relatively close to the top pick — indicating more distributed attention"}.`;
    return `Maximum observed attention peak: ${pct}% (source token "${focusedToken}"). Token "${spreadToken}" shows the distribution closest to uniform across positions. Summing received attention by column, "${mostAttendedToken}" concentrates the highest total — indicating greater positional relevance for this specific head.${breakdownNote}`;
  },

  nextToken(lang, mode, { topToken, topProb, entropyRatio }) {
    const pct = Math.round(topProb * 100);
    const confident = entropyRatio < 0.35;
    const uncertain = entropyRatio > 0.65;
    if (lang === "pt") {
      if (mode === "simple") {
        if (confident) return `O modelo está bem confiante aqui: "${topToken}" tem ${pct}% de chance, bem à frente das outras opções — provavelmente essa combinação de palavras apareceu muito no treinamento dele.`;
        if (uncertain) return `O modelo está bem em dúvida aqui: "${topToken}" só tem ${pct}% de chance, disputando de perto com várias outras palavras — várias continuações fariam sentido.`;
        return `O modelo tem uma preferência moderada: "${topToken}" lidera com ${pct}% de chance, mas sem dominar totalmente as alternativas.`;
      }
      const entropyPct = Math.round(entropyRatio * 100);
      return `Token mais provável: "${topToken}" (p=${(topProb).toFixed(3)}). Entropia da distribuição: ${entropyPct}% da entropia máxima possível para este vocabulário — ${confident ? "baixa, indicando alta confiança" : uncertain ? "alta, indicando grande incerteza" : "moderada"}.`;
    }
    if (mode === "simple") {
      if (confident) return `The model is quite confident here: "${topToken}" has a ${pct}% chance, well ahead of the alternatives — this word combination likely appeared often during training.`;
      if (uncertain) return `The model is quite unsure here: "${topToken}" only has a ${pct}% chance, closely competing with several other words — many continuations would make sense.`;
      return `The model has a moderate preference: "${topToken}" leads with ${pct}% chance, but doesn't fully dominate the alternatives.`;
    }
    const entropyPct = Math.round(entropyRatio * 100);
    return `Most probable token: "${topToken}" (p=${(topProb).toFixed(3)}). Distribution entropy: ${entropyPct}% of the maximum possible entropy for this vocabulary — ${confident ? "low, indicating high confidence" : uncertain ? "high, indicating great uncertainty" : "moderate"}.`;
  },
};

// ============================================================================
// APLICAR TEXTOS ESTÁTICOS NA PÁGINA
// ============================================================================
function applyStaticText() {
  document.getElementById("loading-title").textContent = t.loadingTitle;
  document.getElementById("loading-note-text").textContent = t.loadingNote;
  document.getElementById("input-eyebrow").textContent = t.inputEyebrow;
  document.getElementById("analyze-btn").textContent = t.analyzeBtn;
  document.getElementById("prompt-input").value = t.defaultPrompt;
  document.getElementById("write-input").value = t.defaultWrite;
  document.querySelector('.pipeline .stage[data-stage="text"]').textContent = t.pipelineText;
  document.querySelector('.pipeline .stage[data-stage="token"]').textContent = t.pipelineToken;
  document.querySelector('.pipeline .stage[data-stage="embedding"]').textContent = t.pipelineEmbedding;
  document.querySelector('.pipeline .stage[data-stage="attention"]').textContent = t.pipelineAttention;
  document.querySelector('.pipeline .stage[data-stage="prediction"]').textContent = t.pipelinePrediction;
  document.getElementById("tokens-title").textContent = t.tokensTitle;
  document.getElementById("tokens-hint").textContent = t.tokensHint;
  document.getElementById("attention-title").textContent = t.attentionTitle;
  document.getElementById("attention-hint").textContent = t.attentionHint;
  document.getElementById("attention-breakdown-title").textContent = t.attentionBreakdownTitle;
  document.getElementById("layer-label").textContent = t.layerLabel;
  document.getElementById("head-label").textContent = t.headLabel;
  document.getElementById("next-title").textContent = t.nextTitle;
  document.getElementById("next-hint").textContent = t.nextHint;
  document.getElementById("write-title").textContent = t.writeTitle;
  document.getElementById("write-hint").textContent = t.writeHint;
  document.getElementById("write-start-btn").textContent = t.writeStart;
  document.getElementById("write-reset-btn").textContent = t.writeReset;
  document.getElementById("footer-text").textContent = t.footer;
  document.getElementById("sobre-link").textContent = LANG === "pt" ? "Sobre" : "About";
  document.getElementById("sobre-link").href = t.sobreLink;

  document.getElementById("about-role").textContent = t.aboutRole;
  document.getElementById("about-project-heading").textContent = t.aboutProjectHeading;
  document.getElementById("about-project-text").textContent = t.aboutProjectText;
  document.getElementById("about-stack-heading").textContent = t.aboutStackHeading;

  const examplesContainer = document.getElementById("example-chips");
  examplesContainer.innerHTML = "";
  t.examples.forEach((ex) => {
    const chip = document.createElement("button");
    chip.className = "example-chip";
    chip.textContent = ex;
    chip.onclick = () => {
      document.getElementById("prompt-input").value = ex;
      analyze();
    };
    examplesContainer.appendChild(chip);
  });

  document.getElementById("lang-pt").classList.toggle("active", LANG === "pt");
  document.getElementById("lang-en").classList.toggle("active", LANG === "en");
  document.getElementById("lang-pt").href = "app.html?lang=pt";
  document.getElementById("lang-en").href = "app.html?lang=en";

  document.getElementById("mode-label-simple").textContent = LANG === "pt" ? "Iniciante" : "Beginner";
  document.getElementById("mode-label-technical").textContent = LANG === "pt" ? "Técnico" : "Technical";
  // Estado inicial do toggle: modo "Iniciante" ativo por padrão
  document.getElementById("mode-label-simple").classList.add("active");

  applyLessonTexts();
}

// Escreve `text` dentro de `el`, envolvendo termos do glossário em spans
// ".term" com tooltip (".tip") — usados pra explicar jargão sem poluir o texto.
function renderLessonText(el, text, glossary) {
  const terms = Object.keys(glossary || {}).sort((a, b) => b.length - a.length);
  if (!terms.length) {
    el.textContent = text;
    return;
  }
  const escaped = terms.map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const regex = new RegExp(`\\b(${escaped.join("|")})\\b`, "gi");

  el.innerHTML = "";
  let lastIndex = 0;
  let match;
  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      el.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
    }
    const matchedTerm = terms.find((term) => term.toLowerCase() === match[0].toLowerCase());
    const span = document.createElement("span");
    span.className = "term";
    span.appendChild(document.createTextNode(match[0]));
    const tip = document.createElement("span");
    tip.className = "tip";
    tip.textContent = glossary[matchedTerm];
    span.appendChild(tip);
    el.appendChild(span);
    lastIndex = regex.lastIndex;
  }
  if (lastIndex < text.length) {
    el.appendChild(document.createTextNode(text.slice(lastIndex)));
  }
}

function applyLessonTexts() {
  // Tooltips de termos: pouco intrusivas no modo Iniciante (só o glossário básico),
  // mais completas no modo Técnico (glossário estendido com o jargão usado nesse texto).
  const glossary = explanationMode === "technical" ? t.glossaryTechnical : t.glossaryBasic;
  renderLessonText(document.getElementById("tokens-lesson"), t.lessons.tokens[explanationMode], glossary);
  renderLessonText(document.getElementById("attention-lesson"), t.lessons.attention[explanationMode], glossary);
  renderLessonText(document.getElementById("next-lesson"), t.lessons.next[explanationMode], glossary);
  renderLessonText(document.getElementById("write-lesson"), t.lessons.write[explanationMode], glossary);
}

document.getElementById("mode-switch").addEventListener("click", () => {
  explanationMode = explanationMode === "simple" ? "technical" : "simple";
  document.getElementById("mode-switch").classList.toggle("on", explanationMode === "technical");
  document.getElementById("mode-label-simple").classList.toggle("active", explanationMode === "simple");
  document.getElementById("mode-label-technical").classList.toggle("active", explanationMode === "technical");
  applyLessonTexts();
  // Reaplica a análise atual (se já tiver rodado) para atualizar as explicações dinâmicas também
  if (lastAnalysis) renderInsights(lastAnalysis);
});

// ============================================================================
// PAINEL "SOBRE" — abre por cima da página em vez de navegar pra sobre.html,
// assim o modelo já carregado na memória não é perdido (nem precisa recarregar).
// O link mantém o href de verdade como fallback (abrir em nova aba, Ctrl/Cmd+clique).
// ============================================================================
function openAbout() {
  document.getElementById("about-overlay").classList.add("open");
}
function closeAbout() {
  document.getElementById("about-overlay").classList.remove("open");
}
document.getElementById("sobre-link").addEventListener("click", (e) => {
  if (e.ctrlKey || e.metaKey || e.shiftKey || e.button === 1) return; // deixa abrir em nova aba normalmente
  e.preventDefault();
  openAbout();
});
document.getElementById("about-close").addEventListener("click", closeAbout);
document.getElementById("about-overlay").addEventListener("click", (e) => {
  if (e.target.id === "about-overlay") closeAbout();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeAbout();
});

// ============================================================================
// CARREGAMENTO DO MODELO (com barra de progresso real)
// ============================================================================
let tokenizer = null;
let model = null;
let modelReady = false;

function updateProgress(pct, label) {
  document.getElementById("progress-fill").style.width = `${pct}%`;
  document.getElementById("loading-pct").textContent = `${Math.round(pct)}%${label ? " · " + label : ""}`;
}

async function loadModel() {
  const modelId = MODEL_IDS[LANG];

  const progressCallback = (info) => {
    if (info.status === "progress" && info.total) {
      const pct = (info.loaded / info.total) * 100;
      updateProgress(pct, info.file);
    }
  };

  tokenizer = await AutoTokenizer.from_pretrained(modelId, { progress_callback: progressCallback });
  model = await AutoModelForCausalLM.from_pretrained(modelId, {
    dtype: "fp32",
    progress_callback: progressCallback,
  });

  modelReady = true;
  updateProgress(100, null);
  document.getElementById("loading-overlay").classList.add("hidden");
}

// ============================================================================
// ANÁLISE DO MODELO — tokenização, atenção, próximo token
// ============================================================================
let lastAnalysis = null;

async function runAttention(inputs) {
  const outputs = await model({ ...inputs });
  const expectedLayers = model.config.n_layer ?? 12;
  const attentions = [];
  for (let i = 0; i < expectedLayers; i++) {
    const layerOutput = outputs[`attentions.${i}`];
    if (!layerOutput) break;
    attentions.push(layerOutput);
  }
  return attentions;
}

async function computeNextTokenDistribution(text, topK = 10) {
  const inputs = await tokenizer(text);
  const { logits } = await model({ ...inputs });

  const [, seqLen, vocabSize] = logits.dims;
  const lastTokenOffset = (seqLen - 1) * vocabSize;
  const lastLogits = logits.data.slice(lastTokenOffset, lastTokenOffset + vocabSize);

  let maxLogit = -Infinity;
  for (let i = 0; i < lastLogits.length; i++) if (lastLogits[i] > maxLogit) maxLogit = lastLogits[i];
  let sumExp = 0;
  const expVals = new Float32Array(lastLogits.length);
  for (let i = 0; i < lastLogits.length; i++) {
    const e = Math.exp(lastLogits[i] - maxLogit);
    expVals[i] = e;
    sumExp += e;
  }

  const probs = new Float32Array(expVals.length);
  let entropy = 0;
  for (let i = 0; i < expVals.length; i++) {
    const p = expVals[i] / sumExp;
    probs[i] = p;
    if (p > 0) entropy -= p * Math.log2(p);
  }
  const maxEntropy = Math.log2(vocabSize);
  const entropyRatio = entropy / maxEntropy;

  const indexed = [];
  for (let i = 0; i < probs.length; i++) indexed.push({ id: i, probability: probs[i] });
  indexed.sort((a, b) => b.probability - a.probability);
  const top = indexed.slice(0, topK).map((item) => ({
    token: tokenizer.decode([item.id]),
    id: item.id,
    probability: item.probability,
  }));

  return { top, entropyRatio };
}

// ---------- Renderização: Tokens ----------
function renderTokens(tokenIds) {
  const container = document.getElementById("tokens-view");
  container.innerHTML = "";
  tokenIds.forEach((id) => {
    const chip = document.createElement("span");
    chip.className = "token-chip";
    chip.textContent = tokenizer.decode([id]);
    chip.title = `id: ${id}`;
    container.appendChild(chip);
  });
}

// ---------- Renderização: Atenção ----------
let currentAttentions = null;
let currentTokenTexts = null;

function renderAttentionControls() {
  const layerSelect = document.getElementById("layer-select");
  const headSelect = document.getElementById("head-select");
  layerSelect.innerHTML = "";
  headSelect.innerHTML = "";

  const numLayers = currentAttentions.length;
  const numHeads = currentAttentions[0].dims[1];

  for (let i = 0; i < numLayers; i++) {
    const opt = document.createElement("option");
    opt.value = i; opt.textContent = i;
    layerSelect.appendChild(opt);
  }
  for (let i = 0; i < numHeads; i++) {
    const opt = document.createElement("option");
    opt.value = i; opt.textContent = i;
    headSelect.appendChild(opt);
  }
  // Começa na primeira camada — pra frase padrão do site, é onde o padrão de atenção
  // fica mais variado/legível pra quem tá vendo pela primeira vez.
  layerSelect.value = 0;

  layerSelect.onchange = renderAttentionGrid;
  headSelect.onchange = renderAttentionGrid;
}

function tensorToMatrix(tensor, headIdx) {
  const [, numHeads, seqLen] = tensor.dims;
  const data = tensor.data;
  const matrix = [];
  for (let i = 0; i < seqLen; i++) {
    const row = [];
    for (let j = 0; j < seqLen; j++) {
      const flatIdx = headIdx * seqLen * seqLen + i * seqLen + j;
      row.push(data[flatIdx]);
    }
    matrix.push(row);
  }
  return matrix;
}

// "Destrincha" a matriz de atenção: para cada token, mostra as top-3 posições
// pra onde ele mais olha, como barras — mais fácil de ler que a linha inteira do heatmap.
function renderAttentionBreakdown(matrix, tokens) {
  const container = document.getElementById("attention-breakdown");
  container.innerHTML = "";

  matrix.forEach((row, i) => {
    const indexed = row.map((weight, j) => ({ weight, token: tokens[j] }));
    indexed.sort((a, b) => b.weight - a.weight);
    const top = indexed.slice(0, 3).filter((item) => item.weight > 0.01);
    if (!top.length) return;
    const maxWeight = top[0].weight;

    const rowEl = document.createElement("div");
    rowEl.className = "breakdown-row";

    const source = document.createElement("span");
    source.className = "breakdown-source";
    source.textContent = tokens[i];
    rowEl.appendChild(source);

    const targets = document.createElement("div");
    targets.className = "breakdown-targets";
    top.forEach((item, rank) => {
      const targetRow = document.createElement("div");
      targetRow.className = rank === 0 ? "breakdown-target breakdown-target-top" : "breakdown-target";
      targetRow.title = `${(item.weight * 100).toFixed(1)}%`;

      const label = document.createElement("span");
      label.className = "breakdown-target-label";
      label.textContent = item.token;

      const barBg = document.createElement("div");
      barBg.className = "breakdown-bar-bg";
      const barFill = document.createElement("div");
      barFill.className = "breakdown-bar-fill";
      barFill.style.width = `${(item.weight / maxWeight) * 100}%`;
      barBg.appendChild(barFill);

      const pct = document.createElement("span");
      pct.className = "breakdown-target-pct";
      pct.textContent = `${Math.round(item.weight * 100)}%`;

      targetRow.append(label, barBg, pct);
      targets.appendChild(targetRow);
    });
    rowEl.appendChild(targets);
    container.appendChild(rowEl);
  });
}

function renderAttentionGrid() {
  const layer = parseInt(document.getElementById("layer-select").value || 0);
  const head = parseInt(document.getElementById("head-select").value || 0);
  const matrix = tensorToMatrix(currentAttentions[layer], head);
  const tokens = currentTokenTexts;

  const table = document.createElement("table");
  table.className = "attention-grid";

  const headerRow = document.createElement("tr");
  headerRow.appendChild(document.createElement("th"));
  tokens.forEach((tok) => {
    const th = document.createElement("th");
    th.textContent = tok;
    headerRow.appendChild(th);
  });
  table.appendChild(headerRow);

  // Estatísticas pra explicação dinâmica
  let focusedRow = 0, focusedScore = -1;
  let spreadRow = 0, spreadScore = Infinity;
  const colSums = new Array(tokens.length).fill(0);

  matrix.forEach((row, i) => {
    const tr = document.createElement("tr");
    const rowLabel = document.createElement("th");
    rowLabel.textContent = tokens[i];
    tr.appendChild(rowLabel);

    const rowMax = Math.max(...row);
    if (rowMax > focusedScore) { focusedScore = rowMax; focusedRow = i; }
    if (rowMax < spreadScore) { spreadScore = rowMax; spreadRow = i; }

    row.forEach((weight, j) => {
      colSums[j] += weight;
      const td = document.createElement("td");
      td.style.background = `rgba(79, 209, 197, ${weight.toFixed(2)})`;
      if (weight === rowMax && weight > 0.3) td.classList.add("hot-cell");
      td.title = weight.toFixed(3);
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });

  const container = document.getElementById("attention-view");
  container.innerHTML = "";
  container.appendChild(table);

  renderAttentionBreakdown(matrix, tokens);

  const mostAttendedIdx = colSums.indexOf(Math.max(...colSums));

  // Segundo colocado na linha mais concentrada — liga o insight ao que o
  // destrinchamento abaixo mostra (concentrado num vizinho vs. dividido entre poucos).
  const focusedRowSorted = matrix[focusedRow]
    .map((weight, j) => ({ weight, token: tokens[j] }))
    .sort((a, b) => b.weight - a.weight);
  const secondary = focusedRowSorted[1] || { weight: 0, token: tokens[focusedRow] };

  const text = EXPLAIN.attention(LANG, explanationMode, {
    focusedToken: tokens[focusedRow],
    focusedScore,
    spreadToken: tokens[spreadRow],
    mostAttendedToken: tokens[mostAttendedIdx],
    secondaryToken: secondary.token,
    secondaryScore: secondary.weight,
  });
  document.getElementById("attention-insight").textContent = text;
}

// ---------- Renderização: Próximo token ----------
function renderNextTokenDistribution(result, targetTextId = "generated-text") {
  const { top, entropyRatio } = result;
  const container = document.getElementById("next-token-view");
  container.innerHTML = "";

  const maxProb = Math.max(...top.map((d) => d.probability));

  top.forEach((item) => {
    const row = document.createElement("div");
    row.className = "prob-row";
    const label = document.createElement("span");
    label.textContent = item.token;
    const barBg = document.createElement("div");
    barBg.className = "prob-bar-bg";
    const barFill = document.createElement("div");
    barFill.className = "prob-bar-fill";
    barFill.style.width = `${(item.probability / maxProb) * 100}%`;
    barBg.appendChild(barFill);
    const pct = document.createElement("span");
    pct.textContent = `${(item.probability * 100).toFixed(1)}%`;
    row.append(label, barBg, pct);

    row.onclick = async () => {
      const input = document.getElementById("prompt-input");
      input.value += item.token;
      await analyze();
    };
    container.appendChild(row);
  });

  const text = EXPLAIN.nextToken(LANG, explanationMode, {
    topToken: top[0].token,
    topProb: top[0].probability,
    entropyRatio,
  });
  document.getElementById("next-insight").textContent = text;
}

// ============================================================================
// FLUXO PRINCIPAL — analisar frase completa
// ============================================================================
function renderInsights(analysis) {
  // Reaplica todas as explicações (usado ao trocar Simples/Técnico sem reanalisar)
  document.getElementById("tokens-insight").textContent = EXPLAIN.tokens(LANG, explanationMode, analysis.tokenStats);
  renderAttentionGrid();
  renderNextTokenDistribution(analysis.nextTokenResult);
}

async function analyze() {
  const text = document.getElementById("prompt-input").value.trim();
  if (!text || !modelReady) return;

  try {
    const inputs = await tokenizer(text);
    const tokenIds = Array.from(inputs.input_ids.data).map(Number);
    renderTokens(tokenIds);
    currentTokenTexts = tokenIds.map((id) => tokenizer.decode([id]));

    currentAttentions = await runAttention(inputs);
    renderAttentionControls();

    const nextTokenResult = await computeNextTokenDistribution(text);

    const wordCount = text.split(/\s+/).filter(Boolean).length;
    const tokenStats = { text, tokenCount: tokenIds.length, wordCount };

    lastAnalysis = { tokenStats, nextTokenResult };

    document.getElementById("tokens-insight").textContent = EXPLAIN.tokens(LANG, explanationMode, tokenStats);
    renderAttentionGrid();
    renderNextTokenDistribution(nextTokenResult);

    ["section-tokens", "section-attention", "section-next", "section-write"].forEach((id) => {
      document.getElementById(id).style.display = "block";
    });
  } catch (err) {
    alert(t.errorMsg);
    console.error(err);
  }
}

// ============================================================================
// SEÇÃO 4 — Escreva junto com o modelo
// ============================================================================
let writeText = "";
let writing = false;

async function updateWriteSuggestions() {
  const result = await computeNextTokenDistribution(writeText, 5);
  const container = document.getElementById("write-suggestions");
  container.innerHTML = "";
  result.top.forEach((item) => {
    const chip = document.createElement("button");
    chip.className = "suggestion-chip";
    chip.textContent = item.token.trim() || "⏎";
    chip.onclick = async () => {
      writeText += item.token;
      document.getElementById("write-output").textContent = writeText;
      await updateWriteSuggestions();
    };
    container.appendChild(chip);
  });
}

document.getElementById("write-start-btn").addEventListener("click", async () => {
  writeText = document.getElementById("write-input").value.trim();
  if (!writeText || !modelReady) return;
  writing = true;
  document.getElementById("write-output").style.display = "block";
  document.getElementById("write-output").textContent = writeText;
  document.getElementById("write-reset-btn").style.display = "inline-flex";
  document.getElementById("write-input").style.display = "none";
  document.getElementById("write-start-btn").style.display = "none";
  await updateWriteSuggestions();
});

document.getElementById("write-reset-btn").addEventListener("click", () => {
  writing = false;
  writeText = "";
  document.getElementById("write-output").style.display = "none";
  document.getElementById("write-suggestions").innerHTML = "";
  document.getElementById("write-reset-btn").style.display = "none";
  document.getElementById("write-input").style.display = "block";
  document.getElementById("write-start-btn").style.display = "inline-flex";
});

// ============================================================================
// INICIALIZAÇÃO
// ============================================================================
document.addEventListener("DOMContentLoaded", async () => {
  applyStaticText();
  document.getElementById("analyze-btn").disabled = true;
  document.getElementById("analyze-btn").addEventListener("click", analyze);
  await loadModel();
  document.getElementById("analyze-btn").disabled = false;
});
