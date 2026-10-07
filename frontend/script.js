// ============================================================================
// CONFIGURAÇÃO
// ============================================================================
// V2: hidden states por camada (usados pelo "elevador de camadas") + atenção
// + quantização int8. PT usa GPorTuguese-2 (fine-tunado); EN usa o GPT-2
// small padrão da OpenAI (modelo base, não fine-tunado -- por isso as
// distribuições de probabilidade dele tendem a ser mais "achatadas").
const MODEL_IDS = {
  pt: "GusHil/gpt2-small-portuguese-onnx-v2",
  en: "GusHil/gpt2-english-onnx-v2",
};

import {
  AutoTokenizer,
  AutoModelForCausalLM,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0";
import { initSimulator } from "./simulator.js";

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
    defaultPrompt: "Ela disse que ele nunca voltaria",
    defaultWrite: "Era uma vez",
    examples: [
      "Ela disse que ele nunca voltaria",
      "Ela ficou impressionadíssima",
      "Ela mora no Rio de",
    ],
    tokensTitle: "Tokenização",
    tokensHint: "Cada bloco é um token — a unidade mínima que o modelo enxerga.",
    attentionTitle: "Atenção",
    attentionHint: "Escolha uma camada e uma cabeça para ver para onde cada token \"olha\".",
    attentionBreakdownTitle: "Para onde cada token olha (top 3)",
    layerLabel: "Camada",
    headLabel: "Cabeça",
    headCompareToggle: "Comparar cabeças",
    headCompareBack: "← Voltar",
    headComparisonCaption: "A mesma camada, três cabeças diferentes — cada uma pode aprender a notar um tipo diferente de relação entre as palavras.",
    headLabelShort: "Cabeça",
    hiddenTitle: "Camadas ocultas",
    hiddenHint: "Escolha uma camada para ver como o modelo reposiciona cada token internamente.",
    arcsCaption: "Cada seta aponta para o token mais atendido; a espessura da linha indica a força da atenção. Um anel sobre o próprio token significa que ele atende principalmente a si mesmo.",
    elevatorCaption: "Só a distância entre os pontos importa aqui — a posição exata e os eixos não têm significado próprio.",
    elevatorLegendNear: "mais parecido",
    elevatorLegendFar: "mais diferente",
    hiddenLayerLabel: "Camada",
    hiddenEmbeddingLabel: "Embedding (entrada)",
    hiddenLayerName: "Camada",
    nextTitle: "Próxima palavra",
    nextHint: "A distribuição de probabilidade que o modelo calculou para a próxima palavra. Clique numa barra para continuar a frase com ela.",
    writeTitle: "Escreva junto com o modelo",
    writeHint: "Continue uma frase e deixe o modelo sugerir a próxima palavra, uma de cada vez.",
    writeStart: "Começar",
    writeReset: "Recomeçar",
    compareTitle: "Comparar frases",
    compareHint: "Veja como pequenas mudanças na frase alteram o que o modelo \"enxerga\".",
    compareLabelA: "Frase A",
    compareLabelB: "Frase B",
    compareBtn: "Comparar",
    compareDefaultA: "Maria ajudou João",
    compareDefaultB: "João ajudou Maria",
    compareAttentionInsight: (tok) => `O token que mais recebe atenção é "${tok}".`,
    simTitle: "Construa um jogo com a IA",
    simHint: "Converse com o modelo e veja ele \"programar\" um jogo da cobrinha, peça por peça.",
    footer: "Projeto de código aberto — TCC Pós-Graduação IA Generativa & LLMs (PUC-Rio / ICA).",
    sobreLink: "sobre.html?lang=pt",
    errorMsg: "Algo deu errado ao carregar ou rodar o modelo. Abra o Console (F12) para detalhes.",
    aboutRole: "TCC — Pós-Graduação em Inteligência Artificial Generativa & LLMs, PUC-Rio / Laboratório ICA",
    aboutProjectHeading: "Sobre o projeto",
    aboutProjectText: "O LLM Lens é uma ferramenta interativa de código aberto que mostra, passo a passo, como um modelo de linguagem processa e gera texto — tokenização, atenção e previsão de próxima palavra — rodando inteiramente no navegador, sem nenhum servidor por trás.",
    aboutStackHeading: "Stack técnica",
    modelType: {
      slm: {
        label: "Modelo pequeno (este site)",
        stages: ["Texto", "Token", "Embedding", "Atenção (12×12)", "Previsão"],
        text: "É exatamente isso que roda aqui, dentro do seu navegador: um modelo pequeno (SLM, \"Small Language Model\"), com cerca de 124 milhões de parâmetros, que só entende texto. A frase que você digita é quebrada em tokens, cada token vira uma lista de números (embedding), essa informação passa por 12 camadas — cada uma com 12 \"cabeças\" de atenção trabalhando em paralelo — e no final o modelo calcula a probabilidade de cada palavra possível ser a próxima. É leve e rápido o suficiente pra rodar inteiro no seu computador ou celular, mas tem bem menos conhecimento e capacidade de raciocínio do que os modelos grandes usados em assistentes online.",
      },
      llm: {
        label: "LLM grande",
        stages: ["Texto", "Token", "Embedding", "Dezenas de camadas", "Memória / Busca", "Ferramentas", "Resposta"],
        text: "Os assistentes de IA que você usa no dia a dia (grandes chatbots online) seguem a mesma ideia de base — tokenizar, transformar em números, passar por camadas de atenção —, mas numa escala muito maior: de bilhões a centenas de bilhões de parâmetros, com dezenas de camadas empilhadas. Eles também costumam vir com peças extras que o modelo deste site não tem: memória de conversa mais longa, sistemas de busca pra trazer informação atualizada da internet, e a capacidade de \"usar ferramentas\" (como calculadora ou execução de código) antes de responder. Por isso, precisam rodar em servidores potentes, não no seu navegador.",
      },
      multimodal: {
        label: "Multimodal",
        stages: ["Texto / Imagem / Áudio", "Codificadores", "Fusão", "Transformer", "Resposta"],
        text: "Alguns modelos vão além do texto: também enxergam imagens, ouvem áudio e até processam vídeo. Para isso, cada tipo de informação passa primeiro por um \"codificador\" especializado (um para texto, um para imagem, um para áudio...) que transforma tudo no mesmo tipo de números. Só depois dessa etapa de \"fusão\" a informação combinada entra em camadas de atenção parecidas com as que você está vendo aqui — só que preparadas pra lidar com vários sentidos ao mesmo tempo, e capazes de responder também em mais de um formato.",
      },
    },
    lessons: {
      tokens: {
        simple: "Um modelo de linguagem não lê palavras inteiras como nós — ele quebra o texto em pedacinhos chamados tokens antes de processar qualquer coisa. Às vezes um token é uma palavra inteira, às vezes é só um pedaço dela, dependendo do quão comum ela foi durante o treinamento. É como se o modelo enxergasse tudo através de um vocabulário fixo de dezenas de milhares de \"peças de Lego\", e qualquer texto precisa ser remontado a partir delas.",
        technical: "A tokenização usa Byte-Pair Encoding (BPE): um vocabulário fixo de subpalavras (aqui, ~50 mil unidades) construído a partir da frequência estatística de pares de caracteres no corpus de treinamento. Palavras frequentes viram um único token; palavras raras ou compostas são fragmentadas em múltiplas subpalavras. É a primeira etapa do pipeline — o modelo nunca vê caracteres soltos, só os IDs desses tokens.",
      },
      attention: {
        simple: "Atenção é o mecanismo que permite ao modelo \"olhar para trás\" na frase enquanto processa cada palavra, decidindo quais outras são relevantes. É parecido com o que você faz ao ler \"ela disse que ele nunca voltaria\" — pra saber quem é \"ele\", seu cérebro busca automaticamente lá atrás por um nome que combine. O modelo faz algo parecido, várias vezes, em paralelo, em cada camada.",
        technical: "Self-attention calcula, para cada token, uma distribuição de pesos sobre todos os outros tokens da sequência, via projeções de Query, Key e Value seguidas de um softmax escalado. Cada camada tem múltiplas \"cabeças\" de atenção, cada uma podendo capturar um tipo diferente de relação (sintática, correferencial, posicional). Empilhando camadas, o modelo constrói representações cada vez mais contextuais.",
      },
      headsExplainer: {
        simple: "Antes de mexer nos seletores abaixo, vale entender o que são \"camada\" e \"cabeça\". O modelo não processa a frase de uma vez só — ela passa por 12 \"camadas\" empilhadas, uma depois da outra, cada uma refinando um pouco mais o entendimento do texto (tipo uma linha de montagem). Dentro de cada camada, existem 12 \"cabeças\" de atenção trabalhando ao mesmo tempo, em paralelo — cada uma olhando a frase com um \"foco\" diferente, sem que ninguém tenha programado manualmente qual foco cada uma teria. É por isso que a comparação entre cabeças, mais abaixo, mostra padrões tão diferentes mesmo estando todas na mesma camada.",
        technical: "\"Camada\" e \"cabeça\" são dois eixos diferentes de organização do modelo. As camadas são sequenciais: a saída da camada N vira a entrada da camada N+1, empilhando 12 blocos de self-attention + feed-forward, cada um refinando a representação contextual. As cabeças são paralelas, dentro de uma mesma camada: a atenção é dividida em 12 subespaços independentes (cada um com suas próprias projeções de Query/Key/Value), permitindo que cada cabeça se especialize, durante o treinamento, num tipo de padrão diferente — sem que esse padrão seja definido manualmente. Duas cabeças na mesma camada podem ter comportamentos completamente distintos, como mostra a comparação abaixo.",
      },
      hidden: {
        simple: "Cada camada não só decide onde \"olhar\" (isso é a atenção) — ela também atualiza a representação interna de cada palavra, um conjunto de 768 números que muda a cada camada. É informação demais pra desenhar de uma vez, então aqui a gente comprime esses 768 números em só 2 (com uma técnica chamada PCA) pra poder ver, de forma aproximada, se o modelo está tratando duas palavras como \"parecidas\" (ficam perto no desenho) ou \"diferentes\" (ficam longe) naquela camada.",
        technical: "Cada camada produz um hidden state de 768 dimensões por token. Aqui projetamos esses vetores em 2D via PCA (as duas componentes de maior variância), calculado no navegador a partir da matriz de Gram token-a-token — não é uma redução \"oficial\" do modelo, é uma aproximação visual. Posições próximas indicam similaridade relativa nessa camada especificamente, não um rótulo semântico direto como \"tom\" ou \"sentimento\".",
      },
      next: {
        simple: "Depois de processar a frase inteira, o modelo não \"escolhe\" a próxima palavra direto — ele calcula uma probabilidade pra cada palavra possível do vocabulário (todas as dezenas de milhares!) e normalmente escolhe uma das mais prováveis. É como se apostasse em várias palavras ao mesmo tempo, com fichas diferentes em cada uma.",
        technical: "A camada final produz um vetor de logits do tamanho do vocabulário, convertido em probabilidades via softmax. A escolha do próximo token pode ser determinística (greedy: sempre o de maior probabilidade) ou estocástica (sampling com temperature, top-k ou top-p) — o que afeta diretamente a criatividade vs. previsibilidade do texto gerado.",
      },
      write: {
        simple: "Aqui você usa exatamente o mesmo mecanismo da seção anterior, só que repetido várias vezes seguidas — cada palavra que você aceita vira parte do contexto pra calcular a próxima. É basicamente como funciona o autocomplete do seu celular, só que agora você está vendo por dentro.",
        technical: "Cada clique dispara um novo forward pass com o texto acumulado como entrada, recalculando toda a distribuição do zero (sem cache incremental nessa implementação). É a mesma geração autoregressiva usada em produção, só exposta passo a passo em vez de rodar automaticamente até um token de parada.",
      },
      compare: {
        simple: "Duas frases quase iguais, com só a ordem das palavras trocada, podem fazer o modelo \"prestar atenção\" de um jeito completamente diferente — e às vezes até mudar o quão confiante ele fica sobre a próxima palavra. Compare duas versões da mesma frase e veja o que muda.",
        technical: "Compara tokenização, o alvo de maior atenção (na camada/cabeça selecionada na seção de Atenção acima) e a distribuição top-5 da próxima palavra entre duas frases — útil pra ver como mudanças estruturais (ordem, sujeito/objeto) afetam as representações internas do modelo.",
      },
      sim: {
        simple: "Até aqui o modelo só previa a próxima palavra. Agora ele vira um \"agente\": você pede algo do seu jeito e ele muda um jogo. O truque é que este modelo pequeno não sabe programar — todas as peças do jogo já estão prontas, e o trabalho dele é só entender qual peça você pediu. Pra isso, ele transforma seu pedido numa lista de números (como na seção 3) e procura os mais parecidos entre 100 pedidos de exemplo.\n\nOs assistentes grandes usam ferramentas de um jeito parecido: o modelo escolhe a ferramenta e um programa comum executa — a diferença é que eles também conseguem escrever o código. E os bugs do jogo são de propósito: assistentes de código de verdade também erram, e é por isso que alguém sempre precisa testar e revisar o que eles fazem.",
        technical: "Roteamento de intenção por similaridade: em cada camada de 4 a 12, tiramos a média dos hidden states dos tokens do pedido (sem o 1º, cujas ativações gigantes dominariam a média no GPT-2). Os vetores são centralizados pela média do banco e comparados por similaridade de cosseno com 100 exemplos rotulados (10 por comando); a pontuação de cada comando é a média dos 2 vizinhos mais próximos, mais um bônus fixo de 0,15 quando o pedido contém alguma palavra-chave daquele comando.\n\nAntes testamos o caminho \"clássico\" de few-shot prompting (prever o nome do comando como próximo token): com 124M de parâmetros, acertou só 36% — o modelo é pequeno demais pra aprender a tarefa pelo contexto. Com embeddings, acertou 19 de 20 pedidos num conjunto de teste novo (a versão em inglês roda outro modelo, o GPT-2 original, com banco e limiares próprios, e acertou 20 de 20 no seu teste). Nos casos incertos ele pergunta em vez de chutar. É o mesmo esqueleto do tool calling em LLMs grandes: o modelo produz uma chamada estruturada e código determinístico a executa.",
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
      "hidden state": "Um vetor de números que representa o estado interno de um token numa camada específica do modelo.",
      "PCA": "Técnica que reduz um monte de números (aqui, 768) para só 2, tentando preservar o máximo possível das diferenças entre eles.",
      "matriz de Gram": "Uma tabela auxiliar usada aqui pra calcular a PCA de forma mais rápida quando há poucos tokens.",
      "similaridade de cosseno": "Mede o quanto dois vetores apontam na mesma direção: 1 = idênticos, 0 = sem relação.",
      "few-shot prompting": "Colocar alguns exemplos resolvidos no próprio texto de entrada pra que o modelo \"pegue o jeito\" da tarefa sem ser treinado de novo.",
      "tool calling": "Quando o modelo, em vez de responder com texto, gera uma chamada estruturada (ex: JSON) pra um programa executar.",
    },
  },
  en: {
    loadingTitle: "Getting your model ready…",
    loadingNote: "This only happens the first time — after that it's cached in your browser. Everything runs locally, no data ever leaves your machine.",
    inputEyebrow: "TYPE A SENTENCE",
    analyzeBtn: "Analyze",
    defaultPrompt: "She said he would never come back",
    defaultWrite: "Once upon a time",
    examples: [
      "She said he would never come back",
      "She felt overjoyed",
      "The capital of France is",
    ],
    tokensTitle: "Tokenization",
    tokensHint: "Each block is a token — the smallest unit the model sees.",
    attentionTitle: "Attention",
    attentionHint: "Pick a layer and a head to see where each token \"looks\".",
    attentionBreakdownTitle: "Where each token looks (top 3)",
    layerLabel: "Layer",
    headLabel: "Head",
    headCompareToggle: "Compare heads",
    headCompareBack: "← Back",
    headComparisonCaption: "Same layer, three different heads — each one can learn to notice a different kind of relationship between words.",
    headLabelShort: "Head",
    hiddenTitle: "Hidden layers",
    hiddenHint: "Pick a layer to see how the model repositions each token internally.",
    arcsCaption: "Each arrow points to the most-attended token; the line's thickness shows how strong that attention is. A ring over a token means it mostly attends to itself.",
    elevatorCaption: "Only the distance between points matters here — exact position and the axes carry no meaning on their own.",
    elevatorLegendNear: "most similar",
    elevatorLegendFar: "most different",
    hiddenLayerLabel: "Layer",
    hiddenEmbeddingLabel: "Embedding (input)",
    hiddenLayerName: "Layer",
    nextTitle: "Next word",
    nextHint: "The probability distribution the model computed for the next word. Click a bar to continue the sentence with it.",
    writeTitle: "Write together with the model",
    writeHint: "Continue a sentence and let the model suggest the next word, one at a time.",
    writeStart: "Start",
    writeReset: "Reset",
    compareTitle: "Compare sentences",
    compareHint: "See how small changes in the sentence change what the model \"sees\".",
    compareLabelA: "Sentence A",
    compareLabelB: "Sentence B",
    compareBtn: "Compare",
    compareDefaultA: "Mary helped John",
    compareDefaultB: "John helped Mary",
    compareAttentionInsight: (tok) => `The token that receives the most attention is "${tok}".`,
    simTitle: "Build a game with the AI",
    simHint: "Chat with the model and watch it \"code\" a snake game, piece by piece.",
    footer: "Open-source project — Capstone, Generative AI & LLMs Postgrad (PUC-Rio / ICA).",
    sobreLink: "sobre.html?lang=en",
    errorMsg: "Something went wrong loading or running the model. Open the Console (F12) for details.",
    aboutRole: "Capstone Project — Postgraduate in Generative AI & LLMs, PUC-Rio / ICA Lab",
    aboutProjectHeading: "About the project",
    aboutProjectText: "LLM Lens is an open-source interactive tool that shows, step by step, how a language model processes and generates text — tokenization, attention, and next-word prediction — running entirely in the browser, with no server behind it.",
    aboutStackHeading: "Tech stack",
    modelType: {
      slm: {
        label: "Small model (this site)",
        stages: ["Text", "Token", "Embedding", "Attention (12×12)", "Prediction"],
        text: "This is exactly what runs here, inside your browser: a small model (SLM, \"Small Language Model\"), with about 124 million parameters, that only understands text. The sentence you type gets split into tokens, each token becomes a list of numbers (embedding), that information passes through 12 layers — each with 12 attention \"heads\" working in parallel — and at the end the model computes the probability of every possible word being next. It's light and fast enough to run entirely on your computer or phone, but has far less knowledge and reasoning ability than the large models behind online assistants.",
      },
      llm: {
        label: "Large LLM",
        stages: ["Text", "Token", "Embedding", "Dozens of layers", "Memory / Retrieval", "Tools", "Answer"],
        text: "The AI assistants you use day to day (big online chatbots) follow the same basic idea — tokenize, turn into numbers, pass through attention layers — but at a much larger scale: from billions to hundreds of billions of parameters, with dozens of stacked layers. They also usually come with extra pieces this site's model doesn't have: longer conversation memory, retrieval systems that pull in up-to-date information from the internet, and the ability to \"use tools\" (like a calculator or code execution) before answering. That's why they need to run on powerful servers, not in your browser.",
      },
      multimodal: {
        label: "Multimodal",
        stages: ["Text / Image / Audio", "Encoders", "Fusion", "Transformer", "Answer"],
        text: "Some models go beyond text: they also see images, hear audio, and even process video. To do that, each type of information first passes through a specialized \"encoder\" (one for text, one for images, one for audio...) that turns everything into the same kind of numbers. Only after this \"fusion\" step does the combined information enter attention layers similar to the ones you're seeing here — just built to handle several senses at once, and able to answer back in more than one format too.",
      },
    },
    lessons: {
      tokens: {
        simple: "A language model doesn't read whole words like we do — it breaks text into small pieces called tokens before processing anything. Sometimes a token is a whole word, sometimes just a fragment of one, depending on how common it was during training. It's as if the model saw everything through a fixed vocabulary of tens of thousands of \"Lego pieces\", and any text has to be rebuilt from them.",
        technical: "Tokenization uses Byte-Pair Encoding (BPE): a fixed subword vocabulary (here, ~50k units) built from the statistical frequency of character pairs in the training corpus. Frequent words become a single token; rare or compound words get fragmented into multiple subwords. This is the first pipeline stage — the model never sees raw characters, only these token IDs.",
      },
      attention: {
        simple: "Attention is the mechanism that lets the model \"look back\" at the sentence while processing each word, deciding which others are relevant. It's similar to what you do reading \"she said he would never come back\" — to know who \"he\" is, your brain automatically searches earlier for a matching name. The model does something similar, many times, in parallel, at every layer.",
        technical: "Self-attention computes, for each token, a weighted distribution over every other token in the sequence, via Query, Key, and Value projections followed by a scaled softmax. Each layer has multiple attention \"heads\", each potentially capturing a different kind of relationship (syntactic, coreferential, positional). Stacking layers builds increasingly contextual representations.",
      },
      headsExplainer: {
        simple: "Before touching the selectors below, it's worth understanding what \"layer\" and \"head\" mean. The model doesn't process the sentence all at once — it passes through 12 stacked \"layers\", one after another, each one refining the text's understanding a bit further (like an assembly line). Inside each layer, there are 12 attention \"heads\" working at the same time, in parallel — each one looking at the sentence with a different \"focus\", without anyone manually programming what that focus should be. That's why the head comparison further below shows such different patterns even within the same layer.",
        technical: "\"Layer\" and \"head\" are two different organizing axes of the model. Layers are sequential: layer N's output becomes layer N+1's input, stacking 12 self-attention + feed-forward blocks, each refining the contextual representation. Heads are parallel, within a single layer: attention is split into 12 independent subspaces (each with its own Query/Key/Value projections), letting each head specialize, during training, in a different kind of pattern — without that pattern being manually defined. Two heads in the same layer can behave completely differently, as the comparison below shows.",
      },
      hidden: {
        simple: "Each layer doesn't just decide where to \"look\" (that's attention) — it also updates each word's internal representation, a set of 768 numbers that changes at every layer. That's too much to draw at once, so here we compress those 768 numbers down to just 2 (using a technique called PCA) to roughly see whether the model is treating two words as \"similar\" (they end up close together) or \"different\" (far apart) at that layer.",
        technical: "Each layer produces a 768-dimensional hidden state per token. Here we project those vectors into 2D via PCA (the two highest-variance components), computed in the browser from the token-by-token Gram matrix — this isn't an \"official\" reduction from the model, it's a visual approximation. Nearby positions indicate relative similarity at that specific layer, not a direct semantic label like \"tone\" or \"sentiment\".",
      },
      next: {
        simple: "After processing the whole sentence, the model doesn't \"pick\" the next word directly — it computes a probability for every possible word in the vocabulary (tens of thousands of them!) and usually picks one of the most likely ones. It's as if it were betting on many words at once, with different amounts on each.",
        technical: "The final layer produces a logits vector the size of the vocabulary, converted into probabilities via softmax. Next-token selection can be deterministic (greedy: always the highest probability) or stochastic (sampling with temperature, top-k, or top-p) — directly affecting the creativity vs. predictability of the generated text.",
      },
      write: {
        simple: "Here you're using the exact same mechanism from the previous section, just repeated over and over — each word you accept becomes part of the context for computing the next one. It's basically how your phone's autocomplete works, except now you're seeing inside it.",
        technical: "Each click triggers a new forward pass with the accumulated text as input, recomputing the full distribution from scratch (no incremental cache in this implementation). This is the same autoregressive generation used in production, just exposed step-by-step instead of run automatically until a stop token.",
      },
      compare: {
        simple: "Two nearly identical sentences, with just the word order swapped, can make the model \"pay attention\" in a completely different way — and sometimes even change how confident it is about the next word. Compare two versions of the same sentence and see what changes.",
        technical: "Compares tokenization, the top attention target (at the layer/head selected in the Attention section above), and the next-word top-5 distribution between two sentences — useful for seeing how structural changes (word order, subject/object) affect the model's internal representations.",
      },
      sim: {
        simple: "So far the model only predicted the next word. Now it becomes an \"agent\": you ask for something in your own words and it changes a game. The trick is that this small model can't code — every piece of the game is already built, and its only job is to figure out which piece you asked for. To do that, it turns your request into a list of numbers (like in section 3) and finds the most similar ones among 100 example requests.\n\nBig assistants use tools in a similar way: the model picks the tool and ordinary code runs it — the difference is that they can also write the code. And the game's bugs are on purpose: real coding assistants make mistakes too, which is why someone always needs to test and review what they do.",
        technical: "Similarity-based intent routing: at each layer from 4 to 12, we average the hidden states of the request's tokens (excluding the 1st, whose huge activations would dominate the mean in GPT-2). The vectors are centered on the bank's mean and compared via cosine similarity against 100 labeled examples (10 per command); each command scores the mean of its 2 nearest neighbors, plus a fixed 0.15 bonus when the request contains any of that command's keywords.\n\nWe first tried the \"classic\" few-shot prompting route (predicting the command name as the next token), but at 124M parameters it only got 36% right (measured on the Portuguese model) — too small to learn the task from context alone. With embeddings, it got 20 out of 20 right on a fresh test set (the Portuguese version runs a different model, GPorTuguese-2, with its own bank and thresholds, and got 19 out of 20 on its test). When unsure, it asks instead of guessing. It's the same skeleton as tool calling in large LLMs: the model produces a structured call and deterministic code executes it.",
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
      "hidden state": "A vector of numbers representing a token's internal state at a specific layer of the model.",
      "PCA": "A technique that reduces a bunch of numbers (here, 768) down to just 2, trying to preserve as much of the differences between them as possible.",
      "Gram matrix": "A helper table used here to compute PCA faster when there are few tokens.",
      "cosine similarity": "Measures how much two vectors point in the same direction: 1 = identical, 0 = unrelated.",
      "few-shot prompting": "Putting a few solved examples in the input text itself so the model \"gets the hang\" of the task without retraining.",
      "tool calling": "When the model, instead of replying with text, produces a structured call (e.g. JSON) for a program to execute.",
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

  hidden(lang, mode, { layerName, closestA, closestB, closestDist, farthestA, farthestB, farthestDist, varianceExplainedPct }) {
    if (lang === "pt") {
      if (mode === "simple") {
        return `Nessa camada (${layerName}), "${closestA}" e "${closestB}" ficaram bem próximos nesse desenho — o modelo parece estar tratando os dois de forma parecida por aqui. Já "${farthestA}" e "${farthestB}" ficaram bem afastados, tratados como bem diferentes entre si.`;
      }
      return `Camada: ${layerName}. Par mais próximo na projeção 2D: "${closestA}" – "${closestB}" (distância ${closestDist.toFixed(2)}). Par mais distante: "${farthestA}" – "${farthestB}" (distância ${farthestDist.toFixed(2)}). As duas componentes principais capturam ${varianceExplainedPct}% da variância total dos hidden states nessa camada — o resto foi descartado na compressão pra 2D.`;
    }
    if (mode === "simple") {
      return `At this layer (${layerName}), "${closestA}" and "${closestB}" ended up close together in this drawing — the model seems to be treating them similarly here. Meanwhile "${farthestA}" and "${farthestB}" ended up far apart, treated as quite different from each other.`;
    }
    return `Layer: ${layerName}. Closest pair in the 2D projection: "${closestA}" – "${closestB}" (distance ${closestDist.toFixed(2)}). Farthest pair: "${farthestA}" – "${farthestB}" (distance ${farthestDist.toFixed(2)}). The two principal components capture ${varianceExplainedPct}% of the total hidden-state variance at this layer — the rest was discarded when compressing to 2D.`;
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
  document.getElementById("tokens-title").textContent = t.tokensTitle;
  document.getElementById("tokens-hint").textContent = t.tokensHint;
  document.getElementById("attention-title").textContent = t.attentionTitle;
  document.getElementById("attention-hint").textContent = t.attentionHint;
  document.getElementById("attention-breakdown-title").textContent = t.attentionBreakdownTitle;
  document.getElementById("layer-label").textContent = t.layerLabel;
  document.getElementById("head-label").textContent = t.headLabel;
  document.getElementById("head-compare-toggle").textContent = t.headCompareToggle;
  document.getElementById("head-comparison-caption").textContent = t.headComparisonCaption;
  document.getElementById("hidden-title").textContent = t.hiddenTitle;
  document.getElementById("hidden-hint").textContent = t.hiddenHint;
  document.getElementById("hidden-layer-label").textContent = t.hiddenLayerLabel;
  document.getElementById("attention-arcs-caption").textContent = t.arcsCaption;
  document.getElementById("hidden-elevator-caption").textContent = t.elevatorCaption;
  document.getElementById("elevator-legend-near").textContent = t.elevatorLegendNear;
  document.getElementById("elevator-legend-far").textContent = t.elevatorLegendFar;
  document.getElementById("next-title").textContent = t.nextTitle;
  document.getElementById("next-hint").textContent = t.nextHint;
  document.getElementById("write-title").textContent = t.writeTitle;
  document.getElementById("write-hint").textContent = t.writeHint;
  document.getElementById("write-start-btn").textContent = t.writeStart;
  document.getElementById("write-reset-btn").textContent = t.writeReset;
  document.getElementById("compare-title").textContent = t.compareTitle;
  document.getElementById("compare-hint").textContent = t.compareHint;
  document.getElementById("compare-label-a").textContent = t.compareLabelA;
  document.getElementById("compare-label-b").textContent = t.compareLabelB;
  document.getElementById("compare-btn").textContent = t.compareBtn;
  document.getElementById("compare-input-a").value = t.compareDefaultA;
  document.getElementById("compare-input-b").value = t.compareDefaultB;
  document.getElementById("sim-title").textContent = t.simTitle;
  document.getElementById("sim-hint").textContent = t.simHint;
  document.getElementById("footer-text").textContent = t.footer;
  document.getElementById("sobre-link").textContent = LANG === "pt" ? "Sobre" : "About";
  document.getElementById("sobre-link").href = t.sobreLink;
  document.getElementById("bastidores-link").textContent = LANG === "pt" ? "Bastidores" : "Behind the scenes";
  document.getElementById("bastidores-link").href = `bastidores.html?lang=${LANG}`;
  document.getElementById("about-bastidores-link").textContent = LANG === "pt" ? "🛠️ Bastidores" : "🛠️ Behind the scenes";
  document.getElementById("about-bastidores-link").href = `bastidores.html?lang=${LANG}`;

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

  document.getElementById("model-type-btn-slm").textContent = t.modelType.slm.label;
  document.getElementById("model-type-btn-llm").textContent = t.modelType.llm.label;
  document.getElementById("model-type-btn-multimodal").textContent = t.modelType.multimodal.label;
  setModelType("slm");

  applyLessonTexts();
  updateAttentionViewMode();
}

// Alternador "Modelo pequeno / LLM grande / Multimodal" logo abaixo do
// pipeline: puramente ilustrativo (não afeta a análise, que sempre usa o
// modelo pequeno real) -- serve pra situar o que este site demonstra dentro
// do panorama maior de modelos de linguagem.
function setModelType(type) {
  const info = t.modelType[type];
  ["slm", "llm", "multimodal"].forEach((tp) => {
    document.getElementById(`model-type-btn-${tp}`).classList.toggle("active", tp === type);
  });
  const container = document.getElementById("pipeline");
  container.innerHTML = "";
  info.stages.forEach((label, i) => {
    const span = document.createElement("span");
    span.className = "stage";
    span.textContent = label;
    container.appendChild(span);
    if (i < info.stages.length - 1) {
      const arrow = document.createElement("span");
      arrow.className = "arrow";
      arrow.textContent = "→";
      container.appendChild(arrow);
    }
  });
  document.getElementById("model-type-text").textContent = info.text;
}

document.querySelectorAll(".model-type-btn").forEach((btn) => {
  btn.addEventListener("click", () => setModelType(btn.dataset.type));
});

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
  renderLessonText(document.getElementById("attention-heads-explainer"), t.lessons.headsExplainer[explanationMode], glossary);
  renderLessonText(document.getElementById("hidden-lesson"), t.lessons.hidden[explanationMode], glossary);
  renderLessonText(document.getElementById("next-lesson"), t.lessons.next[explanationMode], glossary);
  renderLessonText(document.getElementById("write-lesson"), t.lessons.write[explanationMode], glossary);
  renderLessonText(document.getElementById("compare-lesson"), t.lessons.compare[explanationMode], glossary);
  renderLessonText(document.getElementById("sim-lesson"), t.lessons.sim[explanationMode], glossary);
}

// Modo Iniciante mostra os arcos de atenção (mais intuitivo); modo Técnico
// mostra a grade + breakdown (mais denso/preciso). Mesma seleção de
// camada/cabeça alimenta as duas visualizações.
// Comparar cabeças substitui a visão de uma cabeça só (não fica lado a lado
// com ela) -- ter os dois ao mesmo tempo, com os mesmos seletores de camada/
// cabeça em cima, ficava confuso sobre o que controlava o quê.
let headCompareMode = false;

function updateAttentionViewMode() {
  if (headCompareMode) {
    document.getElementById("attention-arcs-view").style.display = "none";
    document.getElementById("attention-technical-view").style.display = "none";
    document.getElementById("attention-insight").style.display = "none";
    document.getElementById("head-select-label").style.display = "none";
    return;
  }
  // A grade + breakdown ficam sempre visíveis; os arcos aparecem como
  // complemento extra no modo Iniciante, não como substituto da grade.
  const isSimple = explanationMode === "simple";
  document.getElementById("attention-arcs-view").style.display = isSimple ? "block" : "none";
  document.getElementById("attention-technical-view").style.display = "block";
  document.getElementById("attention-insight").style.display = "block";
  document.getElementById("head-select-label").style.display = "";
}

document.getElementById("mode-switch").addEventListener("click", () => {
  explanationMode = explanationMode === "simple" ? "technical" : "simple";
  document.getElementById("mode-switch").classList.toggle("on", explanationMode === "technical");
  document.getElementById("mode-label-simple").classList.toggle("active", explanationMode === "simple");
  document.getElementById("mode-label-technical").classList.toggle("active", explanationMode === "technical");
  applyLessonTexts();
  updateAttentionViewMode();
  simulator?.setMode(explanationMode);
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
let simulator = null; // seção 7 — criada depois que o modelo carrega

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
  const rawModel = await AutoModelForCausalLM.from_pretrained(modelId, {
    dtype: "q8",
    progress_callback: progressCallback,
  });

  // Fila única de inferências: o ONNX Runtime Web trava com duas chamadas
  // concorrentes na mesma sessão, e o simulador (seção 7) roda o modelo em
  // segundo plano enquanto a pessoa pode estar usando as outras seções.
  let queue = Promise.resolve();
  model = (inputs) => {
    const run = queue.then(() => rawModel(inputs));
    queue = run.catch(() => {});
    return run;
  };
  model.config = rawModel.config;

  modelReady = true;
  updateProgress(100, null);
  document.getElementById("loading-overlay").classList.add("hidden");
}

// ============================================================================
// ANÁLISE DO MODELO — tokenização, atenção, próximo token
// ============================================================================
let lastAnalysis = null;

async function runForward(inputs) {
  const outputs = await model({ ...inputs });
  const expectedLayers = model.config.n_layer ?? 12;

  const attentions = [];
  for (let i = 0; i < expectedLayers; i++) {
    const layerOutput = outputs[`attentions.${i}`];
    if (!layerOutput) break;
    attentions.push(layerOutput);
  }

  // hidden_states tem expectedLayers + 1 tensores: a saída do embedding
  // (índice 0) seguida da saída de cada camada.
  const hiddenStates = [];
  for (let i = 0; i <= expectedLayers; i++) {
    const layerOutput = outputs[`hidden_states.${i}`];
    if (!layerOutput) break;
    hiddenStates.push(layerOutput);
  }

  return { attentions, hiddenStates };
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
let currentHiddenStates = null;
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

  layerSelect.onchange = () => { renderAttentionGrid(); renderAttentionArcs(); renderHeadComparison(); };
  headSelect.onchange = () => { renderAttentionGrid(); renderAttentionArcs(); };
}

// Mostra 3 cabeças da mesma camada lado a lado -- prova visualmente que cada
// cabeça nota um padrão diferente, em vez de só afirmar isso no texto.
function renderHeadComparison() {
  const view = document.getElementById("head-comparison-view");
  if (view.style.display === "none") return; // só recalcula se estiver aberto

  const layer = parseInt(document.getElementById("layer-select").value || 0);
  const numHeads = currentAttentions[0].dims[1];
  const sampleHeads = [0, Math.floor((numHeads - 1) / 2), numHeads - 1];
  const tokens = currentTokenTexts;

  const grid = document.getElementById("head-comparison-grid");
  grid.innerHTML = "";

  // Passar o mouse numa palavra acende ela nos 3 mini-diagramas ao mesmo
  // tempo -- reforça que é a MESMA palavra, só a cabeça (o "olhar") muda.
  const controllers = [];

  sampleHeads.forEach((headIdx, k) => {
    const matrix = tensorToMatrix(currentAttentions[layer], headIdx);

    const item = document.createElement("div");
    item.className = "head-comparison-item";
    const h4 = document.createElement("h4");
    h4.textContent = `${t.headLabelShort} ${headIdx}`;
    item.appendChild(h4);

    // Guarda a célula vencedora (maior valor) de cada linha -- pra acender
    // no hover em vez de deixar fixa, sincronizada com os arcos e as outras
    // duas cabeças comparadas.
    const winnerTdByRow = [];
    const table = document.createElement("table");
    table.className = "head-mini-grid";
    matrix.forEach((row, rowIdx) => {
      const tr = document.createElement("tr");
      const rowMax = Math.max(...row);
      row.forEach((weight) => {
        const td = document.createElement("td");
        td.style.background = `rgba(79, 209, 197, ${weight.toFixed(2)})`;
        if (weight === rowMax) winnerTdByRow[rowIdx] = td;
        tr.appendChild(td);
      });
      table.appendChild(tr);
    });
    item.appendChild(table);

    // Mesmo diagrama de arcos da seção principal, em versão compacta --
    // prova visualmente que cada cabeça "olha" pra lugares diferentes.
    const arcSvg = document.createElementNS(SVG_NS, "svg");
    arcSvg.setAttribute("class", "head-mini-arcs");
    arcSvg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    item.appendChild(arcSvg);
    controllers[k] = drawArcDiagram(arcSvg, matrix, tokens, {
      compact: true,
      onTokenHover: (tokenIdx, isActive) => {
        controllers.forEach((c, ci) => { if (ci !== k) c.setActive(tokenIdx, isActive); });
      },
      onSetActive: (tokenIdx, isActive) => {
        const td = winnerTdByRow[tokenIdx];
        if (td) td.classList.toggle("hot-cell-hover", isActive);
      },
    });

    grid.appendChild(item);
  });
}

document.getElementById("head-compare-toggle").addEventListener("click", () => {
  headCompareMode = !headCompareMode;
  document.getElementById("head-comparison-view").style.display = headCompareMode ? "block" : "none";
  document.getElementById("head-compare-toggle").textContent = headCompareMode ? t.headCompareBack : t.headCompareToggle;
  updateAttentionViewMode();
  if (headCompareMode) renderHeadComparison();
});

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

// "Arco de atenção": para cada token, desenha uma linha curva até o token que
// ele mais atende (top-1, ignorando a si mesmo) — a mesma informação da
// grade/breakdown, só que em formato mais intuitivo pro modo Iniciante.
const SVG_NS = "http://www.w3.org/2000/svg";

// Dispositivos sem mouse de verdade (touch/tablet) não têm estado de hover --
// usado pra trocar a interação por uma animação automática nesses casos.
const IS_TOUCH_DEVICE = window.matchMedia("(hover: none)").matches;

function lerpColor(hexA, hexB, t) {
  const a = [1, 3, 5].map((i) => parseInt(hexA.slice(i, i + 2), 16));
  const b = [1, 3, 5].map((i) => parseInt(hexB.slice(i, i + 2), 16));
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * t));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

// Desenha o diagrama de arcos (setas + anéis de auto-atenção) dentro de um
// <svg> qualquer -- usado tanto na visualização principal quanto nas versões
// compactas da comparação entre cabeças, pra não duplicar a lógica.
function drawArcDiagram(svg, matrix, tokens, { compact = false, onTokenHover = null, onSetActive = null } = {}) {
  const n = tokens.length;
  const slotWidth = compact ? 64 : 78;
  const width = Math.max(compact ? 260 : 420, n * slotWidth);
  const height = compact ? 150 : 200;
  const baselineY = compact ? 110 : 150;
  const labelY = compact ? 130 : 172;

  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = "";

  const xFor = (i) => (width / n) * (i + 0.5);

  // Ponto e tangente (derivada) na curva de Bézier quadrática, parâmetro t
  // entre 0 (início) e 1 (fim).
  const bezierPoint = (p0, p1, p2, t) => ({
    x: (1 - t) ** 2 * p0.x + 2 * (1 - t) * t * p1.x + t ** 2 * p2.x,
    y: (1 - t) ** 2 * p0.y + 2 * (1 - t) * t * p1.y + t ** 2 * p2.y,
  });
  const bezierTangent = (p0, p1, p2, t) => {
    const dx = 2 * (1 - t) * (p1.x - p0.x) + 2 * t * (p2.x - p1.x);
    const dy = 2 * (1 - t) * (p1.y - p0.y) + 2 * t * (p2.y - p1.y);
    const len = Math.hypot(dx, dy) || 1;
    return { x: dx / len, y: dy / len };
  };

  // Top-1 alvo de verdade por linha da matriz -- inclui o próprio token.
  // Excluir a auto-atenção (como a versão anterior fazia) escondia o padrão
  // real em cabeças que atendem principalmente a si mesmas: o "segundo
  // colocado" nesses casos costuma ser só um resíduo pequeno, não uma
  // relação de verdade -- mostrar ele como se fosse o alvo principal engana.
  const links = matrix.map((row, i) => {
    let bestJ = -1;
    let bestW = -1;
    row.forEach((w, j) => {
      if (w > bestW) { bestW = w; bestJ = j; }
    });
    return { source: i, target: bestJ, weight: bestW };
  });

  const arcPaths = [];

  links.forEach(({ source, target, weight }) => {
    if (target < 0 || weight <= 0.01) return;
    const x1 = xFor(source);

    if (target === source) {
      // Alvo principal é o próprio token -- em vez de forçar uma seta pra
      // outro lugar (enganoso), desenha um anel de auto-atenção no token.
      const group = document.createElementNS(SVG_NS, "g");
      group.setAttribute("class", "arc-group");
      group.dataset.source = source;
      const ring = document.createElementNS(SVG_NS, "circle");
      ring.setAttribute("cx", x1);
      ring.setAttribute("cy", baselineY);
      ring.setAttribute("r", compact ? 8 : 10);
      ring.setAttribute("fill", "none");
      ring.setAttribute("class", "arc-line");
      ring.setAttribute("stroke-width", (2 + weight * 8).toFixed(2));
      ring.appendChild(Object.assign(document.createElementNS(SVG_NS, "title"), {
        textContent: `${Math.round(weight * 100)}%`,
      }));
      group.appendChild(ring);
      svg.appendChild(group);
      arcPaths.push(group);
      return;
    }

    const x2 = xFor(target);
    const arcHeight = Math.min(110, 30 + Math.abs(x2 - x1) * 0.45);
    const midX = (x1 + x2) / 2;
    const ctrlY = baselineY - arcHeight;
    const p0 = { x: x1, y: baselineY };
    const p1 = { x: midX, y: ctrlY };
    const p2 = { x: x2, y: baselineY };
    const strokeWidth = 2 + weight * 8;

    const group = document.createElementNS(SVG_NS, "g");
    group.setAttribute("class", "arc-group");
    group.dataset.source = source;

    // A linha para exatamente onde o triângulo da seta começa (t=0.9) --
    // desenhados a partir do mesmo ponto, encaixam sem sobra nem buraco.
    const basePoint = bezierPoint(p0, p1, p2, 0.9);
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", `M ${x1} ${baselineY} Q ${midX} ${ctrlY} ${basePoint.x} ${basePoint.y}`);
    path.setAttribute("class", "arc-line");
    path.setAttribute("stroke-width", strokeWidth.toFixed(2));
    path.setAttribute("stroke-linecap", "round");
    group.appendChild(path);

    // Seta desenhada manualmente (não como SVG <marker>) -- assim controlamos
    // exatamente onde ela encosta na linha e no ponto, sem os problemas de
    // escala/sobreposição que os marcadores nativos do SVG têm.
    const dir = bezierTangent(p0, p1, p2, 0.9);
    const perp = { x: -dir.y, y: dir.x };
    const arrowLen = compact ? 10 : 13;
    const arrowWidth = (compact ? 4 : 5) + weight * 4;
    const tip = { x: basePoint.x + dir.x * arrowLen, y: basePoint.y + dir.y * arrowLen };
    const corner1 = { x: basePoint.x + perp.x * arrowWidth, y: basePoint.y + perp.y * arrowWidth };
    const corner2 = { x: basePoint.x - perp.x * arrowWidth, y: basePoint.y - perp.y * arrowWidth };
    const arrow = document.createElementNS(SVG_NS, "polygon");
    arrow.setAttribute("points", `${tip.x},${tip.y} ${corner1.x},${corner1.y} ${corner2.x},${corner2.y}`);
    arrow.setAttribute("class", "arc-arrowhead");
    group.appendChild(arrow);

    svg.appendChild(group);
    arcPaths.push(group);
  });

  const dotsByIndex = [];

  const setActive = (i, isActive) => {
    const dot = dotsByIndex[i];
    if (dot) dot.classList.toggle("active", isActive);
    arcPaths.forEach((p) => { if (parseInt(p.dataset.source) === i) p.classList.toggle("active", isActive); });
    if (onSetActive) onSetActive(i, isActive);
  };

  tokens.forEach((tok, i) => {
    const x = xFor(i);

    const dot = document.createElementNS(SVG_NS, "circle");
    dot.setAttribute("cx", x);
    dot.setAttribute("cy", baselineY);
    dot.setAttribute("r", compact ? 4 : 6);
    dot.setAttribute("class", "arc-token-dot");
    dot.dataset.index = i;
    dotsByIndex[i] = dot;

    const label = document.createElementNS(SVG_NS, "text");
    label.setAttribute("x", x);
    label.setAttribute("y", labelY);
    label.setAttribute("class", "arc-token");
    label.textContent = tok;
    label.dataset.index = i;

    const highlight = () => { setActive(i, true); if (onTokenHover) onTokenHover(i, true); };
    const unhighlight = () => { setActive(i, false); if (onTokenHover) onTokenHover(i, false); };
    [dot, label].forEach((el) => {
      el.addEventListener("mouseenter", highlight);
      el.addEventListener("mouseleave", unhighlight);
    });

    svg.appendChild(dot);
    svg.appendChild(label);
  });

  return { arcPaths, setActive };
}

function renderAttentionArcs() {
  if (!currentAttentions || !currentTokenTexts) return;
  const layer = parseInt(document.getElementById("layer-select").value || 0);
  const head = parseInt(document.getElementById("head-select").value || 0);
  const matrix = tensorToMatrix(currentAttentions[layer], head);
  const tokens = currentTokenTexts;

  const svg = document.getElementById("attention-arcs-svg");
  const { arcPaths } = drawArcDiagram(svg, matrix, tokens);

  if (IS_TOUCH_DEVICE) {
    // Sem hover disponível: revela os arcos um de cada vez ao longo de uns
    // 4-5s, em vez de todos aparecerem juntos -- dá pra perceber a criação.
    arcPaths.forEach((group) => group.classList.add("pending"));
    const totalMs = 4500;
    const stepMs = arcPaths.length ? totalMs / arcPaths.length : 0;
    arcPaths.forEach((group, i) => {
      setTimeout(() => group.classList.remove("pending"), 300 + i * stepMs);
    });
  }
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
    th.className = "col-header";
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
    rowLabel.className = "row-label";
    tr.appendChild(rowLabel);

    const rowMax = Math.max(...row);
    if (rowMax > focusedScore) { focusedScore = rowMax; focusedRow = i; }
    if (rowMax < spreadScore) { spreadScore = rowMax; spreadRow = i; }

    row.forEach((weight, j) => {
      colSums[j] += weight;
      const td = document.createElement("td");
      td.style.background = `rgba(79, 209, 197, ${weight.toFixed(2)})`;
      if (weight === rowMax) td.classList.add("hot-cell");
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

// ---------- Renderização: Camadas ocultas (elevador + PCA) ----------
// PCA calculada no navegador via "dual PCA": em vez de decompor a matriz de
// covariância (768x768, cara), decompomos a matriz de Gram token-a-token
// (seqLen x seqLen, minúscula) — os autovetores dela, escalados pela raiz do
// autovalor, já são exatamente as coordenadas 2D projetadas de cada token.
function powerIterationEigen(matrix, n, deflateVec) {
  let v = new Array(n).fill(1 / Math.sqrt(n));
  let lastVal = 0;
  for (let iter = 0; iter < 200; iter++) {
    const next = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < n; j++) sum += matrix[i][j] * v[j];
      next[i] = sum;
    }
    if (deflateVec) {
      const dot = next.reduce((s, x, i) => s + x * deflateVec[i], 0);
      for (let i = 0; i < n; i++) next[i] -= dot * deflateVec[i];
    }
    const norm = Math.sqrt(next.reduce((s, x) => s + x * x, 0)) || 1;
    for (let i = 0; i < n; i++) next[i] /= norm;
    v = next;
  }
  let val = 0;
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) sum += matrix[i][j] * v[j];
    val += v[i] * sum;
  }
  return { vec: v, val };
}

function computePCA2D(vectors) {
  const n = vectors.length;
  const dim = vectors[0].length;

  const mean = new Float32Array(dim);
  for (const vec of vectors) for (let d = 0; d < dim; d++) mean[d] += vec[d] / n;
  const centered = vectors.map((vec) => {
    const c = new Float32Array(dim);
    for (let d = 0; d < dim; d++) c[d] = vec[d] - mean[d];
    return c;
  });

  const gram = [];
  for (let i = 0; i < n; i++) {
    gram.push(new Array(n).fill(0));
    for (let j = 0; j < n; j++) {
      let dot = 0;
      for (let d = 0; d < dim; d++) dot += centered[i][d] * centered[j][d];
      gram[i][j] = dot;
    }
  }

  const totalVariance = gram.reduce((s, row, i) => s + row[i], 0) || 1;

  const pc1 = powerIterationEigen(gram, n, null);
  const pc2 = n > 1 ? powerIterationEigen(gram, n, pc1.vec) : { vec: new Array(n).fill(0), val: 0 };

  const scale1 = Math.sqrt(Math.max(pc1.val, 0));
  const scale2 = Math.sqrt(Math.max(pc2.val, 0));
  const points = pc1.vec.map((v, i) => ({ x: v * scale1, y: pc2.vec[i] * scale2 }));

  const varianceExplainedPct = Math.round(((Math.max(pc1.val, 0) + Math.max(pc2.val, 0)) / totalVariance) * 100);

  return { points, varianceExplainedPct };
}

function populateHiddenLayerSelect() {
  const select = document.getElementById("hidden-layer-select");
  select.innerHTML = "";
  const numStates = currentHiddenStates.length; // n_layer + 1

  for (let i = 0; i < numStates; i++) {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = i === 0 ? t.hiddenEmbeddingLabel : `${t.hiddenLayerName} ${i}`;
    select.appendChild(opt);
  }
  select.value = 0; // começa na entrada (embedding), ecoando o "andar térreo" do elevador

  select.onchange = renderLayerElevator;
}

function renderLayerElevator() {
  if (!currentHiddenStates || !currentTokenTexts) return;
  const layerIdx = parseInt(document.getElementById("hidden-layer-select").value || 0);
  const tensor = currentHiddenStates[layerIdx];
  const [, seqLen, hiddenSize] = tensor.dims;
  const data = tensor.data;

  const vectors = [];
  for (let i = 0; i < seqLen; i++) {
    vectors.push(data.slice(i * hiddenSize, (i + 1) * hiddenSize));
  }

  const { points, varianceExplainedPct } = computePCA2D(vectors);
  const tokens = currentTokenTexts;

  const width = 480;
  const height = 320;
  const padding = 40;

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;

  const svgX = (x) => padding + ((x - minX) / rangeX) * (width - 2 * padding);
  const svgY = (y) => height - padding - ((y - minY) / rangeY) * (height - 2 * padding);

  const svg = document.getElementById("hidden-elevator-svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = "";

  const axisX = document.createElementNS(SVG_NS, "line");
  axisX.setAttribute("x1", padding); axisX.setAttribute("x2", width - padding);
  axisX.setAttribute("y1", height / 2); axisX.setAttribute("y2", height / 2);
  axisX.setAttribute("class", "elevator-axis");
  svg.appendChild(axisX);
  const axisY = document.createElementNS(SVG_NS, "line");
  axisY.setAttribute("x1", width / 2); axisY.setAttribute("x2", width / 2);
  axisY.setAttribute("y1", padding); axisY.setAttribute("y2", height - padding);
  axisY.setAttribute("class", "elevator-axis");
  svg.appendChild(axisY);

  // Distâncias par a par na projeção 2D -- alimentam o insight dinâmico e a
  // escala de cor ciano->azul usada nas linhas de distância (hover/animação).
  let closestDist = Infinity, farthestDist = -Infinity;
  let closestPair = [0, 1], farthestPair = [0, 1];
  for (let i = 0; i < tokens.length; i++) {
    for (let j = i + 1; j < tokens.length; j++) {
      const d = Math.hypot(points[i].x - points[j].x, points[i].y - points[j].y);
      if (d < closestDist) { closestDist = d; closestPair = [i, j]; }
      if (d > farthestDist) { farthestDist = d; farthestPair = [i, j]; }
    }
  }
  const distRange = farthestDist - closestDist || 1;
  const colorForDistance = (d) => lerpColor("#4FD1C5", "#F0A868", Math.min(1, Math.max(0, (d - closestDist) / distRange)));

  // Linhas de distância entre um token e todos os outros -- reveladas no
  // hover (desktop) ou animadas automaticamente (toque, sem hover de verdade).
  const distanceLines = tokens.map(() => []);
  tokens.forEach((_, i) => {
    tokens.forEach((_, j) => {
      if (i === j) return;
      const d = Math.hypot(points[i].x - points[j].x, points[i].y - points[j].y);
      const line = document.createElementNS(SVG_NS, "line");
      line.setAttribute("x1", svgX(points[i].x));
      line.setAttribute("y1", svgY(points[i].y));
      line.setAttribute("x2", svgX(points[j].x));
      line.setAttribute("y2", svgY(points[j].y));
      line.setAttribute("class", "elevator-distance-line");
      line.setAttribute("stroke", colorForDistance(d));
      line.setAttribute("stroke-width", 2);
      svg.appendChild(line);
      distanceLines[i].push(line);
    });
  });

  tokens.forEach((tok, i) => {
    const cx = svgX(points[i].x);
    const cy = svgY(points[i].y);

    const dot = document.createElementNS(SVG_NS, "circle");
    dot.setAttribute("cx", cx);
    dot.setAttribute("cy", cy);
    dot.setAttribute("r", 5);
    dot.setAttribute("class", "elevator-point");
    dot.addEventListener("mouseenter", () => distanceLines[i].forEach((l) => l.classList.add("visible")));
    dot.addEventListener("mouseleave", () => distanceLines[i].forEach((l) => l.classList.remove("visible")));
    svg.appendChild(dot);

    const label = document.createElementNS(SVG_NS, "text");
    label.setAttribute("x", cx + 8);
    label.setAttribute("y", cy - 8);
    label.setAttribute("class", "elevator-label");
    label.textContent = tok;
    svg.appendChild(label);
  });

  if (IS_TOUCH_DEVICE) {
    // Sem hover disponível: anima a revelação das linhas do par mais próximo
    // e do par mais distante, uma de cada vez, em vez de tudo aparecer junto.
    const closestLine = distanceLines[closestPair[0]].find((l) => {
      const x2 = parseFloat(l.getAttribute("x2")), y2 = parseFloat(l.getAttribute("y2"));
      return Math.abs(x2 - svgX(points[closestPair[1]].x)) < 0.5 && Math.abs(y2 - svgY(points[closestPair[1]].y)) < 0.5;
    });
    const farthestLine = distanceLines[farthestPair[0]].find((l) => {
      const x2 = parseFloat(l.getAttribute("x2")), y2 = parseFloat(l.getAttribute("y2"));
      return Math.abs(x2 - svgX(points[farthestPair[1]].x)) < 0.5 && Math.abs(y2 - svgY(points[farthestPair[1]].y)) < 0.5;
    });
    setTimeout(() => closestLine && closestLine.classList.add("visible"), 1200);
    setTimeout(() => farthestLine && farthestLine.classList.add("visible"), 2800);
  }

  const layerName = layerIdx === 0 ? t.hiddenEmbeddingLabel : `${t.hiddenLayerName} ${layerIdx}`;
  const text = EXPLAIN.hidden(LANG, explanationMode, {
    layerName,
    closestA: tokens[closestPair[0]],
    closestB: tokens[closestPair[1]],
    closestDist,
    farthestA: tokens[farthestPair[0]],
    farthestB: tokens[farthestPair[1]],
    farthestDist,
    varianceExplainedPct,
  });
  document.getElementById("hidden-insight").textContent = text;
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
  renderAttentionArcs();
  renderHeadComparison();
  renderLayerElevator();
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

    const forward = await runForward(inputs);
    currentAttentions = forward.attentions;
    currentHiddenStates = forward.hiddenStates;
    renderAttentionControls();
    renderAttentionArcs();
    renderHeadComparison();
    populateHiddenLayerSelect();
    renderLayerElevator();

    const nextTokenResult = await computeNextTokenDistribution(text);

    const wordCount = text.split(/\s+/).filter(Boolean).length;
    const tokenStats = { text, tokenCount: tokenIds.length, wordCount };

    lastAnalysis = { tokenStats, nextTokenResult };

    document.getElementById("tokens-insight").textContent = EXPLAIN.tokens(LANG, explanationMode, tokenStats);
    renderAttentionGrid();
    renderNextTokenDistribution(nextTokenResult);

    ["section-tokens", "section-attention", "section-hidden", "section-next", "section-write", "section-compare", "section-sim"].forEach((id) => {
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
// SEÇÃO 6 — Comparar frases
// ============================================================================
async function analyzeForCompare(text) {
  const inputs = await tokenizer(text);
  const tokenIds = Array.from(inputs.input_ids.data).map(Number);
  const tokenTexts = tokenIds.map((id) => tokenizer.decode([id]));

  const forward = await runForward(inputs);
  const layer = parseInt(document.getElementById("layer-select").value || 0);
  const head = parseInt(document.getElementById("head-select").value || 0);
  const matrix = tensorToMatrix(forward.attentions[layer], head);

  const colSums = new Array(tokenTexts.length).fill(0);
  matrix.forEach((row) => row.forEach((w, j) => { colSums[j] += w; }));
  const mostAttendedIdx = colSums.indexOf(Math.max(...colSums));

  const nextTokenResult = await computeNextTokenDistribution(text, 5);

  return { tokenTexts, mostAttendedToken: tokenTexts[mostAttendedIdx], nextTokenResult };
}

function renderCompareBars(container, top) {
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
    container.appendChild(row);
  });
}

function renderCompareResult(containerId, result) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";

  const tokensDiv = document.createElement("div");
  tokensDiv.className = "compare-tokens";
  result.tokenTexts.forEach((tok) => {
    const chip = document.createElement("span");
    chip.className = "token-chip";
    chip.textContent = tok;
    tokensDiv.appendChild(chip);
  });
  container.appendChild(tokensDiv);

  const insight = document.createElement("p");
  insight.className = "compare-insight";
  insight.textContent = t.compareAttentionInsight(result.mostAttendedToken);
  container.appendChild(insight);

  const nextView = document.createElement("div");
  nextView.className = "next-token-view";
  renderCompareBars(nextView, result.nextTokenResult.top);
  container.appendChild(nextView);
}

document.getElementById("compare-btn").addEventListener("click", async () => {
  const textA = document.getElementById("compare-input-a").value.trim();
  const textB = document.getElementById("compare-input-b").value.trim();
  if (!textA || !textB || !modelReady) return;

  try {
    // Sequencial, não Promise.all -- rodar duas inferências concorrentes na
    // mesma sessão do ONNX Runtime Web trava (não é thread-safe pra chamadas
    // paralelas nesse ambiente).
    const resultA = await analyzeForCompare(textA);
    const resultB = await analyzeForCompare(textB);
    renderCompareResult("compare-result-a", resultA);
    renderCompareResult("compare-result-b", resultB);
    document.getElementById("compare-results").style.display = "grid";
  } catch (err) {
    alert(t.errorMsg);
    console.error(err);
  }
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
  simulator = initSimulator({ tokenizer, model, lang: LANG, mode: explanationMode });
});
