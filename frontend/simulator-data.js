// ============================================================================
// DADOS DO SIMULADOR "CONSTRUA UM JOGO COM A IA"
// ----------------------------------------------------------------------------
// O modelo NÃO escreve código: ele só escolhe qual comando executar, comparando
// o vetor interno (hidden states) do pedido com os vetores destes exemplos.
// Tudo o que é executado já está pronto aqui e em simulator.js.
//
// Banco de exemplos, palavras-chave e limiares foram calibrados com
// scripts/calibrar_simulador.py (que roda o mesmo modelo quantizado via ONNX).
// Se mudar algo aqui, rode o script de novo pra conferir a taxa de acerto.
// ============================================================================

export const COMMANDS = ["criar", "comida", "parede", "portal", "corpo", "placar", "velocidade", "pausa", "cores", "corrigir"];

// Bugs que cada comando "introduz" sem querer — como no desenvolvimento de verdade.
export const BUGS_FROM = {
  criar: ["trail", "offset", "reverse"],
  comida: ["foodInSnake"],
  parede: ["wallEarly"],
  portal: ["portalShift"],
  corpo: [],
  placar: ["scoreSticky"],
  velocidade: ["turbo"],
  pausa: ["pauseJump"],
  cores: [],
};

// ----------------------------------------------------------------------------
// Banco de exemplos (10 por comando). É contra eles que o pedido é comparado.
// ----------------------------------------------------------------------------
export const BANK = {
  pt: {
    criar: ["faça um jogo da cobrinha", "crie o jogo da cobra", "programa um snake pra mim",
            "construa um jogo", "quero um jogo da cobrinha", "começa o jogo do zero",
            "desenvolve o jogo da cobra", "monta um snake", "cria um jogo novo", "gera um jogo da cobrinha"],
    comida: ["quero que a cobra coma maçãs e cresça", "adicione comida", "coloque frutas para a cobra comer",
             "a cobra precisa comer", "bota uma maçã na tela", "ela tem que crescer quando comer",
             "acrescenta a comida", "aparece comida no mapa", "a cobra come e aumenta de tamanho", "coloca alimento"],
    parede: ["se ela bater na parede tem que morrer", "adicione a colisão com a parede", "morrer ao bater na borda",
             "a parede deve matar a cobra", "game over quando bater na parede", "sistema de batida na parede",
             "bater no muro mata", "encostar no limite da tela é fatal", "colisão com as bordas", "quando a cobra bater na parede ela morre"],
    portal: ["deixa a cobra atravessar a borda e sair do outro lado", "passar pela parede", "sair de um lado e aparecer do outro",
             "a tela dá a volta", "atravessar as paredes", "teletransportar para o outro lado da tela",
             "sem paredes", "ela passa direto pela borda", "reaparece do lado oposto", "a cobra não morre na parede, atravessa"],
    corpo: ["ela deve morrer quando morder o próprio rabo", "colisão com o próprio corpo", "se a cobra bater nela mesma perde",
            "morrer ao morder a cauda", "não pode encostar no próprio corpo", "a cobra se mordeu, tem que morrer",
            "bater no próprio corpo é game over", "não deixa ela passar por cima dela mesma", "se cruzar o corpo morre", "auto colisão"],
    placar: ["mostra quantos pontos eu fiz", "adicione um placar", "contar a pontuação", "quero ver o score",
             "guardar o recorde", "mostrar os pontos na tela",
             "adicione pontos", "cada maçã vale um ponto", "coloca um contador de pontos", "mostra o recorde"],
    velocidade: ["o jogo podia ficar mais rápido com o tempo", "aumentar a velocidade", "a cobra fica mais rápida quando come",
                 "deixa o jogo mais difícil", "tá muito devagar", "acelera a cobra",
                 "aumenta a dificuldade", "fica mais veloz a cada fase", "a cobra está lenta demais", "acelera toda vez que ela come"],
    pausa: ["quero poder parar o jogo um pouco", "adicione pausa", "pausar com a tecla espaço", "botão de pausa",
            "congelar o jogo", "parar e continuar depois",
            "aperta espaço pra pausar", "deixa eu pausar", "quero dar uma pausa no meio", "tecla para pausar"],
    cores: ["deixa o jogo mais bonito e colorido", "melhore o visual", "mude as cores", "capricha no design",
            "deixa mais bonito", "coloca olhos na cobra",
            "tá muito feio", "deixa com cara profissional", "muda a aparência", "coloca um tema bonito"],
    corrigir: ["tem uns bugs estranhos, conserta aí", "corrija os bugs", "arruma os erros", "o jogo está bugado",
               "conserta o rastro da cobra", "tem algo errado, corrige",
               "corrija o jogo", "tem um problema", "resolve os bugs", "o jogo está estranho, arruma"],
  },
  en: {
    criar: ["make a snake game", "build the snake game", "code a snake game for me", "create a game",
            "I want a snake game", "start the game from scratch", "develop the snake game", "set up a snake game",
            "create a new game", "generate a snake game"],
    comida: ["I want the snake to eat apples and grow", "add food", "put fruit for the snake to eat",
             "the snake needs to eat", "put an apple on the screen", "it should grow when it eats",
             "add the food", "food appears on the map", "the snake eats and gets longer", "add something to eat"],
    parede: ["if it hits the wall it should die", "add wall collision", "die when hitting the edge",
             "the wall should kill the snake", "game over when hitting the wall", "wall crash system",
             "hitting the border kills it", "touching the edge of the screen is fatal", "collision with the borders",
             "when the snake hits the wall it dies"],
    portal: ["let the snake go through the edge and come out the other side", "pass through the wall",
             "exit one side and appear on the other", "the screen wraps around", "go through the walls",
             "teleport to the other side of the screen", "no walls", "it passes right through the border",
             "reappears on the opposite side", "the snake doesn't die at the wall, it wraps"],
    corpo: ["it should die when it bites its own tail", "collision with its own body", "if the snake hits itself it loses",
            "die when biting the tail", "it can't touch its own body", "the snake bit itself, it should die",
            "hitting its own body is game over", "don't let it pass over itself", "if it crosses its body it dies",
            "self collision"],
    placar: ["show how many points I got", "add a scoreboard", "count the score", "I want to see the score",
             "save the high score", "show the points on screen", "add points", "each apple is worth one point",
             "add a point counter", "show the best score"],
    velocidade: ["the game could get faster over time", "increase the speed", "the snake gets faster when it eats",
                 "make the game harder", "it's too slow", "speed up the snake", "increase the difficulty",
                 "faster every level", "the snake is way too slow", "it speeds up every time it eats"],
    pausa: ["I want to be able to stop the game for a bit", "add pause", "pause with the space key", "pause button",
            "freeze the game", "stop and continue later", "press space to pause", "let me pause",
            "I want to take a break mid game", "a key to pause"],
    cores: ["make the game prettier and colorful", "improve the visuals", "change the colors", "polish the design",
            "make it look nicer", "give the snake eyes", "it looks ugly", "make it look professional",
            "change the look", "add a nice theme"],
    corrigir: ["there are some weird bugs, fix them", "fix the bugs", "fix the errors", "the game is buggy",
               "fix the snake's trail", "something is wrong, fix it", "fix the game", "there's a problem",
               "solve the bugs", "the game looks weird, fix it"],
  },
};

// ----------------------------------------------------------------------------
// Palavras-chave (sem acento, minúsculas): dão um bônus fixo à pontuação do
// comando — reforçam a decisão do modelo, mas não decidem sozinhas.
// ----------------------------------------------------------------------------
export const KEYWORDS = {
  pt: {
    criar: ["cobrinha", "snake", "construa", "constroi", "crie", "criar", "do zero", "jogo novo", "novo jogo"],
    comida: ["comida", "comer", "coma", "come ", "maca", "fruta", "aliment", "cresc"],
    parede: ["parede", "muro", "borda", "batida", "limite"],
    portal: ["atravess", "outro lado", "passa direto", "portal", "teletransport", "reaparec", "lado oposto", "dar a volta", "da a volta"],
    corpo: ["propri", "rabo", "cauda", "corpo", "morde", "mesma"],
    placar: ["placar", "ponto", "pontuacao", "score", "recorde"],
    velocidade: ["rapid", "velocidade", "veloz", "acelera", "devagar", "lent", "dificuldade", "dificil"],
    pausa: ["paus", "congela", "espaco"],
    cores: ["cor ", "cores", "colorid", "bonit", "visual", "design", "feio", "aparencia", "profissional", "tema"],
    corrigir: ["bug", "corrij", "corrig", "consert", "arrum", "erro", "problema"],
  },
  en: {
    criar: ["snake game", "build", "create", "from scratch", "new game"],
    comida: ["food", "eat", "apple", "fruit", "grow"],
    parede: ["wall", "border", "edge", "crash"],
    portal: ["through", "wrap", "other side", "opposite side", "teleport", "portal", "no walls"],
    corpo: ["itself", "own body", "tail", "bite", "self"],
    placar: ["score", "point"],
    velocidade: ["fast", "speed", "slow", "harder", "difficult", "quick"],
    pausa: ["pause", "freeze", "stop the game", "space"],
    cores: ["color", "colour", "pretty", "prettier", "visual", "design", "ugly", "look", "theme", "beautiful"],
    corrigir: ["bug", "fix", "error", "problem", "broken", "wrong"],
  },
};

// ----------------------------------------------------------------------------
// "Código" mostrado no painel quando cada comando roda (diff: "+ ", "- ", "  ").
// É ilustrativo — o jogo de verdade roda a partir das opções liga/desliga em
// simulator.js —, mas cada bug do jogo corresponde a um erro visível aqui.
// ----------------------------------------------------------------------------
export const CODE = {
  criar: `+ const GRID = 20, CELL = 20;
+ let snake = [{ x: 10, y: 10 }, { x: 9, y: 10 }, { x: 8, y: 10 }, { x: 7, y: 10 }];
+ let dir = { x: 1, y: 0 };
+
+ document.addEventListener("keydown", (e) => {
+   if (e.key === "ArrowUp")    dir = { x: 0, y: -1 };
+   if (e.key === "ArrowDown")  dir = { x: 0, y: 1 };
+   if (e.key === "ArrowLeft")  dir = { x: -1, y: 0 };
+   if (e.key === "ArrowRight") dir = { x: 1, y: 0 };
+ });
+
+ function step() {
+   const head = { x: snake[0].x + dir.x, y: snake[0].y + dir.y };
+   snake.unshift(head);
+   snake.pop();
+ }
+
+ function draw() {
+   ctx.fillStyle = "lime";
+   for (const part of snake) {
+     ctx.fillRect(part.x * CELL + CELL / 2, part.y * CELL + CELL / 2, CELL, CELL);
+   }
+ }
+
+ setInterval(() => { step(); draw(); }, 150);`,

  comida: `+ let food = randomCell();
+
+ function randomCell() {
+   return { x: Math.floor(Math.random() * GRID), y: Math.floor(Math.random() * GRID) };
+ }
+
  function step() {
    const head = { x: snake[0].x + dir.x, y: snake[0].y + dir.y };
    snake.unshift(head);
-   snake.pop();
+   if (head.x === food.x && head.y === food.y) {
+     food = randomCell();          // comeu: não remove o rabo, então cresce
+   } else {
+     snake.pop();
+   }
  }`,

  parede: `  function step() {
    const head = { x: snake[0].x + dir.x, y: snake[0].y + dir.y };
+   if (head.x < 0 || head.y < 0 || head.x >= GRID - 1 || head.y >= GRID - 1) {
+     return gameOver();
+   }
    snake.unshift(head);`,

  portal: `  function step() {
    const head = { x: snake[0].x + dir.x, y: snake[0].y + dir.y };
+   if (head.x >= GRID) { head.x = 0; head.y += 1; }
+   if (head.x < 0)     head.x = GRID - 1;
+   if (head.y >= GRID) head.y = 0;
+   if (head.y < 0)     head.y = GRID - 1;
    snake.unshift(head);`,

  corpo: `  function step() {
    const head = { x: snake[0].x + dir.x, y: snake[0].y + dir.y };
+   if (snake.some((part) => part.x === head.x && part.y === head.y)) {
+     return gameOver();
+   }
    snake.unshift(head);`,

  placar: `+ let score = 0, best = 0;
+
    if (head.x === food.x && head.y === food.y) {
      food = randomCell();
+     score += 1;
+     best = Math.max(best, score);
+     scoreLabel.textContent = \`\${score} · \${best}\`;
    }

  function restart() {
    snake = [{ x: 10, y: 10 }, { x: 9, y: 10 }, { x: 8, y: 10 }, { x: 7, y: 10 }];
    dir = { x: 1, y: 0 };
  }`,

  velocidade: `+ let interval = 150;
+
    if (head.x === food.x && head.y === food.y) {
      food = randomCell();
+     interval = interval * 0.7;
+     clearInterval(timer);
+     timer = setInterval(tick, interval);
    }`,

  pausa: `+ let paused = false, missedSteps = 0;
+
+ document.addEventListener("keydown", (e) => {
+   if (e.key === " ") paused = !paused;
+ });
+
  function tick() {
+   if (paused) { missedSteps++; return; }
+   for (; missedSteps > 0; missedSteps--) step();   // "recupera" o tempo parado
    step();
    draw();
  }`,

  cores: `  function draw() {
-   ctx.fillStyle = "lime";
-   for (const part of snake) {
-     ctx.fillRect(part.x * CELL + CELL / 2, part.y * CELL + CELL / 2, CELL, CELL);
-   }
+   drawGrid("#171C27");
+   snake.forEach((part, i) => {
+     ctx.fillStyle = mix("#4FD1C5", "#2C7A7B", i / snake.length);
+     roundRect(part.x * CELL + 1, part.y * CELL + 1, CELL - 2, CELL - 2, 6);
+   });
+   drawEyes(snake[0], dir);
+   ctx.shadowColor = ctx.fillStyle = "#F0A868";
+   ctx.shadowBlur = 12;
+   circle(food.x * CELL + CELL / 2, food.y * CELL + CELL / 2, CELL / 2.6);
  }`,
};

// Diff de correção de cada bug (usado pelo comando "corrigir").
export const FIX_CODE = {
  trail: `  function draw() {
+   ctx.clearRect(0, 0, canvas.width, canvas.height);   // apaga o quadro anterior`,
  offset: `-     ctx.fillRect(part.x * CELL + CELL / 2, part.y * CELL + CELL / 2, CELL, CELL);
+     ctx.fillRect(part.x * CELL, part.y * CELL, CELL, CELL);`,
  reverse: `  document.addEventListener("keydown", (e) => {
+   const next = KEYS[e.key];
+   if (!next || (next.x === -dir.x && next.y === -dir.y)) return;   // proíbe meia-volta
+   dir = next;`,
  foodInSnake: `  function randomCell() {
-   return { x: Math.floor(Math.random() * GRID), y: Math.floor(Math.random() * GRID) };
+   let cell;
+   do {
+     cell = { x: Math.floor(Math.random() * GRID), y: Math.floor(Math.random() * GRID) };
+   } while (snake.some((part) => part.x === cell.x && part.y === cell.y));
+   return cell;
  }`,
  wallEarly: `-   if (head.x < 0 || head.y < 0 || head.x >= GRID - 1 || head.y >= GRID - 1) {
+   if (head.x < 0 || head.y < 0 || head.x >= GRID || head.y >= GRID) {`,
  portalShift: `-   if (head.x >= GRID) { head.x = 0; head.y += 1; }
+   if (head.x >= GRID) head.x = 0;`,
  scoreSticky: `  function restart() {
    snake = [{ x: 10, y: 10 }, { x: 9, y: 10 }, { x: 8, y: 10 }, { x: 7, y: 10 }];
    dir = { x: 1, y: 0 };
+   score = 0;`,
  turbo: `-     interval = interval * 0.7;
+     interval = Math.max(70, interval * 0.95);`,
  pauseJump: `-   if (paused) { missedSteps++; return; }
-   for (; missedSteps > 0; missedSteps--) step();   // "recupera" o tempo parado
+   if (paused) return;`,
};

// ----------------------------------------------------------------------------
// Textos da interface do simulador
// ----------------------------------------------------------------------------
export const SIM_TEXT = {
  pt: {
    intro: "Oi! Eu sou o modelo deste site. Não sei programar de verdade — mas consigo entender qual \"peça pronta\" você está pedindo e encaixá-la no jogo. Comece pedindo um jogo da cobrinha!",
    placeholder: "Peça algo para o modelo…",
    send: "Enviar",
    bankIdle: "O modelo vai ler 100 exemplos de pedidos antes de conversar.",
    bankLoading: (n, total) => `🧠 O modelo está lendo os exemplos de pedidos… ${n}/${total}`,
    bankReady: "Pronto — pode pedir!",
    thinking: "Lendo seu pedido…",
    hood: "Por baixo do capô",
    hoodTokens: "1 · Seu pedido virou estes tokens:",
    hoodScores: "2 · O quanto ele se parece com os exemplos de cada comando:",
    hoodNearest: (ex) => `O exemplo mais parecido foi: «${ex}»`,
    hoodKeyword: (cmds) => `Palavra-chave reforçou: ${cmds}`,
    hoodCall: "3 · Comando que o modelo disparou:",
    hoodRule: ({ execMin, margin, askMin }) => {
      const f = (x) => x.toFixed(2).replace(".", ",");
      return execMin > askMin
        ? `Regra: executa se a pontuação for ≥ ${f(execMin)} e ficar ≥ ${f(margin)} à frente da segunda; entre ${f(askMin)} e ${f(execMin)} (ou empate), pergunta; abaixo de ${f(askMin)}, recusa.`
        : `Regra: executa se a pontuação for ≥ ${f(execMin)} e ficar ≥ ${f(margin)} à frente da segunda; se empatar com a segunda, pergunta; abaixo de ${f(execMin)}, recusa.`;
    },
    hoodSplit: (parts) => `Pedido dividido em ${parts} partes — cada uma foi analisada separadamente.`,
    hoodNone: "nenhum (pontuação baixa demais)",
    hoodUnsure: "nenhum ainda — ficou em dúvida entre os primeiros",
    num: (x) => x.toFixed(2).replace(".", ","),
    askTie: (a, b, part) => `${part ? `Sobre «${part}», fiquei` : "Fiquei"} dividido entre ${a} e ${b}. Qual dos dois — ou foi outra coisa?`,
    askLow: (a, part) => `${part ? `Sobre «${part}»: achei` : "Achei"} parecido com ${a}, mas não tenho certeza. Foi uma destas?`,
    starLegend: "★ = o pedido tem uma palavra-chave desse comando, o que dá um empurrãozinho na pontuação.",
    insight: {
      simple: {
        clear: ({ top, second }) => `Decisão tranquila: seu pedido ficou bem mais parecido com os exemplos de ${top} do que com os de qualquer outro comando — o segundo colocado, ${second}, ficou bem atrás.`,
        close: ({ top, second }) => `Foi por pouco: ${top} venceu, mas ${second} chegou perto — seu pedido tinha um pouco dos dois.`,
        split: ({ parts }) => `Seu pedido tinha ${parts.length} partes, e o modelo analisou cada uma separadamente: ${parts.map((p) => `«${p.text}» virou ${p.label}`).join("; ")}.`,
        tie: ({ top, second }) => `${top} e ${second} ficaram quase empatados — em vez de chutar, o modelo preferiu perguntar. Assistentes bem feitos fazem isso quando não têm certeza.`,
        low: ({ top }) => `O pedido lembrou ${top}, mas não o suficiente pra ter certeza — por isso a pergunta.`,
        reject: ({ top }) => `Nada no banco de exemplos se pareceu muito com esse pedido (o mais próximo, ${top}, ficou longe), então o modelo recusou em vez de inventar.`,
      },
      technical: {
        clear: ({ top, s1, second, s2, gap, margin }) => `${top}: ${s1} vs. ${second}: ${s2} — folga de ${gap} (mínimo exigido: ${margin}). Margem confortável, executado direto.`,
        close: ({ top, s1, second, s2, gap, margin }) => `${top}: ${s1} vs. ${second}: ${s2} — folga de ${gap}, pouco acima do mínimo de ${margin}. Executado, mas por margem estreita.`,
        split: ({ parts }) => `Pedido dividido em ${parts.length} cláusulas (por vírgula/"e"); cada uma passou pelo modelo separadamente: ${parts.map((p) => `${p.label} ${p.score}`).join(", ")}.`,
        tie: ({ top, s1, second, s2, gap, margin }) => `${top}: ${s1} vs. ${second}: ${s2} — folga de ${gap}, abaixo do mínimo de ${margin} → pergunta.`,
        low: ({ top, s1, askMin, execMin }) => `${top}: ${s1} — acima de ${askMin}, mas abaixo do mínimo de ${execMin} pra executar → pergunta.`,
        reject: ({ top, s1, askMin }) => `Maior pontuação: ${top} com ${s1}, abaixo do mínimo de ${askMin} → recusa.`,
      },
    },
    reject: "Isso eu não sei fazer — só sei mexer neste jogo da cobrinha. Tente uma destas:",
    needCreate: "Ainda não existe jogo! Primeiro peça para eu criar o jogo da cobrinha.",
    already: (label) => `Isso já está no jogo (${label}).`,
    alreadyAlt: "Talvez você tenha querido dizer outra coisa — a segunda opção do modelo foi:",
    alreadyCreated: "O jogo já existe! Para recomeçar do zero, clique na v0.1 na linha do tempo.",
    replies: {
      criar: () => "Pronto! Criei a v0.1 do jogo da cobrinha (é o único jogo que eu sei fazer). Clique no jogo e use as setas para jogar. Fiz bem rápido… pode ter uns probleminhas 😅",
      comida: () => "Adicionei a comida 🍎 — quando a cobra come, ela cresce.",
      parede: ({ hadPortal }) => `Agora bater na parede é fim de jogo.${hadPortal ? " Tirei o \"atravessar a parede\", já que as duas coisas não combinam." : ""}`,
      portal: ({ hadWall }) => `Agora a cobra sai de um lado da tela e aparece do outro.${hadWall ? " Tirei a colisão com a parede, já que as duas coisas não combinam." : ""}`,
      corpo: ({ hasReverse }) => `Agora morder o próprio corpo é fim de jogo.${hasReverse ? " (Hmm… experimente apertar a seta contrária à direção da cobra 👀)" : ""}`,
      placar: ({ hasFood }) => `Coloquei o placar e o recorde no canto da tela.${hasFood ? "" : " Mas ainda não tem comida, então o placar vai ficar no zero 😬"}`,
      velocidade: ({ hasFood }) => `A cada comida, a cobra fica mais rápida.${hasFood ? "" : " Só que, sem comida, nada vai mudar — peça a comida!"}`,
      pausa: () => "Aperte espaço (ou o botão ⏸) para pausar.",
      cores: ({ fixedOffset }) => `Dei um tapa no visual: cores, grade e olhinhos na cobra.${fixedOffset ? " Ao reescrever o desenho, o desalinhamento sumiu de brinde." : ""}`,
    },
    fixNone: "Não encontrei nenhum bug conhecido 🎉",
    fixDone: (n) => `Corrigi ${n} bug${n > 1 ? "s" : ""}:`,
    commandLabels: {
      criar: "🐍 Criar o jogo",
      comida: "🍎 Comida",
      parede: "🧱 Morrer na parede",
      portal: "🌀 Atravessar a parede",
      corpo: "🪢 Morrer no próprio corpo",
      placar: "🏆 Placar",
      velocidade: "⚡ Velocidade",
      pausa: "⏸ Pausa",
      cores: "🎨 Visual",
      corrigir: "🔧 Corrigir bugs",
    },
    suggestions: {
      criar: "Construa um jogo da cobrinha",
      comida: "Agora acrescente a comida",
      parede: "Faça ela morrer quando bater na parede",
      portal: "Deixe ela atravessar as paredes",
      corpo: "Ela deve morrer se morder o próprio corpo",
      placar: "Mostre a pontuação",
      velocidade: "Deixe a cobra mais rápida com o tempo",
      pausa: "Quero poder pausar",
      cores: "Deixe o jogo mais bonito",
      corrigir: "Corrija os bugs",
    },
    bugs: {
      trail: "a cobra deixava rastro (a tela nunca era apagada)",
      offset: "a cobra era desenhada meia casa fora do lugar",
      reverse: "dava para virar 180° e andar por cima do próprio corpo",
      foodInSnake: "a comida podia nascer escondida embaixo da cobra",
      wallEarly: "a cobra morria uma casa antes das paredes da direita e de baixo",
      portalShift: "ao atravessar a parede da direita, a cobra descia uma linha",
      scoreSticky: "o placar não zerava ao recomeçar",
      turbo: "a velocidade aumentava rápido demais (30% por comida)",
      pauseJump: "depois da pausa, a cobra dava vários passos de uma vez",
    },
    game: {
      empty: "O jogo aparece aqui",
      clickToPlay: "Clique aqui para jogar",
      pressArrow: "Use as setas (ou W A S D) para começar",
      gameOver: "Fim de jogo",
      restartHint: "Enter ou toque para recomeçar",
      paused: "Pausado",
      fled: "🙈 A cobra fugiu da tela! (ninguém disse a ela que existem bordas)",
      score: "Pontos",
      best: "Recorde",
    },
    versions: "Versões",
    codeTitle: "game.js",
    codeEmpty: "O código aparece aqui quando o modelo começar a \"programar\".",
    codeFixTitle: "correções",
  },
  en: {
    intro: "Hi! I'm this site's model. I can't actually code — but I can figure out which \"ready-made piece\" you're asking for and plug it into the game. Start by asking for a snake game!",
    placeholder: "Ask the model for something…",
    send: "Send",
    bankIdle: "The model will read 100 example requests before chatting.",
    bankLoading: (n, total) => `🧠 The model is reading the example requests… ${n}/${total}`,
    bankReady: "Ready — ask away!",
    thinking: "Reading your request…",
    hood: "Under the hood",
    hoodTokens: "1 · Your request became these tokens:",
    hoodScores: "2 · How similar it is to each command's examples:",
    hoodNearest: (ex) => `The closest example was: «${ex}»`,
    hoodKeyword: (cmds) => `Keyword boost: ${cmds}`,
    hoodCall: "3 · Command the model fired:",
    hoodRule: ({ execMin, margin, askMin }) => {
      const f = (x) => x.toFixed(2);
      return execMin > askMin
        ? `Rule: run it if the score is ≥ ${f(execMin)} and ≥ ${f(margin)} ahead of the runner-up; between ${f(askMin)} and ${f(execMin)} (or a near tie), ask; below ${f(askMin)}, refuse.`
        : `Rule: run it if the score is ≥ ${f(execMin)} and ≥ ${f(margin)} ahead of the runner-up; on a near tie, ask; below ${f(execMin)}, refuse.`;
    },
    hoodSplit: (parts) => `Request split into ${parts} parts — each one was analyzed separately.`,
    hoodNone: "none (score too low)",
    hoodUnsure: "none yet — torn between the top options",
    num: (x) => x.toFixed(2),
    askTie: (a, b, part) => `${part ? `About «${part}», I was` : "I was"} torn between ${a} and ${b}. Which one — or was it something else?`,
    askLow: (a, part) => `${part ? `About «${part}»: it` : "It"} sounded like ${a}, but I'm not sure. Was it one of these?`,
    starLegend: "★ = the request has one of this command's keywords, which gives its score a little nudge.",
    insight: {
      simple: {
        clear: ({ top, second }) => `Easy call: your request was much closer to the examples for ${top} than to any other command's — the runner-up, ${second}, trailed far behind.`,
        close: ({ top, second }) => `That was close: ${top} won, but ${second} came near — your request had a bit of both.`,
        split: ({ parts }) => `Your request had ${parts.length} parts, and the model analyzed each one separately: ${parts.map((p) => `«${p.text}» became ${p.label}`).join("; ")}.`,
        tie: ({ top, second }) => `${top} and ${second} were nearly tied — instead of guessing, the model chose to ask. Well-built assistants do that when they're unsure.`,
        low: ({ top }) => `The request reminded it of ${top}, but not enough to be sure — hence the question.`,
        reject: ({ top }) => `Nothing in the example bank looked much like this request (the closest, ${top}, was far off), so the model refused instead of making something up.`,
      },
      technical: {
        clear: ({ top, s1, second, s2, gap, margin }) => `${top}: ${s1} vs. ${second}: ${s2} — gap of ${gap} (required minimum: ${margin}). Comfortable margin, run directly.`,
        close: ({ top, s1, second, s2, gap, margin }) => `${top}: ${s1} vs. ${second}: ${s2} — gap of ${gap}, just above the ${margin} minimum. Run, but by a narrow margin.`,
        split: ({ parts }) => `Request split into ${parts.length} clauses (on commas/"and"); each went through the model separately: ${parts.map((p) => `${p.label} ${p.score}`).join(", ")}.`,
        tie: ({ top, s1, second, s2, gap, margin }) => `${top}: ${s1} vs. ${second}: ${s2} — gap of ${gap}, below the ${margin} minimum → ask.`,
        low: ({ top, s1, askMin, execMin }) => `${top}: ${s1} — above ${askMin} but below the ${execMin} needed to run → ask.`,
        reject: ({ top, s1, askMin }) => `Top score: ${top} at ${s1}, below the ${askMin} minimum → refuse.`,
      },
    },
    reject: "I can't do that — I only know how to change this snake game. Try one of these:",
    needCreate: "There's no game yet! First ask me to create the snake game.",
    already: (label) => `That's already in the game (${label}).`,
    alreadyAlt: "Maybe you meant something else — the model's runner-up was:",
    alreadyCreated: "The game already exists! To start over, click v0.1 on the timeline.",
    replies: {
      criar: () => "Done! I made v0.1 of the snake game (it's the only game I know how to make). Click the game and use the arrow keys to play. I made it really fast… there may be a few issues 😅",
      comida: () => "Added food 🍎 — when the snake eats, it grows.",
      parede: ({ hadPortal }) => `Now hitting a wall is game over.${hadPortal ? " I removed \"go through walls\", since the two don't mix." : ""}`,
      portal: ({ hadWall }) => `Now the snake leaves one side of the screen and comes back on the other.${hadWall ? " I removed wall collision, since the two don't mix." : ""}`,
      corpo: ({ hasReverse }) => `Now biting its own body is game over.${hasReverse ? " (Hmm… try pressing the arrow opposite to the snake's direction 👀)" : ""}`,
      placar: ({ hasFood }) => `Added the score and high score in the corner.${hasFood ? "" : " But there's no food yet, so the score will stay at zero 😬"}`,
      velocidade: ({ hasFood }) => `Every bite makes the snake faster.${hasFood ? "" : " But without food nothing will change — ask for food!"}`,
      pausa: () => "Press space (or the ⏸ button) to pause.",
      cores: ({ fixedOffset }) => `Gave it a makeover: colors, grid and little eyes on the snake.${fixedOffset ? " Rewriting the drawing code fixed the misalignment as a bonus." : ""}`,
    },
    fixNone: "I couldn't find any known bugs 🎉",
    fixDone: (n) => `Fixed ${n} bug${n > 1 ? "s" : ""}:`,
    commandLabels: {
      criar: "🐍 Create the game",
      comida: "🍎 Food",
      parede: "🧱 Die at the wall",
      portal: "🌀 Go through walls",
      corpo: "🪢 Die on own body",
      placar: "🏆 Score",
      velocidade: "⚡ Speed",
      pausa: "⏸ Pause",
      cores: "🎨 Visuals",
      corrigir: "🔧 Fix bugs",
    },
    suggestions: {
      criar: "Build a snake game",
      comida: "Now add the food",
      parede: "Make it die when it hits the wall",
      portal: "Let it go through the walls",
      corpo: "It should die if it bites its own body",
      placar: "Show the score",
      velocidade: "Make the snake speed up over time",
      pausa: "I want to be able to pause",
      cores: "Make the game prettier",
      corrigir: "Fix the bugs",
    },
    bugs: {
      trail: "the snake left a trail (the screen was never cleared)",
      offset: "the snake was drawn half a cell out of place",
      reverse: "you could turn 180° and walk over your own body",
      foodInSnake: "food could spawn hidden under the snake",
      wallEarly: "the snake died one cell before the right and bottom walls",
      portalShift: "wrapping through the right wall moved the snake down a row",
      scoreSticky: "the score didn't reset on restart",
      turbo: "speed ramped up way too fast (30% per bite)",
      pauseJump: "after pausing, the snake took several steps at once",
    },
    game: {
      empty: "The game shows up here",
      clickToPlay: "Click here to play",
      pressArrow: "Use the arrow keys (or W A S D) to start",
      gameOver: "Game over",
      restartHint: "Enter or tap to restart",
      paused: "Paused",
      fled: "🙈 The snake ran off the screen! (nobody told it there are edges)",
      score: "Score",
      best: "Best",
    },
    versions: "Versions",
    codeTitle: "game.js",
    codeEmpty: "The code shows up here once the model starts \"coding\".",
    codeFixTitle: "fixes",
  },
};
