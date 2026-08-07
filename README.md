# 🧠 LLM Lens — Como um LLM Pensa

Uma ferramenta interativa de código aberto que abre a "caixa-preta" de um modelo de linguagem
e mostra, passo a passo, o que acontece entre você digitar um prompt e o modelo gerar uma resposta.

> Projeto desenvolvido como Trabalho de Conclusão de Curso da Pós-Graduação em
> Inteligência Artificial Generativa & LLMs (Laboratório ICA — PUC-Rio).

## O que ele mostra

1. **Tokenização** — como o texto é quebrado em tokens (subpalavras), com cores e IDs.
2. **Embeddings** — projeção 2D (PCA/t-SNE) dos vetores de cada token, mostrando proximidade semântica.
3. **Atenção** — mapa de calor interativo mostrando para onde cada cabeça de atenção "olha" em
   cada camada do Transformer, ao processar cada palavra.
4. **Geração passo a passo** — a distribuição de probabilidade completa (top-k) antes de cada
   token ser escolhido, com controle de temperature/top-p, e a opção de escolher manualmente
   um token diferente do mais provável para ver como a frase muda.

## Por que isso importa

A maioria das ferramentas educacionais sobre LLMs mostra o modelo como uma caixa preta:
texto entra, texto sai. O objetivo aqui é o oposto — tornar visível o mecanismo interno
(tokenização → embeddings → atenção → predição), usando um modelo pequeno o suficiente para
rodar localmente e ser totalmente instrumentado.

## Stack

- **Frontend**: HTML/CSS/JS vanilla + [Transformers.js](https://huggingface.co/docs/transformers.js) —
  o modelo roda **inteiramente no navegador** de quem acessa o site (via ONNX Runtime/WASM),
  sem nenhum servidor por trás.
- **Backend (`backend/`)**: mantido no repositório como referência/prototipagem em Python —
  útil para desenvolvimento no Colab e para explicar os internals no TCC, mas **não é usado
  em produção**. A versão publicada roda 100% client-side.
- **Modelo**: `gpt2-small-portuguese` (GPT-2 small fine-tunado em português), convertido para
  ONNX e publicado no Hugging Face Hub.

## Por que 100% no navegador (sem backend hospedado)?

Testamos duas alternativas de hospedagem de backend (Google Colab e Hugging Face Spaces) e
ambas têm limitações reais para um site público e permanente: Colab não sustenta sessões
públicas 24/7 no tier gratuito, e o Hugging Face Spaces passou a exigir plano pago para
SDKs com computação (Docker/Gradio) — só o SDK estático continua gratuito. Rodando o modelo
inteiramente no navegador via Transformers.js, o site fica 100% estático (compatível com
GitHub Pages), gratuito para sempre, e escala para qualquer número de visitantes sem
nenhum custo ou gargalo de servidor compartilhado.

## Como rodar localmente (desenvolvimento)

```bash
# 1. Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r ../requirements.txt
uvicorn app:app --reload --port 8000

# 2. Frontend
cd ../frontend
# Abra index.html no navegador, ou sirva com:
python -m http.server 5500
```

Na primeira execução, o backend vai baixar os pesos do modelo do Hugging Face
(precisa de internet nesse primeiro momento; depois roda 100% offline).

## Deploy em produção (site público, sempre no ar, 100% de graça)

Arquitetura final: **tudo estático, tudo no GitHub Pages**. Sem backend pra manter no ar.

### 1. Converter o modelo para ONNX e publicar no Hugging Face Hub

Use o notebook `colab_teste_viabilidade.ipynb` (seção "Converter para ONNX e publicar no
Hugging Face Hub") — roda no Colab gratuito, gera os pesos ONNX e sobe pra um repositório
de modelo público no Hub (isso é diferente de um Space; repositórios de modelo público
são gratuitos e ilimitados).

### 2. Apontar o frontend pro modelo publicado

No arquivo `frontend/script.js`, troque a constante `MODEL_ID` pelo ID gerado no passo
anterior (ex: `"seu-usuario/gpt2-small-portuguese-onnx"`).

### 3. Publicar o frontend no GitHub Pages

1. No repositório do GitHub (este projeto), vá em **Settings → Pages**.
2. Em "Source", selecione a branch principal e a pasta `/frontend`.
3. Salve — o GitHub te dá uma URL tipo `https://seu-usuario.github.io/llm-lens/`.

Pronto: site público, gratuito para sempre, sem servidor, escalando pra qualquer
quantidade de visitantes sem gargalo — cada um roda o modelo na própria máquina.

## Roadmap acadêmico (seções sugeridas do TCC)

- [x] MVP: tokenização + atenção + geração passo a passo
- [ ] Comparação entre tamanhos de modelo (0.5B vs 7B) no mesmo prompt
- [ ] Estudo de caso com frases ambíguas em português (resolução de correferência)
- [ ] (Avançado) Treinar um mini-modelo do zero (estilo nanoGPT) e comparar a atenção
      entre época 1 e época N do treinamento
- [ ] Avaliação de usabilidade: teste A/B medindo se a ferramenta melhora a compreensão
      de conceitos de Transformers em usuários iniciantes

## Estrutura do repositório

```
llm-visualizer/
├── backend/
│   ├── app.py            # API FastAPI
│   └── model_utils.py    # Carregamento do modelo e extração de internals
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── requirements.txt
├── LICENSE
└── README.md
```

## Licença

MIT — veja [LICENSE](LICENSE). Contribuições são bem-vindas.

## Autor

Gustav — desenvolvido no contexto da Pós-Graduação em IA Generativa & LLMs (PUC-Rio / ICA).
