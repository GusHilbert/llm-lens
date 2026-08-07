"""
model_utils.py

Carrega um modelo de linguagem pequeno (GPT-2 ou similar) e expõe funções que
extraem os "internals" do modelo em cada etapa: tokenização, embeddings,
pesos de atenção por camada/cabeça, e distribuição de probabilidade da
próxima palavra.

Projetado para modelos pequenos (GPT-2 small, DistilGPT-2, Qwen2.5-0.5B) para
que toda a análise rode rápido em CPU ou GPU modesta.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

# Nome do modelo padrão. Pode ser trocado por qualquer causal LM pequeno
# disponível no Hugging Face Hub (ex: "distilgpt2", "Qwen/Qwen2.5-0.5B").
# Usamos a versão em português (fine-tuning do GPT-2 small na Wikipédia PT),
# validada no notebook de teste de viabilidade no Colab.
DEFAULT_MODEL_NAME = "pierreguillou/gpt2-small-portuguese"


class LLMLens:
    """Encapsula um modelo causal + tokenizer e expõe introspecção passo a passo."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # GPT-2 não tem pad token por padrão; usamos o eos como pad.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_attentions=True,
            output_hidden_states=True,
        ).to(self.device)
        self.model.eval()

        self.num_layers = self.model.config.n_layer if hasattr(self.model.config, "n_layer") else self.model.config.num_hidden_layers
        self.num_heads = self.model.config.n_head if hasattr(self.model.config, "n_head") else self.model.config.num_attention_heads

    # ------------------------------------------------------------------ #
    # 1. Tokenização
    # ------------------------------------------------------------------ #
    def tokenize(self, text: str) -> list[dict]:
        """Retorna a lista de tokens com seus IDs e texto decodificado individualmente."""
        ids = self.tokenizer.encode(text)
        tokens = []
        for tid in ids:
            piece = self.tokenizer.decode([tid])
            tokens.append({"id": tid, "text": piece})
        return tokens

    # ------------------------------------------------------------------ #
    # 2. Embeddings (projetados em 2D via PCA para visualização)
    # ------------------------------------------------------------------ #
    def embeddings_2d(self, text: str) -> list[dict]:
        tokens = self.tokenize(text)
        ids = torch.tensor([[t["id"] for t in tokens]]).to(self.device)

        with torch.no_grad():
            # hidden_states[0] = embeddings de entrada (antes de qualquer camada Transformer)
            outputs = self.model(ids, output_hidden_states=True)
            raw_embeddings = outputs.hidden_states[0][0].cpu().numpy()  # (seq_len, hidden_dim)

        # PCA precisa de pelo menos 2 amostras; se só houver 1 token, duplicamos.
        data = raw_embeddings if len(raw_embeddings) > 1 else np.vstack([raw_embeddings, raw_embeddings])
        n_components = min(2, data.shape[0], data.shape[1])
        coords = PCA(n_components=n_components).fit_transform(data)

        result = []
        for i, tok in enumerate(tokens):
            x, y = (coords[i][0], coords[i][1]) if n_components == 2 else (coords[i][0], 0.0)
            result.append({"text": tok["text"], "x": float(x), "y": float(y)})
        return result

    # ------------------------------------------------------------------ #
    # 3. Atenção por camada e cabeça
    # ------------------------------------------------------------------ #
    def attention_maps(self, text: str) -> dict:
        tokens = self.tokenize(text)
        ids = torch.tensor([[t["id"] for t in tokens]]).to(self.device)

        with torch.no_grad():
            outputs = self.model(ids, output_attentions=True)

        # outputs.attentions: tupla de (num_layers) tensores, cada um
        # (batch, num_heads, seq_len, seq_len)
        layers = []
        for layer_idx, layer_attn in enumerate(outputs.attentions):
            heads = []
            for head_idx in range(layer_attn.shape[1]):
                matrix = layer_attn[0, head_idx].cpu().numpy().tolist()
                heads.append(matrix)
            layers.append(heads)

        return {
            "tokens": [t["text"] for t in tokens],
            "num_layers": len(layers),
            "num_heads": self.num_heads,
            "attention": layers,  # layers[layer][head][from_token][to_token]
        }

    # ------------------------------------------------------------------ #
    # 4. Próximo token: distribuição de probabilidade (top-k)
    # ------------------------------------------------------------------ #
    def next_token_distribution(self, text: str, top_k: int = 10, temperature: float = 1.0) -> list[dict]:
        ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self.model(ids).logits[0, -1, :]  # logits do último token
            probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)

        top_probs, top_ids = torch.topk(probs, top_k)
        result = []
        for prob, tid in zip(top_probs.tolist(), top_ids.tolist()):
            result.append({
                "token": self.tokenizer.decode([tid]),
                "id": tid,
                "probability": prob,
            })
        return result

    # ------------------------------------------------------------------ #
    # 5. Geração passo a passo (permite escolher manualmente um token)
    # ------------------------------------------------------------------ #
    def generate_step(self, text: str, chosen_token_id: int | None = None,
                       top_k: int = 10, temperature: float = 1.0) -> dict:
        """
        Retorna a distribuição de probabilidade do próximo token e, se
        `chosen_token_id` for fornecido, o novo texto após adicionar esse token.
        Isso permite que o front-end deixe o usuário "dirigir" a geração manualmente.
        """
        distribution = self.next_token_distribution(text, top_k=top_k, temperature=temperature)

        new_text = None
        if chosen_token_id is not None:
            new_text = text + self.tokenizer.decode([chosen_token_id])

        return {"distribution": distribution, "new_text": new_text}


# Instância global reutilizada entre requisições (evita recarregar o modelo a cada chamada).
_lens_instance: LLMLens | None = None


def get_lens(model_name: str = DEFAULT_MODEL_NAME) -> LLMLens:
    global _lens_instance
    if _lens_instance is None or _lens_instance.model_name != model_name:
        _lens_instance = LLMLens(model_name=model_name)
    return _lens_instance
