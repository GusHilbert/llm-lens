"""
Publica o modelo V2 (quantizado) num repositório NOVO no Hugging Face,
separado do repositório da V1 que o site em produção já usa -- assim nada
quebra pro público até o frontend estar pronto pra usar hidden states
(arcos de atenção, camadas ocultas) e a gente trocar o MODEL_IDS de propósito.

Não lida com o token diretamente -- usa a credencial já salva localmente
por `hf auth login`.
"""

from huggingface_hub import HfApi

HF_USERNAME = "GusHil"
MODEL_REPO_ID_V2 = f"{HF_USERNAME}/gpt2-small-portuguese-onnx-v2"
QUANT_DIR = "onnx_model_v2_quant"

api = HfApi()
api.create_repo(repo_id=MODEL_REPO_ID_V2, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path=f"./{QUANT_DIR}",
    repo_id=MODEL_REPO_ID_V2,
    repo_type="model",
)

print(f"\nModelo V2 publicado em: https://huggingface.co/{MODEL_REPO_ID_V2}")
print("Repositório V1 (produção atual) continua intocado.")
print(f"Quando o frontend estiver pronto, trocar MODEL_IDS no script.js para: '{MODEL_REPO_ID_V2}'")
