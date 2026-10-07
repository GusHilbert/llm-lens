"""
Publica o modelo V2 em inglês (quantizado) num repositório novo no Hugging Face.

Não lida com o token diretamente -- usa a credencial já salva localmente
por `hf auth login`.
"""

from huggingface_hub import HfApi

HF_USERNAME = "GusHil"
MODEL_REPO_ID_EN = f"{HF_USERNAME}/gpt2-english-onnx-v2"
QUANT_DIR = "onnx_model_v2_en_quant"

api = HfApi()
api.create_repo(repo_id=MODEL_REPO_ID_EN, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path=f"./{QUANT_DIR}",
    repo_id=MODEL_REPO_ID_EN,
    repo_type="model",
)

print(f"\nModelo EN publicado em: https://huggingface.co/{MODEL_REPO_ID_EN}")
