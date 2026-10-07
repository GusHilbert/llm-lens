"""
Exportação V2 do modelo (GPorTuguese-2 -> ONNX) — versão local, sem Colab.

Diferença pra V1: além de logits + atenção, o grafo ONNX agora também expõe
os hidden states de cada camada (embedding + saída de cada bloco) como
saídas nomeadas. Isso não é computação extra — o modelo já calcula esses
vetores internamente a cada forward pass, só não os devolvia antes.

Roda em dois checkpoints de validação antes de deixar pronto pra publicar:
  1. fp32 (ONNX) vs PyTorch — confirma que adicionar hidden_states não
     quebrou logits/atenção que já funcionavam.
  2. fp32 vs int8 (quantizado) — confirma que a quantização não degradou
     a qualidade de forma perceptível (mesmo token no topo, top-5 coerente).

Publicação no Hugging Face é uma etapa separada (precisa de login), não é
feita por este script.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MODEL_NAME = "pierreguillou/gpt2-small-portuguese"
OUTPUT_DIR = "onnx_model_v2"
QUANT_DIR = "onnx_model_v2_quant"

TEST_SENTENCES = [
    "O rato que o gato perseguiu fugiu",
    "A inteligência artificial no Brasil",
    "Ela disse que ele nunca voltaria",
]


class GPT2WithAttentionsAndHidden(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
        )
        return (outputs.logits,) + tuple(outputs.attentions) + tuple(outputs.hidden_states)


def export(base_model, tokenizer):
    wrapped = GPT2WithAttentionsAndHidden(base_model)
    dummy = tokenizer("Isto é um texto de exemplo para exportação", return_tensors="pt")

    num_layers = base_model.config.n_layer
    attention_names = [f"attentions.{i}" for i in range(num_layers)]
    hidden_state_names = [f"hidden_states.{i}" for i in range(num_layers + 1)]
    output_names = ["logits"] + attention_names + hidden_state_names

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }
    for name in attention_names:
        dynamic_axes[name] = {0: "batch_size", 2: "sequence_length", 3: "sequence_length"}
    for name in hidden_state_names:
        dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}

    os.makedirs(f"{OUTPUT_DIR}/onnx", exist_ok=True)

    torch.onnx.export(
        wrapped,
        (dummy["input_ids"], dummy["attention_mask"]),
        f"{OUTPUT_DIR}/onnx/model.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        dynamo=False,
    )

    tokenizer.save_pretrained(OUTPUT_DIR)
    base_model.config.save_pretrained(OUTPUT_DIR)

    print(f"Exportação V2 concluída em {OUTPUT_DIR}/onnx/model.onnx")
    print(f"Total de saídas: {len(output_names)} ({len(attention_names)} camadas de atenção, {len(hidden_state_names)} hidden states)")
    return output_names, num_layers


def checkpoint1_parity(base_model, tokenizer, output_names, num_layers):
    import onnxruntime as ort

    print("\n--- Checkpoint 1: paridade fp32 (ONNX vs PyTorch) ---")
    session = ort.InferenceSession(f"{OUTPUT_DIR}/onnx/model.onnx")

    # Recarrega uma cópia fresca do modelo pra comparação -- o objeto que
    # acabou de passar pelo tracing do torch.onnx.export fica num estado
    # sutilmente diferente (a implementação de atenção escolhida internamente
    # muda) e produz resultados levemente diferentes do mesmo modelo "limpo".
    fresh_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    fresh_model.eval()

    sample = tokenizer("O rato que o gato perseguiu fugiu", return_tensors="pt")
    ort_inputs = {
        "input_ids": sample["input_ids"].numpy(),
        "attention_mask": sample["attention_mask"].numpy(),
    }
    ort_outputs = session.run(output_names, ort_inputs)
    ort_by_name = dict(zip(output_names, ort_outputs))

    with torch.no_grad():
        torch_outputs = fresh_model(
            input_ids=sample["input_ids"],
            attention_mask=sample["attention_mask"],
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
        )

    logits_diff = np.abs(ort_by_name["logits"] - torch_outputs.logits.numpy()).max()
    print(f"Maior diferença absoluta nos logits: {logits_diff:.6f}")
    assert logits_diff < 1e-3, "Logits divergiram mais do que o esperado."

    attn0_diff = np.abs(ort_by_name["attentions.0"] - torch_outputs.attentions[0].numpy()).max()
    print(f"Maior diferença absoluta na atenção (camada 0): {attn0_diff:.6f}")
    assert attn0_diff < 1e-3, "Atenção divergiu mais do que o esperado."

    for i, hs in enumerate(torch_outputs.hidden_states):
        onnx_hs = ort_by_name[f"hidden_states.{i}"]
        assert onnx_hs.shape == tuple(hs.shape), f"Formato inesperado em hidden_states.{i}"
    print(f"Formato de cada hidden state: {ort_by_name['hidden_states.0'].shape}  (batch, seq_len, hidden_size)")
    print(f"Total de hidden states: {len(torch_outputs.hidden_states)} (embedding + {num_layers} camadas)")
    print("Checkpoint 1 OK.")
    return session


def quantize(tokenizer, base_model):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print("\n--- Quantização dinâmica (int8) ---")
    os.makedirs(f"{QUANT_DIR}/onnx", exist_ok=True)

    quantize_dynamic(
        model_input=f"{OUTPUT_DIR}/onnx/model.onnx",
        model_output=f"{QUANT_DIR}/onnx/model_quantized.onnx",
        weight_type=QuantType.QInt8,
    )

    # O tokenizer/config também precisam ir junto -- sem eles o Transformers.js
    # não consegue carregar o modelo no navegador (só o .onnx não basta).
    tokenizer.save_pretrained(QUANT_DIR)
    base_model.config.save_pretrained(QUANT_DIR)

    fp32_size = os.path.getsize(f"{OUTPUT_DIR}/onnx/model.onnx") / 1e6
    int8_size = os.path.getsize(f"{QUANT_DIR}/onnx/model_quantized.onnx") / 1e6
    print(f"Tamanho fp32: {fp32_size:.1f} MB")
    print(f"Tamanho int8: {int8_size:.1f} MB ({int8_size / fp32_size * 100:.0f}% do original)")


def checkpoint2_quality(tokenizer, session_fp32):
    import onnxruntime as ort

    print("\n--- Checkpoint 2: qualidade fp32 vs int8 ---")
    session_quant = ort.InferenceSession(f"{QUANT_DIR}/onnx/model_quantized.onnx")

    def top5(session, text):
        enc = tokenizer(text, return_tensors="pt")
        inputs = {
            "input_ids": enc["input_ids"].numpy(),
            "attention_mask": enc["attention_mask"].numpy(),
        }
        logits = session.run(["logits"], inputs)[0]
        last_logits = logits[0, -1, :]
        probs = np.exp(last_logits - last_logits.max())
        probs = probs / probs.sum()
        top_ids = np.argsort(probs)[::-1][:5]
        return [(tokenizer.decode([tid]), float(probs[tid])) for tid in top_ids]

    all_match = True
    for sentence in TEST_SENTENCES:
        fp32_top5 = top5(session_fp32, sentence)
        int8_top5 = top5(session_quant, sentence)
        same_top1 = fp32_top5[0][0] == int8_top5[0][0]
        all_match = all_match and same_top1
        print(f"\nFrase: {sentence!r}")
        print(f"  fp32 top-1: {fp32_top5[0][0]!r} ({fp32_top5[0][1]*100:.1f}%)")
        print(f"  int8 top-1: {int8_top5[0][0]!r} ({int8_top5[0][1]*100:.1f}%)  {'OK, igual' if same_top1 else '!! DIFERENTE, revisar'}")
        print(f"  fp32 top-5: {[(t, round(p*100,1)) for t, p in fp32_top5]}")
        print(f"  int8 top-5: {[(t, round(p*100,1)) for t, p in int8_top5]}")

    print(f"\nCheckpoint 2 {'OK' if all_match else 'ATENÇÃO: nem todos os top-1 bateram, revisar antes de publicar'}.")


def main():
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    print(f"Carregando {MODEL_NAME}...")
    base_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    base_model.eval()
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)

    output_names, num_layers = export(base_model, tokenizer)
    session_fp32 = checkpoint1_parity(base_model, tokenizer, output_names, num_layers)
    quantize(tokenizer, base_model)
    checkpoint2_quality(tokenizer, session_fp32)

    print("\nTudo pronto. Próximo passo é publicar (etapa separada, precisa de login no Hugging Face).")


if __name__ == "__main__":
    main()
