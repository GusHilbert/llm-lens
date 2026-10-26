# Dados do Excel: RAG, fine-tuning e avaliação

Material das seções 8 (RAG) e 9 (fine-tuning) do LLM Lens e do experimento que compara as duas técnicas.

| Arquivo | Para que serve |
|---|---|
| `frontend/data/fichas-excel.json` | Base de conhecimento: 66 fichas curtas sobre Excel em português do Brasil. São os documentos que o RAG recupera e o texto de treino do fine-tuning. Fica dentro de `frontend/` porque o site carrega o arquivo. |
| `avaliacao/perguntas-teste-excel.json` | 45 perguntas de teste, escritas como paráfrases. **Nunca entram no treino.** |
| `scripts/validar_dados_excel.py` | Confere os dois arquivos. Rode depois de qualquer edição. |

## Formato

**Ficha:** `id`, `categoria`, `titulo` e `texto`, com 30 a 90 palavras para que três fichas caibam no contexto do GPT-2. Campos opcionais:
- `verificar`: detalhe que ainda precisa ser confirmado num Excel em português;
- `cenario: "atualizacao"`: ficha que fica fora do fine-tuning (veja abaixo).

**Pergunta:** `pergunta`, `inicio_resposta`, `resposta`, 3 `distratores` e `fichas_relevantes`, que são as fichas onde está a resposta e servem para medir a busca do RAG.

## Protocolo do experimento

Cada pergunta é uma **múltipla escolha por verossimilhança**, no mesmo estilo do lm-evaluation-harness. Para cada uma das quatro opções (a resposta e os três distratores), monta-se o texto:

```
Pergunta: {pergunta}
Resposta: {inicio_resposta} {opcao}
```

Em seguida, soma-se a log-probabilidade que o modelo dá aos tokens da opção. O modelo acerta a pergunta quando a resposta certa tem o maior valor. Também será reportada a versão normalizada pelo número de tokens, porque opções longas são penalizadas na soma. Escolher ao acaso acerta 25%.

O formato foi escolhido porque o GPT-2 small não sabe responder perguntas livremente, mas sabe dizer qual continuação acha mais provável. Assim o experimento mede conhecimento, e não a capacidade de conversar.

Configurações comparadas:

1. **Original:** GPT-2 PT sem alterações.
2. **RAG:** as 3 fichas mais parecidas com a pergunta entram antes do prompt. Também se mede a busca em si: a ficha relevante apareceu entre as 3 recuperadas?
3. **Fine-tuning:** GPT-2 PT ajustado com LoRA nas fichas, sem contexto no prompt.
4. **Fine-tuning + RAG.**

### Cenário de atualização (PROCX)

A ficha `procx` tem `cenario: "atualizacao"`. Ela fica **fora** do fine-tuning e entra **só** no RAG, como se a função tivesse sido lançada depois do treino. O validador garante que nenhuma outra ficha menciona o PROCX. A pergunta `t11` mede o efeito: o fine-tuning não tem como saber a resposta, e o RAG sabe assim que a ficha é adicionada. Esse é o argumento prático a favor do RAG para conhecimento que muda.

## Revisão (Gustav)

- [ ] Confirmar no Excel os 9 detalhes marcados com `verificar` (o validador lista quais são). Corrigir o texto se algum estiver errado e apagar o campo.
- [ ] Ler as fichas e corrigir o que não bate com o uso real no dia a dia.
- [ ] Acrescentar fichas da sua experiência com transformação digital (dúvidas que colaboradores costumam ter). Use texto próprio, não copiado da documentação da Microsoft.
- [ ] Conferir se cada pergunta tem uma única resposta certa entre as quatro opções.
- [ ] Rodar `python scripts/validar_dados_excel.py`.
