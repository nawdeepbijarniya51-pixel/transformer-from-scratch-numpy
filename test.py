import numpy as np
from SaL import load_model
from Transformer import Transformer
import json
from EncodeDecode import EncodDecode


with open("checkpoints/epoch_225/config.json", "r") as f:
    config = json.load(f)

model = Transformer(**config)
load_model(model, "checkpoints/epoch_225")

with open("json_data/vocab.json", "r", encoding="utf-8") as f:
    token_to_id = json.load(f)

id_to_token = {str(v): k for k, v in token_to_id.items()}

ed = EncodDecode()


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def sample_top_k(logits, temperature=1.0, top_k=20):
    logits = np.asarray(logits, dtype=np.float64)

    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    logits = logits / temperature

    top_k = min(top_k, len(logits))

    top_indices = np.argpartition(logits, -top_k)[-top_k:]
    top_logits = logits[top_indices]

    probabilities = softmax(top_logits)

    next_id = np.random.choice(
        top_indices,
        p=probabilities
    )

    return int(next_id)


def generate(
    model,
    encoder_ids,
    token_to_id,
    id_to_token,
    max_length=20,
    temperature=0.8,
    top_k=20
):
    START_ID = token_to_id["<START>"]
    END_ID = token_to_id["<END>"]

    encoder_ids = np.array(encoder_ids).reshape(1, -1)
    encoder_mask_1d = np.ones_like(encoder_ids)

    decoder_ids = [START_ID]

    for step in range(max_length):
        decoder_input = np.array(decoder_ids).reshape(1, -1)
        decoder_mask_1d = np.ones_like(decoder_input)

        logits = model.forward(
            encoder_ids,
            decoder_input,
            encoder_mask_1d,
            decoder_mask_1d
        )

        last_logits = logits[0, -1, :]

        next_id = sample_top_k(
            last_logits,
            temperature=temperature,
            top_k=top_k
        )

        decoder_ids.append(next_id)
       
        if next_id == END_ID:
            break

    tokens = [
        id_to_token[str(i)]
        for i in decoder_ids
    ]

    return " ".join(tokens), decoder_ids


temperature = 0.7
top_k = 20
max_length = 20

print("Loaded model from checkpoints/epoch_50")
print("Temperature:", temperature)
print("Top-K:", top_k)
print("Max length:", max_length)
print()

while True:
    test_sentence = input("You : ")

    if not test_sentence.strip():
        break

    encoder_ids = ed.encode(test_sentence)

    output_text, output_ids = generate(
        model,
        encoder_ids,
        token_to_id,
        id_to_token,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k
    )
    final = ""
    x = 1
    for ch in output_text:
      if ch is "<":
          x = 0
      if x is 1:
          final += ch
      if ch is ">":
          x = 1
    print("Output:", final)
    print()
