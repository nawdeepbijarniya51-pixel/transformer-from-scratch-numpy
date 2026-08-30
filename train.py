import numpy as np
import json
import time
from tqdm import tqdm
from SaL import save_model
from Transformer import Transformer
from EncodeDecode import EncodDecode
from TextPreprocessing import TextPreprocessing
from MultiHeadAttention import softmax
from ADAM import Adam
from LogLoss import cross_entropy_backward,cross_entropy_loss


def count_batches(encoded_pairs, batch_size):
    return (len(encoded_pairs) + batch_size - 1) // batch_size



DATASET_PATH = "json_data/pairs.json" 

VOCAB_SAVE_PATH = "json_data/vocab.json"
ENCODED_SAVE_PATH = "json_data/encoded_pairs.json"

embedding_dim = 32
num_heads = 4
hidden_dim = 128
num_layers = 2
learning_rate = 0.07
num_epochs = 1000
batch_size = 4


tp = TextPreprocessing(DATASET_PATH)
tp.load_data()
tp.build_vocab()

token_to_id = tp.token_to_id
id_to_token = tp.id_to_token
vocab_size = len(token_to_id)

with open(VOCAB_SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(token_to_id, f, indent=2)

print(f"Built vocab: {vocab_size} tokens")
ed = EncodDecode()

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_pairs = json.load(f) 

print(f"Loaded {len(raw_pairs)} raw pairs from {DATASET_PATH}")
id_to_token = tp.id_to_token
token_to_id = tp.token_to_id
vocab_size = len(token_to_id)

with open(VOCAB_SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(token_to_id, f, indent=2)

print(f"Built vocab: {vocab_size} tokens")

encoded_pairs = []
for pair in raw_pairs:
    encoder_ids = ed.encode(pair["encoder"])
    decoder_input_ids = ed.encode_decoder_input(pair["decoder"])
    decoder_target_ids = ed.encode_decoder_target(pair["decoder"])
    encoded_pairs.append({
        "encoder_ids": encoder_ids,
        "decoder_input_ids": decoder_input_ids,
        "decoder_target_ids": decoder_target_ids,
    })

with open(ENCODED_SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(encoded_pairs, f, indent=2)

print(f"Encoded {len(encoded_pairs)} pairs")
print("=" * 60)



print(f"Vocab size:     {vocab_size}")
print(f"Embedding dim:  {embedding_dim}")
print(f"Num heads:      {num_heads}")
print(f"Hidden dim:     {hidden_dim}")
print(f"Num layers:     {num_layers}")
print(f"Learning rate:  {learning_rate}")
print(f"Batch size:     {batch_size}")
print(f"Epochs:         {num_epochs}")
print("=" * 60)

model = Transformer(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    num_layers=num_layers
)
optimizer = Adam(learning_rate=learning_rate)

print(f"Total trainable parameters: {len(model.parameters())}")
print("=" * 60)

loss_history = []
total_batches = count_batches(encoded_pairs, batch_size)


config = {
    "vocab_size": vocab_size,
    "embedding_dim": embedding_dim,
    "num_heads": num_heads,
    "hidden_dim": hidden_dim,
    "num_layers": num_layers,
}
for epoch in range(num_epochs):
    epoch_start = time.time()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(
        ed.make_batches(encoded_pairs, batch_size=batch_size),
        total=total_batches,
        desc=f"Epoch {epoch + 1}/{num_epochs}",
        unit="batch",
        ncols=100,
        leave=False
    )
    if (epoch + 1) % 25 == 0:
        save_model(model, f"checkpoints/epoch_{epoch+1}", config)
    for batch in progress_bar:
        logits = model.forward(
            batch["encoder_ids"], batch["decoder_input_ids"],
            batch["encoder_mask"], batch["decoder_mask"]
        )

        loss, probs = cross_entropy_loss(logits, batch["decoder_target_ids"], batch["decoder_mask"])
        grad_logits = cross_entropy_backward(probs, batch["decoder_target_ids"], batch["decoder_mask"])
        model.backward(grad_logits)
        optimizer.step(model.parameters(), model.gradients())

        total_loss += loss
        num_batches += 1
        progress_bar.set_postfix({"loss": f"{total_loss / num_batches:.4f}"})

    avg_loss = total_loss / num_batches
    loss_history.append(avg_loss)
    epoch_time = time.time() - epoch_start

    trend = "↓" if len(loss_history) > 1 and loss_history[-1] < loss_history[-2] else ("-" if len(loss_history) == 1 else "↑")
    print(f"Epoch {epoch + 1:3d}/{num_epochs} | avg loss: {avg_loss:.4f} {trend} | time: {epoch_time:.1f}s")





save_model(model, "checkpoints/final", config)
print("=" * 60)
print("Training complete.")
print(f"Loss: {loss_history[0]:.4f} → {loss_history[-1]:.4f}")
print("=" * 60)
