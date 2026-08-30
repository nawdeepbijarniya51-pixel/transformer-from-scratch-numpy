"""
FastAPI backend for the from-scratch NumPy Transformer.

Loads a trained checkpoint ONCE at startup, then serves generation
and training-step requests over HTTP, exposing the *real* internal
tensors (embeddings, positional encodings, attention weights, FFN
activations, gradients) so a frontend can visualize actual computation
rather than a mock.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /                    info
    GET  /health              health check
    POST /generate             greedy/top-k generation, text only
    POST /generate/trace        generation + full real forward-pass trace
    GET  /train/sample          a random real (encoder, decoder) pair from the dataset
    POST /train/step             ONE real training step (forward+backward+SGD update)
    POST /model/reset            reload original checkpoint weights (undo training steps)
"""

import copy
import json
import os
import random

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from Transformer import Transformer
from SaL import load_model
from EncodeDecode import EncodDecode
from LogLoss import cross_entropy_loss, cross_entropy_backward


# ============================================================
# Config
# ============================================================

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "Final_1/final")
VOCAB_PATH = os.environ.get("VOCAB_PATH", "json_data/vocab.json")
PAIRS_PATH = os.environ.get("PAIRS_PATH", "json_data/pairs.json")

DEFAULT_MAX_LENGTH = 20
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 20
DEFAULT_LEARNING_RATE = 0.01


# ============================================================
# Load model + vocab ONCE, at import time (server startup)
# ============================================================

print(f"Loading vocab from {VOCAB_PATH} ...")
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    token_to_id = json.load(f)
id_to_token = {str(v): k for k, v in token_to_id.items()}

print(f"Loading config from {CHECKPOINT_DIR}/config.json ...")
with open(os.path.join(CHECKPOINT_DIR, "config.json"), "r") as f:
    config = json.load(f)

print("Building model ...")
model = Transformer(**config)

print(f"Loading weights from {CHECKPOINT_DIR} ...")
load_model(model, CHECKPOINT_DIR)

ed = EncodDecode()

START_ID = token_to_id["<START>"]
END_ID = token_to_id["<END>"]
PAD_ID = token_to_id.get("<PAD>")
UNK_ID = token_to_id.get("<UNK>")
SPECIAL_IDS = {START_ID, END_ID, PAD_ID, UNK_ID}

try:
    with open(PAIRS_PATH, "r", encoding="utf-8") as f:
        DATASET_PAIRS = json.load(f)
except FileNotFoundError:
    DATASET_PAIRS = []

print(f"Model ready. vocab_size={len(token_to_id)}  dataset_pairs={len(DATASET_PAIRS)}")


def _snapshot_initial_weights():
    """Keep an in-memory copy of the loaded checkpoint so /model/reset can restore it
    after live training steps have mutated the shared model in place."""
    return {name: arr.copy() for name, arr in model.named_parameters()}


_INITIAL_WEIGHTS = _snapshot_initial_weights()

# running "training session" step counter + loss history, purely for the UI
TRAIN_STEP_COUNT = 0
TRAIN_LOSS_HISTORY = []


# ============================================================
# Generation logic
# ============================================================

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def sample_top_k_with_trace(logits, temperature=1.0, top_k=20):
    """Same math as the original sample_top_k, but also returns the full
    real top-k candidate list (token id, scaled logit, probability) so the
    frontend can render exactly what the sampler saw."""
    logits = np.asarray(logits, dtype=np.float64)

    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    scaled_logits = logits / temperature
    k = min(top_k, len(scaled_logits))

    top_indices = np.argpartition(scaled_logits, -k)[-k:]
    # sort descending by scaled logit for a readable, ranked list
    order = np.argsort(-scaled_logits[top_indices])
    top_indices = top_indices[order]
    top_logits = scaled_logits[top_indices]
    probabilities = softmax(top_logits)

    next_id = int(np.random.choice(top_indices, p=probabilities))

    candidates = [
        {
            "token_id": int(tid),
            "token": id_to_token.get(str(int(tid)), "<UNK>"),
            "raw_logit": float(logits[tid]),
            "scaled_logit": float(scaled_logits[tid]),
            "probability": float(p),
        }
        for tid, p in zip(top_indices.tolist(), probabilities.tolist())
    ]

    return next_id, candidates


def generate_with_trace(encoder_ids, max_length=20, temperature=0.8, top_k=20):
    """Runs real incremental decoding, recording the real top-k/temperature
    softmax distribution the sampler actually drew from at every step."""
    encoder_ids_arr = np.array(encoder_ids).reshape(1, -1)
    encoder_mask_1d = np.ones_like(encoder_ids_arr)

    decoder_ids = [START_ID]
    steps = []

    for step_idx in range(max_length):
        decoder_input = np.array(decoder_ids).reshape(1, -1)
        decoder_mask_1d = np.ones_like(decoder_input)

        logits = model.forward(encoder_ids_arr, decoder_input, encoder_mask_1d, decoder_mask_1d)
        last_logits = logits[0, -1, :]

        next_id, candidates = sample_top_k_with_trace(last_logits, temperature=temperature, top_k=top_k)

        steps.append({
            "step": step_idx,
            "decoder_tokens_so_far": [id_to_token.get(str(i), "<UNK>") for i in decoder_ids],
            "candidates": candidates,
            "chosen_token_id": next_id,
            "chosen_token": id_to_token.get(str(next_id), "<UNK>"),
        })

        decoder_ids.append(next_id)
        if next_id == END_ID:
            break

    clean_tokens = [id_to_token[str(i)] for i in decoder_ids if i not in SPECIAL_IDS]
    return " ".join(clean_tokens), decoder_ids, steps


def get_input_tokens(text):
    if hasattr(ed, "clean_and_tokenize"):
        return ed.clean_and_tokenize(text)
    return text.lower().split()


# ============================================================
# Real intermediate-tensor extraction (shared by inference + training trace)
# ============================================================

def _attention_block(mha, query_tokens, key_tokens):
    """attn_weights cache shape: (1, num_heads, seq_q, seq_k)"""
    w = mha.cache["attn_weights"][0]  # (num_heads, seq_q, seq_k)
    return {
        "query_tokens": query_tokens,
        "key_tokens": key_tokens,
        "num_heads": int(w.shape[0]),
        "per_head": w.round(5).tolist(),
        "mean_over_heads": w.mean(axis=0).round(5).tolist(),
    }


def _ffn_block(ff):
    pre_relu = ff.relu.input[0]        # (seq, hidden_dim)
    post_relu = ff.linear2.input[0]    # (seq, hidden_dim)
    frac_active = float((pre_relu > 0).mean())
    return {
        "hidden_dim": int(pre_relu.shape[-1]),
        "pre_activation_sample": pre_relu[:, :16].round(4).tolist(),
        "post_relu_sample": post_relu[:, :16].round(4).tolist(),
        "fraction_neurons_active": round(frac_active, 4),
    }


def _layernorm_block(ln):
    x, normalized, mean, std = ln.cache
    return {
        "mean": mean[0, :, 0].round(5).tolist(),
        "std": std[0, :, 0].round(5).tolist(),
        "gamma_sample": ln.gamma[:16].round(4).tolist(),
        "beta_sample": ln.beta[:16].round(4).tolist(),
        "normalized_sample": normalized[0][:, :16].round(4).tolist(),
    }


def extract_encoder_layer(layer, input_tokens):
    return {
        "self_attention": _attention_block(layer.mha, input_tokens, input_tokens),
        "norm1": _layernorm_block(layer.norm1),
        "feed_forward": _ffn_block(layer.ff),
        "norm2": _layernorm_block(layer.norm2),
    }


def extract_decoder_layer(layer, decoder_tokens, encoder_tokens):
    return {
        "self_attention": _attention_block(layer.self_atten, decoder_tokens, decoder_tokens),
        "norm1": _layernorm_block(layer.norm1),
        "cross_attention": _attention_block(layer.cross_atten, decoder_tokens, encoder_tokens),
        "norm2": _layernorm_block(layer.norm2),
        "feed_forward": _ffn_block(layer.ff),
        "norm3": _layernorm_block(layer.norm3),
    }


def build_full_trace(input_tokens, encoder_ids, decoder_ids, decoder_tokens_for_forward):
    """Pulls real values out of every layer's forward-pass caches. Requires that
    model.forward() was just called with these exact encoder_ids / decoder ids."""
    embedding_sample = [
        model.embedding.weight[i].round(4).tolist()
        for i in encoder_ids
    ]
    positional_sample = model.pe.positional_encoding[: len(encoder_ids), :16].round(4).tolist()

    encoder_layers = [
        extract_encoder_layer(layer, input_tokens)
        for layer in model.encoder.layers
    ]
    decoder_layers = [
        extract_decoder_layer(layer, decoder_tokens_for_forward, input_tokens)
        for layer in model.decoder.layers
    ]

    return {
        "input_tokens": input_tokens,
        "input_ids": [int(i) for i in encoder_ids],
        "embedding_sample": embedding_sample,
        "embedding_dims_total": config["embedding_dim"],
        "positional_sample": positional_sample,
        "encoder_layers": encoder_layers,
        "decoder_tokens": decoder_tokens_for_forward,
        "decoder_layers": decoder_layers,
        "output_tokens": [id_to_token[str(i)] for i in decoder_ids],
        "num_layers": config["num_layers"],
        "num_heads": config["num_heads"],
    }


def _grad_stats(name, arr):
    if arr is None:
        return {"name": name, "shape": [], "norm": 0.0, "mean_abs": 0.0, "max_abs": 0.0}
    flat = np.asarray(arr).ravel()
    return {
        "name": name,
        "shape": list(np.asarray(arr).shape),
        "norm": float(np.linalg.norm(flat)),
        "mean_abs": float(np.mean(np.abs(flat))) if flat.size else 0.0,
        "max_abs": float(np.max(np.abs(flat))) if flat.size else 0.0,
    }


def build_gradient_report(encoder_ids, decoder_input_ids):
    """Real per-parameter-group gradient norms, plus a few full small matrices
    for close-up inspection, plus the actual (sparse) embedding-row gradients
    for the tokens that were used in this example."""
    named_grads = model.named_gradients()
    summary = [_grad_stats(name, g) for name, g in named_grads]

    named_grad_dict = dict(named_grads)

    spotlight = {}
    for key in [
        "encoder.layers.0.mha.W_q.weight",
        "decoder.layers.0.self_atten.W_q.weight",
        "decoder.layers.0.cross_atten.W_q.weight",
    ]:
        if key in named_grad_dict and named_grad_dict[key] is not None:
            spotlight[key] = named_grad_dict[key].round(6).tolist()

    embed_grad = named_grad_dict.get("embedding.weight")
    used_ids = sorted(set(int(i) for i in list(encoder_ids) + list(decoder_input_ids)))
    embedding_row_grads = []
    if embed_grad is not None:
        for tid in used_ids:
            embedding_row_grads.append({
                "token_id": tid,
                "token": id_to_token.get(str(tid), "<UNK>"),
                "grad_row": embed_grad[tid].round(6).tolist(),
                "grad_row_norm": float(np.linalg.norm(embed_grad[tid])),
            })

    out_proj_grad = named_grad_dict.get("output_projection.linear.weight")  # (embed_dim, vocab)
    top_output_tokens = []
    if out_proj_grad is not None:
        col_norms = np.linalg.norm(out_proj_grad, axis=0)  # (vocab,)
        top_idx = np.argsort(-col_norms)[:10]
        for tid in top_idx.tolist():
            top_output_tokens.append({
                "token_id": tid,
                "token": id_to_token.get(str(tid), "<UNK>"),
                "grad_col_norm": float(col_norms[tid]),
            })

    return {
        "per_parameter_summary": summary,
        "spotlight_matrices": spotlight,
        "embedding_row_gradients": embedding_row_grads,
        "top_output_projection_gradients": top_output_tokens,
    }


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="Transformer From Scratch — Inference & Training API",
    description="A from-scratch NumPy Transformer, serving real generation AND real training-step internals.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=200, description="Input sentence to respond to")
    max_length: int = Field(DEFAULT_MAX_LENGTH, ge=1, le=100)
    temperature: float = Field(DEFAULT_TEMPERATURE, gt=0, le=2.0)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=200)


class GenerateResponse(BaseModel):
    input: str
    output: str
    unknown_words: list[str]


class GenerateTraceResponse(GenerateResponse):
    trace: dict
    generation_steps: list


class TrainStepRequest(BaseModel):
    encoder_text: str = Field(..., min_length=1, max_length=200)
    decoder_text: str = Field(..., min_length=1, max_length=200)
    learning_rate: float = Field(DEFAULT_LEARNING_RATE, gt=0, le=1.0)
    apply_update: bool = Field(True, description="If true, actually applies the SGD update to the live model.")


def _unknown_words(text, tokens, encoder_ids):
    if UNK_ID is None:
        return []
    return [tok for tok, tid in zip(tokens, encoder_ids) if tid == UNK_ID]


@app.get("/api/status")
def api_status():
    return {
        "status": "ok",
        "message": "Transformer From Scratch API. POST /generate, /generate/trace, /train/step.",
        "checkpoint": CHECKPOINT_DIR,
        "vocab_size": len(token_to_id),
        "config": config,
        "train_steps_taken_this_session": TRAIN_STEP_COUNT,
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    encoder_ids = ed.encode(req.text)
    if len(encoder_ids) == 0:
        raise HTTPException(status_code=400, detail="Input could not be tokenized.")

    tokens = get_input_tokens(req.text)
    output_text, _, _ = generate_with_trace(
        encoder_ids,
        max_length=req.max_length,
        temperature=req.temperature,
        top_k=req.top_k,
    )

    return GenerateResponse(
        input=req.text,
        output=output_text,
        unknown_words=_unknown_words(req.text, tokens, encoder_ids),
    )


@app.post("/generate/trace", response_model=GenerateTraceResponse)
def generate_trace_endpoint(req: GenerateRequest):
    """
    Runs real autoregressive generation, recording the real top-k / temperature
    softmax distribution at every decoding step, then re-runs one full
    teacher-forced forward pass over the final sequence so every layer's
    embedding / positional / attention / FFN / layernorm caches reflect the
    complete real output — all returned for the frontend to visualize.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    encoder_ids = ed.encode(req.text)
    if len(encoder_ids) == 0:
        raise HTTPException(status_code=400, detail="Input could not be tokenized.")

    tokens = get_input_tokens(req.text)
    output_text, decoder_ids, generation_steps = generate_with_trace(
        encoder_ids,
        max_length=req.max_length,
        temperature=req.temperature,
        top_k=req.top_k,
    )

    encoder_ids_arr = np.array(encoder_ids).reshape(1, -1)
    encoder_mask_1d = np.ones_like(encoder_ids_arr)
    decoder_input_ids = decoder_ids[:-1] if len(decoder_ids) > 1 else decoder_ids
    decoder_input_arr = np.array(decoder_input_ids).reshape(1, -1)
    decoder_mask_1d = np.ones_like(decoder_input_arr)
    model.forward(encoder_ids_arr, decoder_input_arr, encoder_mask_1d, decoder_mask_1d)

    decoder_tokens_for_forward = [id_to_token[str(i)] for i in decoder_input_ids]
    trace = build_full_trace(tokens, encoder_ids, decoder_ids, decoder_tokens_for_forward)

    return GenerateTraceResponse(
        input=req.text,
        output=output_text,
        unknown_words=_unknown_words(req.text, tokens, encoder_ids),
        trace=trace,
        generation_steps=generation_steps,
    )


@app.get("/train/sample")
def train_sample():
    """A real (encoder, decoder) pair pulled from the actual training dataset,
    to prefill the training tab."""
    if not DATASET_PAIRS:
        raise HTTPException(status_code=404, detail="No dataset pairs available on this server.")
    pair = random.choice(DATASET_PAIRS)
    return {"encoder_text": pair["encoder"], "decoder_text": pair["decoder"]}


@app.post("/train/step")
def train_step_endpoint(req: TrainStepRequest):
    """
    Runs ONE real training step against the live in-memory model:
      1. tokenize encoder_text / decoder_text
      2. teacher-forced forward pass (real embeddings, positional encodings,
         attention, FFN, layernorm — captured layer by layer)
      3. real cross-entropy loss against the true next-token targets
      4. real backward pass (model.backward), producing real gradients
         for every parameter in the network
      5. (optional) a real SGD weight update: param -= lr * grad
      6. a second forward pass with the updated weights, to show the real
         loss after the step

    NOTE: this mutates the shared server-side model in place. Call
    POST /model/reset to restore the original checkpoint weights.
    """
    global TRAIN_STEP_COUNT

    encoder_ids = ed.encode(req.encoder_text)
    decoder_input_ids = ed.encode_decoder_input(req.decoder_text)
    decoder_target_ids = ed.encode_decoder_target(req.decoder_text)

    if len(encoder_ids) == 0 or len(decoder_input_ids) <= 1:
        raise HTTPException(status_code=400, detail="Could not tokenize one of the inputs.")

    encoder_ids_arr = np.array(encoder_ids).reshape(1, -1)
    encoder_mask_1d = np.ones_like(encoder_ids_arr)
    decoder_input_arr = np.array(decoder_input_ids).reshape(1, -1)
    decoder_mask_1d = np.ones_like(decoder_input_arr)
    target_arr = np.array(decoder_target_ids).reshape(1, -1)
    target_mask = np.ones_like(target_arr)

    # ---- forward pass BEFORE the update ----
    logits = model.forward(encoder_ids_arr, decoder_input_arr, encoder_mask_1d, decoder_mask_1d)
    loss_before, probs = cross_entropy_loss(logits, target_arr, target_mask)

    per_token_loss = (-np.log(
        probs[0, np.arange(len(decoder_target_ids)), decoder_target_ids] + 1e-9
    )).round(5).tolist()

    predicted_ids = np.argmax(logits[0], axis=-1).tolist()

    input_tokens = get_input_tokens(req.encoder_text)
    decoder_tokens_for_forward = [id_to_token.get(str(i), "<UNK>") for i in decoder_input_ids]
    target_tokens = [id_to_token.get(str(i), "<UNK>") for i in decoder_target_ids]
    predicted_tokens = [id_to_token.get(str(i), "<UNK>") for i in predicted_ids]

    # this is what the real intermediate-tensor trace looked like DURING this forward pass
    forward_trace = build_full_trace(input_tokens, encoder_ids, decoder_target_ids, decoder_tokens_for_forward)

    # ---- backward pass: real gradients ----
    grad_logits = cross_entropy_backward(probs, target_arr, target_mask)
    model.backward(grad_logits)
    gradient_report = build_gradient_report(encoder_ids, decoder_input_ids)

    loss_after = None
    weight_update_norms = []
    if req.apply_update:
        lr = req.learning_rate
        grad_dict = dict(model.named_gradients())
        for name, param in model.named_parameters():
            grad = grad_dict.get(name)
            if grad is None:
                continue
            delta = lr * grad
            param -= delta
            weight_update_norms.append({"name": name, "update_norm": float(np.linalg.norm(delta))})

        # ---- forward pass AFTER the update, to prove the loss actually moved ----
        logits_after = model.forward(encoder_ids_arr, decoder_input_arr, encoder_mask_1d, decoder_mask_1d)
        loss_after, _ = cross_entropy_loss(logits_after, target_arr, target_mask)
        loss_after = float(loss_after)

        TRAIN_STEP_COUNT += 1
        TRAIN_LOSS_HISTORY.append(loss_after)

    return {
        "encoder_text": req.encoder_text,
        "decoder_text": req.decoder_text,
        "input_tokens": input_tokens,
        "decoder_input_tokens": decoder_tokens_for_forward,
        "target_tokens": target_tokens,
        "predicted_tokens": predicted_tokens,
        "loss_before": float(loss_before),
        "loss_after": loss_after,
        "per_token_loss": per_token_loss,
        "forward_trace": forward_trace,
        "gradient_report": gradient_report,
        "weight_update_norms": weight_update_norms,
        "learning_rate": req.learning_rate,
        "applied_update": req.apply_update,
        "session_step_count": TRAIN_STEP_COUNT,
        "session_loss_history": TRAIN_LOSS_HISTORY[-50:],
    }


@app.post("/model/reset")
def model_reset():
    """Restore the original checkpoint weights, undoing any /train/step updates
    made during this server session."""
    global TRAIN_STEP_COUNT
    for name, param in model.named_parameters():
        param[...] = _INITIAL_WEIGHTS[name]
    TRAIN_STEP_COUNT = 0
    TRAIN_LOSS_HISTORY.clear()
    return {"status": "reset", "message": "Original checkpoint weights restored."}


# ============================================================
# Serve the frontend (static/) at the same origin/port as the API
# ============================================================

_static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
