import numpy as np
from LinearLayer import LinearLayer

def softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def split_heads(x, num_heads):
    batch_size, seq_len, embedding_dim = x.shape
    head_dim = embedding_dim // num_heads
    x = x.reshape(batch_size, seq_len, num_heads, head_dim)
    x = x.transpose(0, 2, 1, 3)
    return x


def combine_heads(x):
    batch_size, num_heads, seq_len, head_dim = x.shape
    x = x.transpose(0, 2, 1, 3)  
    x = x.reshape(batch_size, seq_len, num_heads * head_dim) 
    return x


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k) 

    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)   # mask used as-is, no reshaping here

    weights = softmax(scores, axis=-1)
    output = weights @ V 
    return output, weights


def create_causal_mask(seq_len):
    mask = np.tril(np.ones((seq_len, seq_len)))
    return mask


def create_padding_mask(padding_mask_1d):
    mask = padding_mask_1d[:, np.newaxis, np.newaxis, :]
    return mask.astype(np.float32)


def combine_masks(causal_mask, padding_mask):
    combined = causal_mask[np.newaxis, np.newaxis, :, :] * padding_mask
    return combined


def softmax_backward(grad_output, softmax_output, axis=-1):
    sum_term = np.sum(grad_output * softmax_output, axis=axis, keepdims=True)
    grad_input = softmax_output * (grad_output - sum_term)
    return grad_input


def mask_backward(grad, mask):
    return np.where(mask == 0, 0.0, grad)


def split_heads_backward(grad):
    grad = grad.transpose(0, 2, 1, 3)   
    batch_size, seq_len, num_heads, head_dim = grad.shape
    grad = grad.reshape(batch_size, seq_len, num_heads * head_dim)
    return grad


def combine_heads_backward(grad, num_heads):
    batch_size, seq_len, embedding_dim = grad.shape
    head_dim = embedding_dim // num_heads
    grad = grad.reshape(batch_size, seq_len, num_heads, head_dim)
    grad = grad.transpose(0, 2, 1, 3) 
    return grad


class MultiHeadAttention:
    def __init__(self, embedding_dim, num_heads, seed=42):
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.W_q = LinearLayer(embedding_dim, embedding_dim, seed=seed)
        self.W_k = LinearLayer(embedding_dim, embedding_dim, seed=seed + 1)
        self.W_v = LinearLayer(embedding_dim, embedding_dim, seed=seed + 2)
        self.W_o = LinearLayer(embedding_dim, embedding_dim, seed=seed + 3)

        self.cache = None  

    def forward(self, query, key, value, mask=None):
        Q = self.W_q.forward(query)
        K = self.W_k.forward(key)
        V = self.W_v.forward(value)

        Q_split = split_heads(Q, self.num_heads)
        K_split = split_heads(K, self.num_heads)
        V_split = split_heads(V, self.num_heads)

        attn_output, attn_weights = scaled_dot_product_attention(Q_split, K_split, V_split, mask)
       
        attn_output_combined = combine_heads(attn_output)
        output = self.W_o.forward(attn_output_combined)
        self.cache = {
            "query": query, "key": key, "value": value,
            "Q_split": Q_split, "K_split": K_split, "V_split": V_split,
            "attn_output": attn_output, "attn_weights": attn_weights,
            "attn_output_combined": attn_output_combined,
        }
        return output

    def backward(self, grad_output):
        cache = self.cache
        Q_split, K_split, V_split = cache["Q_split"], cache["K_split"], cache["V_split"]
        attn_weights = cache["attn_weights"]


        grad_attn_combined, grad_Wo_w, grad_Wo_b = self.W_o.backward(grad_output)

        grad_attn = combine_heads_backward(grad_attn_combined, self.num_heads)

        grad_weights = grad_attn @ V_split.transpose(0, 1, 3, 2)
        grad_V_split = attn_weights.transpose(0, 1, 3, 2) @ grad_attn

        grad_scores_masked = softmax_backward(grad_weights, attn_weights)

        d_k = Q_split.shape[-1]
        grad_scores = grad_scores_masked / np.sqrt(d_k)

        grad_Q_split = grad_scores @ K_split
        grad_K_split = grad_scores.transpose(0, 1, 3, 2) @ Q_split

        grad_Q_combined = split_heads_backward(grad_Q_split)
        grad_K_combined = split_heads_backward(grad_K_split)
        grad_V_combined = split_heads_backward(grad_V_split)

        grad_query, grad_Wq_w, grad_Wq_b = self.W_q.backward(grad_Q_combined)
        grad_key, grad_Wk_w, grad_Wk_b = self.W_k.backward(grad_K_combined)
        grad_value, grad_Wv_w, grad_Wv_b = self.W_v.backward(grad_V_combined)

        grads = {
            "Wq": (grad_Wq_w, grad_Wq_b), "Wk": (grad_Wk_w, grad_Wk_b),
            "Wv": (grad_Wv_w, grad_Wv_b), "Wo": (grad_Wo_w, grad_Wo_b),
        }
        return grad_query, grad_key, grad_value, grads

    def parameters(self):
        return (self.W_q.parameters() + self.W_k.parameters() 
                + self.W_v.parameters() + self.W_o.parameters())

    def gradients(self):
        return (self.W_q.gradients() + self.W_k.gradients() 
                + self.W_v.gradients() + self.W_o.gradients())

    def named_parameters(self, prefix=""):
        return (self.W_q.named_parameters(f"{prefix}.W_q")
                + self.W_k.named_parameters(f"{prefix}.W_k")
                + self.W_v.named_parameters(f"{prefix}.W_v")
                + self.W_o.named_parameters(f"{prefix}.W_o"))

    def named_gradients(self, prefix=""):
        return (self.W_q.named_gradients(f"{prefix}.W_q")
                + self.W_k.named_gradients(f"{prefix}.W_k")
                + self.W_v.named_gradients(f"{prefix}.W_v")
                + self.W_o.named_gradients(f"{prefix}.W_o"))