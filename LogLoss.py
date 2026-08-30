import numpy as np 

def softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def cross_entropy_loss(logits, targets, mask):
    probs = softmax(logits, axis=-1)

    batch_size, seq_len, vocab_size = logits.shape
    batch_idx = np.arange(batch_size)[:, None]
    seq_idx = np.arange(seq_len)[None, :]
    correct_probs = probs[batch_idx, seq_idx, targets]

    losses = -np.log(correct_probs + 1e-9)
    losses = losses * mask
    total_loss = losses.sum() / mask.sum()

    return total_loss, probs


def cross_entropy_backward(probs, targets, mask):
    batch_size, seq_len, vocab_size = probs.shape

    one_hot = np.zeros_like(probs)
    batch_idx = np.arange(batch_size)[:, None]
    seq_idx = np.arange(seq_len)[None, :]
    one_hot[batch_idx, seq_idx, targets] = 1

    grad_logits = (probs - one_hot)
    grad_logits = grad_logits * mask[:, :, None]
    grad_logits = grad_logits / mask.sum()

    return grad_logits