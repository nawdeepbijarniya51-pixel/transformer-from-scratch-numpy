import numpy as np 


class LayerNorm:
    def __init__(self, dim, epsilon=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.epsilon = epsilon
        self.cache = None
        self.grad_gamma = None   # NEW
        self.grad_beta = None     # NEW

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        variance = x.var(axis=-1, keepdims=True)
        std = np.sqrt(variance + self.epsilon)
        normalized = (x - mean) / std
        self.cache = (x, normalized, mean, std)
        return normalized * self.gamma + self.beta

    def backward(self, grad_output):
        x, normalized, mean, std = self.cache
        d = x.shape[-1]

        self.grad_gamma = (grad_output * normalized).sum(axis=tuple(range(grad_output.ndim - 1)))   # CHANGED: self.
        self.grad_beta = grad_output.sum(axis=tuple(range(grad_output.ndim - 1)))                      # CHANGED: self.

        dnormalized = grad_output * self.gamma
        term1 = d * dnormalized
        term2 = dnormalized.sum(axis=-1, keepdims=True)
        term3 = normalized * (dnormalized * normalized).sum(axis=-1, keepdims=True)

        dx = (1.0 / d) * (1.0 / std) * (term1 - term2 - term3)

        return dx, self.grad_gamma, self.grad_beta   # CHANGED: return self. versions (same values, now also stored)

    def parameters(self):
        return [self.gamma, self.beta]

    def gradients(self):
        return [self.grad_gamma, self.grad_beta]


    def named_parameters(self, prefix=""):
        return [(f"{prefix}.gamma", self.gamma), (f"{prefix}.beta", self.beta)]

    def named_gradients(self, prefix=""):
        return [(f"{prefix}.gamma", self.grad_gamma), (f"{prefix}.beta", self.grad_beta)]