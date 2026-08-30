from EncoderBlock import EncoderBlock
from EmbaddingLayer import Embedding
import numpy as np


class Encoder:
    def __init__(self, num_layers, embedding_dim, num_heads, hidden_dim, seed=42):
        self.layers = [
            EncoderBlock(
                embedding_dim,
                num_heads,
                hidden_dim,
                seed=seed + i * 100
            )
            for i in range(num_layers)
        ]

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer.forward(x, mask=mask)
        return x

    def backward(self, grad_output):
        grad = grad_output
        all_grads = []
        for layer in reversed(self.layers):   
            grad, layer_grads = layer.backward(grad)
            all_grads.append(layer_grads)
        all_grads.reverse()  
        return grad, all_grads

    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params

    def gradients(self):
        grads = []
        for layer in self.layers:
            grads += layer.gradients()
        return grads

    def named_parameters(self, prefix=""):
        params = []
        for i, layer in enumerate(self.layers):
            params += layer.named_parameters(f"{prefix}.layers.{i}")
        return params

    def named_gradients(self, prefix=""):
        grads = []
        for i, layer in enumerate(self.layers):
            grads += layer.named_gradients(f"{prefix}.layers.{i}")
        return grads