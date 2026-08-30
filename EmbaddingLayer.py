from EncodeDecode import EncodDecode
import numpy as np
from TextPreprocessing import TextPreprocessing

tp = TextPreprocessing()
ed = EncodDecode()


class Embedding:

    def __init__(self, vocab_size, embedding_dim=32, seed=42):
        self.embedding_dim = embedding_dim
        rng = np.random.default_rng(seed)
        self.weight = rng.standard_normal(
            (vocab_size, embedding_dim)
        ) * 0.01
        self.input_ids = None
        self.grad_weight = None   # NEW

    def forward(self, ids):
        self.input_ids = ids
        return self.weight[ids]

    def backward(self, grad_output):
        self.grad_weight = np.zeros_like(self.weight)
        input_ids = np.asarray(self.input_ids)   # ensure numpy array, regardless of what was passed in
        np.add.at(self.grad_weight, input_ids, grad_output)
        return self.grad_weight
    
    def create_embedding(self, txt):
        encoded = ed.encode(txt)
        final = self.forward(encoded)
        return final

    def parameters(self):
        return [self.weight]

    def gradients(self):
        return [self.grad_weight]

    def named_parameters(self, prefix=""):
        return [(f"{prefix}.weight", self.weight)]

    def named_gradients(self, prefix=""):
        return [(f"{prefix}.weight", self.grad_weight)]

