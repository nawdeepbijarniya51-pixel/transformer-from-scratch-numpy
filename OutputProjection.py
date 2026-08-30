from LinearLayer import LinearLayer

class OutputProjection:
    def __init__(self, embedding_dim, vocab_size, seed=42):
        self.linear = LinearLayer(embedding_dim, vocab_size, seed=seed)

    def forward(self, x):
        return self.linear.forward(x)   

    def backward(self, grad_output):
        return self.linear.backward(grad_output)

    def parameters(self):
        return self.linear.parameters()

    def gradients(self):
        return self.linear.gradients()

    def named_parameters(self, prefix=""):
        return self.linear.named_parameters(f"{prefix}.linear")

    def named_gradients(self, prefix=""):
        return self.linear.named_gradients(f"{prefix}.linear")