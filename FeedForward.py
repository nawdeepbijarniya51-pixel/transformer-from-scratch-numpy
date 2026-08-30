from LinearLayer import LinearLayer
from Relu import ReLu


class FeedForward:
    def __init__(self, embedding_dim, hidden_dim, seed=42):
        self.linear1 = LinearLayer(embedding_dim, hidden_dim, seed=seed)
        self.relu = ReLu()
        self.linear2 = LinearLayer(hidden_dim, embedding_dim, seed=seed + 1)

    def forward(self, x):
        hidden = self.linear1.forward(x)         
        activated = self.relu.forward(hidden)      
        output = self.linear2.forward(activated)    
        return output

    def backward(self, grad_output):
        grad_activated, grad_w2, grad_b2 = self.linear2.backward(grad_output)
        grad_hidden = self.relu.backward(grad_activated)
        grad_input, grad_w1, grad_b1 = self.linear1.backward(grad_hidden)

        return grad_input, (grad_w1, grad_b1, grad_w2, grad_b2)

    def parameters(self):
        return self.linear1.parameters() + self.linear2.parameters()

    def gradients(self):
        return self.linear1.gradients() + self.linear2.gradients()

    def named_parameters(self, prefix=""):
        return (self.linear1.named_parameters(f"{prefix}.linear1")
                + self.linear2.named_parameters(f"{prefix}.linear2"))

    def named_gradients(self, prefix=""):
        return (self.linear1.named_gradients(f"{prefix}.linear1")
                + self.linear2.named_gradients(f"{prefix}.linear2"))






    