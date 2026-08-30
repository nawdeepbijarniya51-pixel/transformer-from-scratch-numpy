import numpy as np 


class LinearLayer:
    def __init__(self,input_dim,output_dim,seed = 42):
        self.input_dim = input_dim
        self.output_dim = output_dim
        rng = np.random.default_rng(seed)
        self.weight = rng.standard_normal((input_dim, output_dim)) * 0.01
        self.bias = np.zeros(output_dim)
        self.input = None

    def forward(self,embd_vct):
        self.input = embd_vct
        return embd_vct @ self.weight + self.bias

    
    def backward(self, grad_output):
        x_flat = self.input.reshape(-1, self.input.shape[-1])
        grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
        self.grad_weight = x_flat.T @ grad_output_flat
        self.grad_bias = grad_output_flat.sum(axis=0)
        grad_input = grad_output @ self.weight.T

        return grad_input, self.grad_weight, self.grad_bias

    def parameters(self):        
        return [self.weight, self.bias]

    def gradients(self):          
        return [self.grad_weight, self.grad_bias]
        
    def named_parameters(self, prefix=""):
        return [(f"{prefix}.weight", self.weight), (f"{prefix}.bias", self.bias)]

    def named_gradients(self, prefix=""):
        return [(f"{prefix}.weight", self.grad_weight), (f"{prefix}.bias", self.grad_bias)]
