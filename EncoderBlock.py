import numpy as np 
from FeedForward import FeedForward
from LayerNorm import LayerNorm
from MultiHeadAttention import MultiHeadAttention,create_causal_mask,combine_masks

class EncoderBlock:
    def __init__(self,embedding_dim,num_heads,hidden_dim,seed = 42):
        self.mha = MultiHeadAttention(embedding_dim,num_heads,seed=seed)
        self.norm1 = LayerNorm(embedding_dim)
        self.ff = FeedForward(embedding_dim,hidden_dim,seed = seed+10)
        self.norm2 = LayerNorm(embedding_dim)
        self.cache = None

    def forward(self,x,mask = None):
        attention_out = self.mha.forward(query=x,key=x,value=x,mask=mask)
        x1 = self.norm1.forward(x+attention_out)
        ff_out = self.ff.forward(x1)
        x2 = self.norm2.forward(x1+ff_out)
        self.cache = (x,attention_out,x1,ff_out)
        return x2 

    def backward(self, grad_output):
        x, attn_out, x1, ff_out = self.cache

        grad_x1_plus_ff, grad_gamma2, grad_beta2 = self.norm2.backward(grad_output)
        grad_x1_from_residual = grad_x1_plus_ff          
        grad_ff_out = grad_x1_plus_ff                      
        grad_x1_from_ff, ff_grads = self.ff.backward(grad_ff_out)

      
        grad_x1 = grad_x1_from_residual + grad_x1_from_ff

       
        grad_x_plus_attn, grad_gamma1, grad_beta1 = self.norm1.backward(grad_x1)
        grad_x_from_residual = grad_x_plus_attn
        grad_attn_out = grad_x_plus_attn
        grad_q, grad_k, grad_v, attn_grads = self.mha.backward(grad_attn_out)
        grad_x_from_attn = grad_q + grad_k + grad_v   

       
        grad_x = grad_x_from_residual + grad_x_from_attn

        return grad_x, {"norm1": (grad_gamma1, grad_beta1), "norm2": (grad_gamma2, grad_beta2),
                        "ff": ff_grads, "attn": attn_grads}
        
    def parameters(self):
        return (self.mha.parameters() + self.norm1.parameters() 
                + self.ff.parameters() + self.norm2.parameters())

    def gradients(self):
        return (self.mha.gradients() + self.norm1.gradients() 
                + self.ff.gradients() + self.norm2.gradients())


    def named_parameters(self, prefix=""):
        return (self.mha.named_parameters(f"{prefix}.mha")
                + self.norm1.named_parameters(f"{prefix}.norm1")
                + self.ff.named_parameters(f"{prefix}.ff")
                + self.norm2.named_parameters(f"{prefix}.norm2"))

    def named_gradients(self, prefix=""):
        return (self.mha.named_gradients(f"{prefix}.mha")
                + self.norm1.named_gradients(f"{prefix}.norm1")
                + self.ff.named_gradients(f"{prefix}.ff")
                + self.norm2.named_gradients(f"{prefix}.norm2"))


