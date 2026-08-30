from MultiHeadAttention import MultiHeadAttention
from LayerNorm import LayerNorm
from FeedForward import FeedForward

class DecoderBlock:
    def __init__(self,embedding_dim,num_heads,hidden_dim,seed = 42):
        self.self_atten = MultiHeadAttention(embedding_dim,num_heads,seed = seed)
        self.norm1 = LayerNorm(embedding_dim)

        self.cross_atten = MultiHeadAttention(embedding_dim,num_heads,seed=seed+1)
        self.norm2 = LayerNorm(embedding_dim)

        self.ff = FeedForward(embedding_dim,hidden_dim,seed=seed+10)
        self.norm3 = LayerNorm(embedding_dim)

        self.cache = None

    def forward(self, x, encoder_output, causal_mask=None, cross_mask=None):
        self_atten_out = self.self_atten.forward(query=x, key=x, value=x, mask=causal_mask)
        x1 = self.norm1.forward(x + self_atten_out)

        cross_atten_out = self.cross_atten.forward(query=x1, key=encoder_output, value=encoder_output, mask=cross_mask)
        x2 = self.norm2.forward(cross_atten_out + x1)

        ff_out = self.ff.forward(x2)
        x3 = self.norm3.forward(x2 + ff_out)

        self.cache = (x, self_atten_out, x1, cross_atten_out, x2, ff_out)
        return x3      

    def backward(self, grad_output, ):
        x, self_attn_out, x1, cross_attn_out, x2, ff_out = self.cache

        grad_x2_plus_ff, g3, b3 = self.norm3.backward(grad_output)
        grad_x2_from_residual = grad_x2_plus_ff
        grad_ff_out = grad_x2_plus_ff
        grad_x2_from_ff, ff_grads = self.ff.backward(grad_ff_out)
        grad_x2 = grad_x2_from_residual + grad_x2_from_ff

        grad_x1_plus_cross, g2, b2 = self.norm2.backward(grad_x2)
        grad_x1_from_residual = grad_x1_plus_cross
        grad_cross_out = grad_x1_plus_cross

        grad_query_cross, grad_key_cross, grad_value_cross, cross_grads = self.cross_atten.backward(grad_cross_out)
        grad_x1_from_cross = grad_query_cross                               
        grad_encoder_output = grad_key_cross + grad_value_cross          

        grad_x1 = grad_x1_from_residual + grad_x1_from_cross

        grad_x_plus_self, g1, b1 = self.norm1.backward(grad_x1)
        grad_x_from_residual = grad_x_plus_self
        grad_self_attn_out = grad_x_plus_self

        grad_q, grad_k, grad_v, self_attn_grads = self.self_atten.backward(grad_self_attn_out)
        grad_x_from_self_attn = grad_q + grad_k + grad_v   

        grad_x = grad_x_from_residual + grad_x_from_self_attn

        grads = {"norm1": (g1,b1), "norm2": (g2,b2), "norm3": (g3,b3),
                "self_attn": self_attn_grads, "cross_attn": cross_grads, "ff": ff_grads}

        return grad_x, grad_encoder_output, grads


    def parameters(self):
        return (self.self_atten.parameters() + self.norm1.parameters()
                + self.cross_atten.parameters() + self.norm2.parameters()
                + self.ff.parameters() + self.norm3.parameters())

    def gradients(self):
        return (self.self_atten.gradients() + self.norm1.gradients()
                + self.cross_atten.gradients() + self.norm2.gradients()
                + self.ff.gradients() + self.norm3.gradients())

    def named_parameters(self, prefix=""):
        return (self.self_atten.named_parameters(f"{prefix}.self_atten")
                + self.norm1.named_parameters(f"{prefix}.norm1")
                + self.cross_atten.named_parameters(f"{prefix}.cross_atten")
                + self.norm2.named_parameters(f"{prefix}.norm2")
                + self.ff.named_parameters(f"{prefix}.ff")
                + self.norm3.named_parameters(f"{prefix}.norm3"))

    def named_gradients(self, prefix=""):
        return (self.self_atten.named_gradients(f"{prefix}.self_atten")
                + self.norm1.named_gradients(f"{prefix}.norm1")
                + self.cross_atten.named_gradients(f"{prefix}.cross_atten")
                + self.norm2.named_gradients(f"{prefix}.norm2")
                + self.ff.named_gradients(f"{prefix}.ff")
                + self.norm3.named_gradients(f"{prefix}.norm3"))
