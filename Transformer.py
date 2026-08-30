import numpy as np
from EmbaddingLayer import Embedding
from Encoder import Encoder
from Decoder import Decoder
from OutputProjection import OutputProjection
from positionalEncoding import PositionalEncoding
from MultiHeadAttention import create_padding_mask, create_causal_mask, combine_masks


class Transformer:
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, max_seq_len=50, seed=42):
        self.embedding = Embedding(vocab_size, embedding_dim, seed=seed)

        dummy = np.zeros((1, max_seq_len, embedding_dim))
        self.pe = PositionalEncoding(dummy)

        self.encoder = Encoder(num_layers, embedding_dim, num_heads, hidden_dim, seed=seed)
        self.decoder = Decoder(num_layers, embedding_dim, num_heads, hidden_dim, seed=seed)
        self.output_projection = OutputProjection(embedding_dim, vocab_size, seed=seed)

        self.cache = None

    def forward(self, encoder_ids, decoder_input_ids, encoder_padding_mask_1d, decoder_padding_mask_1d):

        encoder_embeds = self.embedding.forward(encoder_ids)
        encoder_input = encoder_embeds + self.pe.positional_encoding[:encoder_embeds.shape[1]]

        decoder_embeds = self.embedding.forward(decoder_input_ids)
        decoder_input = decoder_embeds + self.pe.positional_encoding[:decoder_embeds.shape[1]]

        encoder_padding = create_padding_mask(encoder_padding_mask_1d)
        decoder_padding = create_padding_mask(decoder_padding_mask_1d)

        decoder_seq_len = decoder_input.shape[1]
        causal = create_causal_mask(decoder_seq_len)
        decoder_causal_mask = combine_masks(causal, decoder_padding)


        encoder_output = self.encoder.forward(encoder_input, mask=encoder_padding)

        decoder_output = self.decoder.forward(
            decoder_input, encoder_output,
            causal_mask=decoder_causal_mask, cross_mask=encoder_padding
        )

        logits = self.output_projection.forward(decoder_output)

        self.cache = (encoder_ids, decoder_input_ids) 
        return logits
    
    def backward(self, grad_logits):
        grad_decoder_output, grad_op_weight, grad_op_bias = self.output_projection.backward(grad_logits)

        grad_decoder_input, grad_encoder_output, decoder_grads = self.decoder.backward(grad_decoder_output)

        grad_encoder_input, encoder_grads = self.encoder.backward(grad_encoder_output)

        self.embedding.input_ids = np.asarray(self.cache[0])  
        self.embedding.backward(grad_encoder_input)
        grad_weight_from_encoder = self.embedding.grad_weight.copy()

        self.embedding.input_ids = np.asarray(self.cache[1])   
        self.embedding.backward(grad_decoder_input)
        self.embedding.grad_weight += grad_weight_from_encoder

    def parameters(self):
        return (self.embedding.parameters() + self.encoder.parameters()
                + self.decoder.parameters() + self.output_projection.parameters())

    def gradients(self):
        return (self.embedding.gradients() + self.encoder.gradients()
                + self.decoder.gradients() + self.output_projection.gradients())

    def named_parameters(self):
        return (self.embedding.named_parameters("embedding")
                + self.encoder.named_parameters("encoder")
                + self.decoder.named_parameters("decoder")
                + self.output_projection.named_parameters("output_projection"))

    def named_gradients(self):
        return (self.embedding.named_gradients("embedding")
                + self.encoder.named_gradients("encoder")
                + self.decoder.named_gradients("decoder")
                + self.output_projection.named_gradients("output_projection"))


