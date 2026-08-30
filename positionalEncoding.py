import numpy as np 

class PositionalEncoding:

    def get_ps_encoding(self):
            mxln = self.mxln
            embd_dim = self.embd_dim
            position = np.arange(mxln)[:,np.newaxis]
            dim_index = np.arange(embd_dim)[np.newaxis,:]
            angle_rates = 1/np.power(10000,(2*(dim_index//2))/embd_dim)
            angles = position*angle_rates
            pe = np.zeros((mxln,embd_dim))
            pe[:,0::2] = np.sin(angles[:,0::2])
            pe[:,1::2] = np.cos(angles[:,1::2])
            return pe 
    
    def __init__(self,token_embeddings):
        self.mxln = token_embeddings.shape[1]
        self.embd_dim = token_embeddings.shape[2]
        self.positional_encoding = self.get_ps_encoding()


    def add_positional_encoding(self,token_embeding):
        return token_embeding + self.positional_encoding
        
