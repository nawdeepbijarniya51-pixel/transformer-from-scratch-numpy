import numpy as np 


class ReLu:
    def __init__(self):
        self.input = None

    def forward(self,x):
        self.input = x
        return np.maximum(0,x)

    def backward(self,grad_output):
        grad_input = grad_output*(self.input>0)
        return grad_input



