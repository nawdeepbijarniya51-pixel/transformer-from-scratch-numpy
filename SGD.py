class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, parameters, gradients):
        for param, grad in zip(parameters, gradients):
            param -= self.learning_rate * grad


    