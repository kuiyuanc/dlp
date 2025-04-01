from layer import Layer


class SGD:
    def __init__(self, model, lr=5e-2):
        self.model = model
        self.lr = lr

    def step(self):
        for i, layer in enumerate(self.model.net):
            if isinstance(layer, Layer):
                layer.w -= self.lr * (self.model.delta[i + 1] @ layer.a.T)
                layer.b -= self.lr * self.model.delta[i + 1]
