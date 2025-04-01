import numpy as np

import activation
from layer import Layer, Linear


class Model:
    def __init__(self):
        self.net = []
        self.delta = np.empty(0, dtype=object)

    def __call__(self, x):
        return self.forward(x)

    def add(self, layer):
        self.net.append(layer)

    def compile(self):
        self.delta = np.empty(len(self.net) + 1, dtype=object)

    def forward(self, x):
        for layer in self.net:
            x = layer.forward(x)
        return x

    def backward(self, delta, predict):
        self.delta[-1] = delta
        for i in reversed(range(len(self.net))):
            if isinstance(self.net[i], Layer):
                self.delta[i] = self.net[i].backward(self.delta[i + 1])
            elif len(self.net) <= i + 1:
                self.delta[i] = self.delta[i + 1] * self.net[i].backward(predict)
            else:
                self.delta[i] = self.delta[i + 1] * self.net[i].backward(self.net[i + 1].a)
