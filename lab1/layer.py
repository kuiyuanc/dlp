from abc import abstractmethod

import numpy as np


class Layer:
    def __init__(self):
        self.w = None
        self.b = None
        self.a = None

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass


class Linear(Layer):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.w = np.random.randn(out_features, in_features)
        self.b = np.random.randn(out_features, 1)

    def forward(self, x):
        self.a = x
        return self.w @ x + self.b

    def backward(self, delta):
        return self.w.T @ delta
