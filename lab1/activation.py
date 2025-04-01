from abc import abstractmethod

import numpy as np


class Activation:
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass


class Sigmoid(Activation):
    def forward(self, z):
        return 1 / (1 + np.exp(-z))

    def backward(self, a):
        return np.multiply(a, 1 - a)


class Identity(Activation):
    def forward(self, z):
        return z

    def backward(self, a):
        return np.ones_like(a)
