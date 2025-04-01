import numpy as np


def BCE(prediction, target, epsilon=1e-6):
    prediction = np.clip(prediction, epsilon, 1 - epsilon)
    loss = np.mean((target - 1) * np.log(1 - prediction) - target * np.log(prediction))
    delta = np.mean((1 - target) / (1 - prediction) - target / prediction)
    return loss, delta
