from math import e
import numpy as np

class Activation:
    def __init__(self, forward=lambda x: max(x, 0), backward=lambda x: 1 if x>0 else 0):
        self.forward = forward
        self.backward = backward

relu = Activation(lambda x: x * (x>0), lambda x: x>0)

f = lambda x: 1/(1+e**(-x))
sigmoid = Activation(f, lambda x: (1-f(x)) * f(x))

idact = Activation(lambda x: x, lambda x: np.ones_like(x))
