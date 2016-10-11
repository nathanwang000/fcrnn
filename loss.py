import numpy as np

class Loss:
    def __init__(self):
        pass

    def forward(self, x, t):
        pass

    def backward(self, x, t):
        pass

class se(Loss): # squared error

    def __init__(self):
        pass
    
    def forward(self, x, t):
        return np.sum(np.square(x-t))        

    def backward(self, x, t): # dE/dx
        return 2*(x-t)
        
