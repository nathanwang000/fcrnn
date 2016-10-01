from neuron import *
import numpy as np

class Data():
    # f is function to approximate    
    def __init__(self, f=None, x=None, y=None,
                 n=10000, tr_p=0.7):
        assert(f or (x and y))
        self.f = f
        self.n = len(x) if x is not None else n
        self.tr_p = tr_p
        if (x is None):
            self.x = np.arange(n).reshape(n,1)
        else:
            self.x = x
        if (y is None):
            self.y = np.array([f(x) for x in self.x])
        else:
            self.y = y
        self.cut = int(self.n*self.tr_p)
        self.pt = 0 # pointer insider data

    def shuffle(self):
        p = np.random.permutation(self.n)
        self.x = self.x[p]
        self.y = self.y[p]

    def getTr(self):
        return Data(self.f, self.x[:self.cut], self.y[:self.cut])

    def getTe(self):
        return Data(self.f, self.x[self.cut:], self.y[self.cut:])

    def getBatch(self, bs):
        if self.pt >= self.n: return False
        d = Data(self.f, self.x[self.pt:min(self.pt+bs,self.n)],
                         self.y[self.pt:min(self.pt+bs,self.n)])
        self.pt = min(self.pt+bs,self.n)
        return d

    def reset(self):
        self.pt = 0

    def progress(self):
        return "{}/{}".format(self.pt, self.n)

    

