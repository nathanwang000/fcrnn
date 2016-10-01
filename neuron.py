import numpy as np
import test
import problem
import loss
from math import e

class Activation:
    def __init__(self, forward=lambda x: max(x, 0), backward=lambda x: 1 if x>0 else 0):
        self.forward = forward
        self.backward = backward

class NN: # nerual network

    def __init__(self):
        self.edges = None
        self.gradW = None
        self.gradB = None
        self.newErrs = None
        self.neurons = set()
        self.inNeurons = set()
        self.outNeurons = set()
        self.hidNeurons = set()
        self.idx = 0

    def _cn(name): #create neuron
        def f(self, num=1, activation=Activation()):
            for i in range(num):
                n = eval(name[0].upper() + name[1:-1] +
                         "(activation)")
                getattr(self, name).add(n)
                self.neurons.add(n)
                n.idx = self.idx
                self.idx += 1
        return f

    createInNeuron = _cn("inNeurons")
    createHidNeuron = _cn("hidNeurons")
    createOutNeuron = _cn("outNeurons")

    def connect(self, n1, n2):
        n1 - n2

    def initEdges(self):
        n = len(self.neurons)
        self.edges = np.random.random([n, n]) * 0.001
        self.gradW = np.zeros([n, n])
        self.gradB = np.zeros(n)
        self.newErrs = np.zeros(n)
        
    def connectAll(self):
        for n1 in self.neurons:
            for n2 in self.neurons:
                self.connect(n1,n2)
        self.initEdges()

    def getW(self, n1, n2):
        return self.edges[n1.idx, n2.idx]

    def setW(self, n1, n2, w):
        self.edges[n1.idx, n2.idx] = w

    def fakeInit(self, n): # for debug
        self.edges.fill(n)
        for ne in self.hidNeurons:
            ne.b = n
            ne.bval = n
        for ne in self.outNeurons:
            ne.b = n
            ne.bval = n            
        
    def forward(self): # forward pass
        # TODO: change this to acommodate for batch mode
        newVals = np.zeros(len(self.neurons))
        for i, n in enumerate(self.neurons):
            for o in n.inlinks:
                newVals[i] += o.val * self.getW(o, n)
            #     print("getw", self.getW(o, n))
            # print("newval", n, newVals[i])
        for i, n in enumerate(self.neurons):
            n.bval = newVals[i]
            newVals[i] = n.activation.forward(newVals[i] + n.b)
            # print("b", n.b)
            # print("a act", n, newVals[i])
            n.val = newVals[i]

    def backward(self): # error gradient for input before activation
        # TODO: change this to acommodate for batch mode        
        newErrs = np.zeros(len(self.neurons))    
        for i, n in enumerate(self.neurons):
            for o in n.outlinks:
                newErrs[i] += o.err * self.getW(n, o)
            #     print("o.err", o,  o.err)
            #     print("getW", self.getW(n, o))
            # print("bErr", n, newErrs[i])                            
        for i, n in enumerate(self.neurons):
            # print("nErr", n, newErrs[i])            
            newErrs[i] *= n.activation.backward(n.bval)
            # print("nErr b", n, n.bval)                        
            # print("nErr a", n, newErrs[i])
            # for batch case
            self.newErrs[i] += newErrs[i]


    def backwardWeight(self): # update bias and weight
        # TODO: change this to acommodate for batch mode        
        for s in self.neurons:
            for t in s.outlinks:
                # += for accounting for batch
                self.gradW[s.idx, t.idx] += t.err * s.val
                # print("terr", s, t, t.err)
                # print("sval", s, t, s.val)                
            self.gradB[s.idx] += s.err
            # print(s, s.err)

    def update(self, lr=0.01):
        # update W, b
        for n in self.outNeurons:
            n.b -= lr * self.gradB[n.idx]
        for n in self.hidNeurons:
            n.b -= lr * self.gradB[n.idx]
        self.edges -= lr * self.gradW
        # update n.err
        for i, n in enumerate(self.neurons):
            n.err = self.newErrs[i]     
        # zero out grad and self.newErrs
        self.gradW.fill(0)
        self.gradB.fill(0)
        self.newErrs.fill(0)
        
    def forwardLoss(self, loss, target):
        # TODO: change this to acommodate for batch mode
        return loss.forward(self.val(self.outNeurons), target)

    def backwardLoss(self, loss, target):
        # TODO: change this to acommodate for batch mode
        l = loss.backward(self.val(self.outNeurons), target)
        # set to before activation: todo
        lb = np.array([n.activation.backward(n.bval) for n in self.outNeurons])
        self.eOut(lb * l)
    
    def train(self, data, loss, bs=1):
        if bs != 1:
            print("bs to be implemented, using bs=1")
            bs = 1
        # epoch starts
        epoch = 0
        while True:
            data.reset() # reset pointer in data
            data.shuffle()
            batch = data.getBatch(bs)
            while batch:
                bs_loss = 0
                for i in range(batch.n):
                    self.vIn(batch.x[i])
                    self.forward()
                    bs_loss += self.forwardLoss(loss, batch.y[i])
                    self.backwardLoss(loss, batch.y[i])
                    self.backward()
                    self.backwardWeight()                    

                # report loss
                bs_loss = bs_loss / batch.n
                print("e{:d} {} {} {} {} {}".format(epoch,
                                              data.progress(),
                                              bs_loss,
                                              batch.x[0],
                                              batch.y[0],
                                              self.val(self.outNeurons)))

                # time start
                self.update()
                batch = data.getBatch(bs)
            epoch += 1

    def test(self, data, loss, bs=64):
        pass
    
    # helper functions
    def val(self, l): # given list of neurons, give list of vals
        return np.array([o.val for o in l])
    
    def _v(name): # set value
        def f(self, vals):
            for i, n in enumerate(getattr(self, name)):
                n.val = vals[i]
        return f
    vIn = _v("inNeurons")
    vHid = _v("hidNeurons")
    vOut = _v("outNeurons")

    def _e(name): # set error
        def f(self, errs):
            for i, n in enumerate(getattr(self, name)):
                n.err = errs[i]
        return f
    eIn = _e("inNeurons")
    eHid = _e("hidNeurons")  
    eOut = _e("outNeurons")    

    def _p(name):
        def f(self):
            print("\t".join(["neuron", "val", "err", "bval", "idx"]))            
            for o in getattr(self, name):
                print("\t".join(map(lambda x: str(x), [o, o.val, o.err, o.bval, o.idx])))
        return f
    pIn = _p("inNeurons")
    pOut = _p("outNeurons")
    pHid = _p("hidNeurons")
        
class Neuron:
    i = 0
    def __init__(self, activation=Activation()):
        self.inlinks = set() # list of incoming neurons
        self.outlinks = set() # list of outgoing neurons
        self.activation = activation # activation function
        self.type = "g" # for general
        self.index = Neuron.i # unique index

        self.b = np.random.random() * 0.001 # bias
        self.val = 0 # signal
        self.bval = 0 # signal before activation
        self.err = 0 # gradient: dE/dN
        
        Neuron.i = Neuron.i + 1

    def connect(self, other):
        pass
    
    def __sub__(self, other): 
        self.connect(other)
        return self
    
    def __repr__(self):
        return self.type + str(self.index)

class InNeuron(Neuron): # no inlink
    def __init__(self, activation=Activation()):
        Neuron.__init__(self, activation)                
        self.type = 'i'
        self.b = 0
        
    def connect(self, other):
        assert(isinstance(other, Neuron))
        if (isinstance(other, InNeuron)): return
        self.outlinks.add(other)
        other.inlinks.add(self)

class HidNeuron(Neuron):
    def __init__(self, activation=Activation()):
        Neuron.__init__(self, activation)
        self.type = 'h'

    def connect(self, other):
        assert(isinstance(other, Neuron))
        if (isinstance(other, InNeuron)): other - self; return
        if (isinstance(other, OutNeuron)): other - self; return        
        self.inlinks.add(other)
        self.outlinks.add(other)
        other.inlinks.add(self)
        other.outlinks.add(self)
        
class OutNeuron(Neuron): # no outlink
    def __init__(self, activation=Activation()):
        Neuron.__init__(self, activation)        
        self.type = 'o'
    
    def connect(self, other):
        assert(isinstance(other, Neuron))
        if (isinstance(other, OutNeuron)): return
        self.inlinks.add(other)
        other.outlinks.add(self)

if __name__ == '__main__':
    # test.test()

    # 1) prediction
    # try identity problem
    idData = problem.Data(f=lambda x: 1 if x>500 else 0, n=1000)
    idData.shuffle()
    # activation
    f = lambda x:1/(1+e**(-x))
    sigmoid = Activation(f, lambda x: (1-f(x)) * f(x))
    # model to use
    n = NN()
    n.createInNeuron(1, sigmoid)
    n.createHidNeuron(10, sigmoid)
    n.createOutNeuron(1, sigmoid)
    n.connectAll()
    l = loss.se()
    # train
    n.train(idData.getTr(), l, 128)
    # n.test(idData.getTe(), l, 128)
    
