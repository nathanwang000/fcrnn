from neuron import *

def test():
    def m(m): print(m * 40)
    
    n = NN()
    n.createInNeuron()
    n.createOutNeuron()
    n.createHidNeuron()
    n.connectAll()

    # test for foward prop
    m('f')
    n.fakeInit(1)
    n.vIn([1])
    n.vHid([0])
    n.vOut([0])        
    n.forward()
    n.pOut() # val = 2 b/c w = 1, b = 1
    n.forward()
    n.pOut() # val = 3
    n.pHid()

    # test for backward
    m('b')
    n.fakeInit(1)
    n.vIn([0])
    n.vHid([0])
    n.vOut([0])
    n.eOut([100])

    n.pOut()
    n.backward()
    n.pIn()
    n.pHid() # err = 100      
    n.pOut()
    n.backward()
    n.pIn()
    n.pHid() # err = 100      
    n.pOut()
    n.backward()
    n.pIn()
    n.pHid() # err = 100      
    n.pOut()

    # test for backward weight
    m('c')
    n.fakeInit(1)
    n.vIn([1])
    n.vHid([0])
    n.vOut([0])
    n.eOut([100])
    n.eHid([0])
    n.eIn([0])    
    n.backwardWeight()
    print(n.gradW)
    print(n.gradB)
