import numpy as np
from Activation import relu, sigmoid
from problem import Data
from loss import se
# for dense case
from scipy.linalg import block_diag as bd
# for sparse case: actually just rewrite how step works
# from scipy.sparse import csr_matrix, block_diag as bd

class FCNN: # fully connected nueral network
    def __init__(self, nin, nhid, nout, \
                 act=sigmoid, time=10, bs=64):
        self.nin = nin
        self.nhid = nhid
        self.nout = nout
        self.bs = bs
        self.time = time # maximum depth of the network
        assert(time > 0)

        # values
        self.I = np.zeros((self.nin, self.bs))
        self.H = np.zeros((self.nhid, self.bs))
        self.O = np.zeros((self.nout, self.bs))

        # connections: m for mask, g for gradient, b for bias
        self.mIH = np.ones((self.nhid, self.nin))
        self.mIO = np.ones((self.nout, self.nin))
        self.mHH = np.ones((self.nhid, self.nhid))
        self.mHO = np.ones((self.nout, self.nhid))
        
        self.IH = np.random.random((self.nhid, self.nin))
        self.IO = np.random.random((self.nout, self.nin))
        self.HH = np.random.random((self.nhid, self.nhid))
        self.HO = np.random.random((self.nout, self.nhid))

        self.bH = np.random.random((self.nhid,1))        
        self.bO = np.random.random((self.nout,1))

        # gradient
        self.resetGrad()

        # activation
        self.Hact = act

    def resetGrad(self):
        self.gIH = np.zeros((self.nhid, self.nin))
        self.gIO = np.zeros((self.nout, self.nin))
        self.gHH = np.zeros((self.nhid, self.nhid))
        self.gHO = np.zeros((self.nout, self.nhid))
        
        self.gbH = np.zeros((self.nhid,1))
        self.gbO = np.zeros((self.nout,1))
        
        # sparse zero matrix
        self._gIH = np.zeros((self.nhid, self.nin * self.nhid,\
                              self.bs))
        self._gIO = np.zeros((self.nout, self.nin * self.nout,\
                              self.bs))
        self._gHH = np.zeros((self.nhid, self.nhid * self.nhid,\
                              self.bs))
        self._gHO = np.zeros((self.nout, self.nhid * self.nout,\
                              self.bs))
        self._gbH = np.zeros((self.nhid, self.nhid, self.bs))
        
    def update(self, lr=0.01):
        params = [self.IH, self.IO, self.HH,
                  self.HO, self.bH, self.bO]
        gparams = [self.gIH, self.gIO, self.gHH,
                   self.gHO, self.gbH, self.gbO]
        # update W, b
        for p, gp in zip(params, gparams):
            p -= lr * gp
        # reset gradient
        self.resetGrad()

    def openErrGate(self, batch, loss):
        # TODO
        # run loss.forward on output to get l
        target = np.zeros_like(self.O)
        for i in range(len(batch.y)):
            target[:,i] = batch.y[i].reshape(target[:,i].shape)
        l = loss.forward(self.O, target)
        # run loss.backward on output to get dE/dO
        # error = loss.backward(self.O, target).\
        #         reshape((1,self.nout,self.bs))
        error = np.zeros((1,self.nout,self.bs))
        # assume row major
        for i in range(self.bs):
            err = error[...,i]
            self.gbH += err.dot(self._gbH[...,i]).\
                        reshape(self.gbH.shape)
            self.gbO += err.reshape(self.gbO.shape)
            self.gIH += err.dot(self._gIH[...,i]).\
                        reshape(self.gIH.shape)
            self.gIO += err.dot(self._gIO[...,i]).\
                        reshape(self.gIO.shape)
            self.gHH += err.dot(self._gHH[...,i]).\
                        reshape(self.gHH.shape)
            self.gHO += err.dot(self._gHO[...,i]).\
                        reshape(self.gHO.shape)
        return l
        
    def inputFeeder(self, batch):
        # TODO: gracefully handle last batch!
        def feed():
            I = np.zeros((self.nin, self.bs))
            for i in range(len(batch.x)):
                if self.t < len(batch.x[i]):
                    I[:,i] = batch.x[i][self.t].\
                             reshape(I[:,i].shape)
            self.I = I
        return feed
    
    def grow(self, data, loss):  # forward time timestep
        self.t = 0
        feed = self.inputFeeder(data)
        while self.t < self.time:
            feed()
            self.step()
            self.t += 1
        # freeze time
        # MAYBE: handle variable length output        
        feed()
        self.freezeStep()
        l = self.openErrGate(data, loss)
        self.update()
        return l

    def freezeStep(self):
        _gIH = np.zeros((self.nout, self.nin*self.nhid, self.bs))
        _gHH = np.zeros((self.nout, self.nhid*self.nhid, self.bs))
        _gbH = np.zeros((self.nout, self.nhid, self.bs))
        
        for i in range(self.bs):
            H, I = self.H[:,i], self.I[:,i]
            self._gHO[...,i] = bd(*([H] * self.nout))
            self._gIO[...,i] = bd(*([I] * self.nout))
            # MAYBE: handle variable length output            
            _gIH[...,i] = self.HO.dot(self._gIH[...,i])
            _gHH[...,i] = self.HO.dot(self._gHH[...,i])
            _gbH[...,i] = self.HO.dot(self._gbH[...,i])
        self._gIH, self._gHH, self._gbH = _gIH, _gHH, _gbH
        
    def step(self): # forward 1 timestep

        def forward():
            nI = np.zeros_like(self.I)
            nH = self.IH.dot(self.I) + self.HH.dot(self.H) + self.bH
            nO = self.IO.dot(self.I) + self.HO.dot(self.H) + self.bO
            self.I, self.H, self.O = nI, self.Hact.forward(nH), nO

        def accGrad():
            for i in range(self.bs):
                H, I = self.H[...,i], self.I[...,i]
                hback = np.diag(self.Hact.backward(H))
                self._gHH[...,i] = hback.dot\
                                   (self.HH.dot(self._gHH[...,i]))+\
                                   hback.dot\
                                   (bd(*([H] * self.nhid)))
                self._gIH[...,i] = hback.dot\
                                   (self.HH.dot(self._gIH[...,i]))+\
                                   hback.dot\
                                   (bd(*([I] * self.nhid)))
                self._gbH[...,i] = hback.dot\
                                   (self.HH.dot(self._gbH[...,i]))+\
                                   hback.dot\
                                   (np.eye(self.nhid))
        
        accGrad()
        forward()

    def train(self, data, loss):
        epoch = 0
        while True:
            data.reset() # reset pointer in data
            data.shuffle()
            batch = data.getBatch(self.bs)
            while batch:
                l = self.grow(batch, loss)
                print("e{:d} {} {}".format(epoch,
                                           data.progress(),
                                           l))
                batch = data.getBatch(self.bs)
            epoch += 1
        

if __name__ == '__main__':
    # model
    model = FCNN(1,30,1,sigmoid,time=30,bs=1)
    # loss
    loss = se()
    # data
    n = 2
    data = Data(f=lambda x: [1] if x>n/2 else [0], n=n)
    data.shuffle()
    # train
    model.train(data.getTr(), loss)