'''
Azad-Academy
Author: J. Rafid Siddiqui
jrs@azaditech.com
https://www.azaditech.com

'''

import numpy as np
import scipy.optimize as optimizer
from utils import *

class LModel:

    def __init__(self, X, Y , use_regression=False):
        
        self.X = X
        self.num_dims = self.X.shape[1]
        if(Y.ndim<2):
            Y = Y[:,np.newaxis]

        self.Y = Y
        self.J = 0
        self.G = 0
        #self.W = np.random.rand(self.num_dims+1)
        self.W = np.zeros(self.num_dims+1)
        self.use_regression = use_regression

    def compute_cost(self, params=None , L=1):
        
        Y = self.Y
        m = len(self.X)

        if(params is not None):
            self.W = params.reshape((self.num_dims+1,1))
        else:
            self.W = self.W.reshape((self.num_dims+1,1))

        X = np.concatenate((np.ones((m,1)),self.X),axis=1)

        z =  X @ self.W
        s = sigmoid(z)
        

        if(self.use_regression):
            Jreg = (L/(2*m))*np.sum(np.square(self.W[2:,]))
            J = (-1/m*(np.sum(Y*np.log2(s)+(1-Y)*np.log2(1-s)))) + Jreg
        else:
            J = (-1/m*(np.sum(Y*np.log2(s)+(1-Y)*np.log2(1-s)))) 

        self.J = J

        return J


    def compute_gradient(self, params=None, L=1):

        m = len(self.X)

        if(params is not None):
            self.W = params.reshape((self.num_dims+1,1))
        else:
            self.W = self.W.reshape((self.num_dims+1,1))

        X = np.concatenate((np.ones((m,1)),self.X),axis=1)
        Y = self.Y
        W = self.W
        
        W[0,0] = 0
        
        z =  X @ self.W
        s = sigmoid(z)

        if(self.use_regression):
            G_reg = L/m * W
            G = 1/m*(np.transpose(X) @ (s-Y)) + G_reg
        else:
            G = 1/m*(np.transpose(X) @ (s-Y))
        self.G = G
        return G.squeeze()

    def train(self,max_iters=100,L=1,verbose=False):

        options = {'maxiter' : '', 'disp':''}
        
        options['maxiter'] = max_iters
        options['disp'] = verbose

        P = optimizer.minimize(fun=self.compute_cost,x0=self.W,method='CG',jac=self.compute_gradient,args=(L),options=options)
        
        return P

    def predict(self, X, thresh=0.5):
        
        m = len(X)
        X = np.concatenate((np.ones((m,1)),X),axis=1)
        z =  X @ self.W
        s = sigmoid(z)
        Y_hat = 1*(s > thresh)

        return Y_hat.squeeze()

    def get_decision_boundary(self):

        num_pts = 100
        X = self.X
        x = np.linspace(min(X[:,0]),max(X[:,0]),num_pts)
        y = np.linspace(min(X[:,1]),max(X[:,1]),num_pts)
        xx, yy = np.meshgrid(x, y)
        
        boundary = np.empty((0,2),float)
        for i in range(num_pts):
            for j in range(num_pts):
                
                boundary_point = np.array([1,xx[j,i],yy[j,i]],dtype='float').reshape(1,3)
                z =  boundary_point @ self.W
                                
                if(abs(z)<0.01):
                    boundary = np.append(boundary,np.array(boundary_point[:,1:]),axis=0)
        
        return boundary

