'''
Azad-Academy
Author: J. Rafid Siddiqui
jrs@azaditech.com
https://www.azaditech.com

'''

import numpy as np
import scipy.optimize as optimizer
from utils import *

class NLModel:

    def __init__(self, X, Y, mtype='poly' , use_regularization=False):
        
        self.X = X
        self.num_dims = self.X.shape[1]
        if(Y.ndim<2):
            Y = Y[:,np.newaxis]

        self.Y = Y
        self.J = 0
        self.G = 0
        self.type = mtype

        self.W = np.ones(5)   # W3*Sin(W1*X1 + W2*X2 + W0) + W4

        if(self.type=='poly'):
            self.W = np.zeros(7)    #  W0 + W1*X1 + W2*X1^2 + W3*X1^3 + W4*X2 + W5*X2^2 + W6*X2^3

        self.use_regularization = use_regularization

        
        

    def generate_polynomial(self,X):

        m = len(X)
        
        B = np.ones((m,1))
        
        X1 = (X[:,0]).reshape((m,1))
        X2 = (X[:,1]).reshape((m,1))

        X1_3 = np.power(X1,3).reshape((m,1))
        X1_2 = np.power(X1,2).reshape((m,1))
        
        X2_3 = np.power(X2,3).reshape((m,1))
        X2_2 = np.power(X2,2).reshape((m,1))            

        X = np.concatenate((B,X1,X2,X1_2,X1_3,X2_2,X2_3),axis=1)

        
        return X
    
    def gen_hypothesis(self,X,params):

        m = len(X)
        if(self.type == 'poly'):
            self.W = params.reshape((7,1))
            X = self.generate_polynomial(X)
            z =  X @ self.W

        else:
            self.W = params.reshape((5,1))
            X = np.concatenate((np.ones((m,1)),X),axis=1)
            z =  (self.W[3] * np.sin( 2*math.pi*(self.W[1]*X[:,1]) + (math.pi/3)*self.W[0]  ) + (self.W[2] * X[:,2]) + self.W[4]).reshape((m,1))                         
            
        Yp = sigmoid(z)

        return Yp,X,z

    def compute_cost(self, params, L=1):
        
        Y = self.Y
        m = len(self.X)      

        Yp,X,z = self.gen_hypothesis(self.X,params)
        
        if(self.use_regularization):
            Jreg = (L/(2*m))*np.sum(np.square(self.W[2:,]))
            J = -1/m*(np.sum(Y*np.log2(Yp)+(1-Y)*np.log2(1-Yp))) + Jreg
        else:
            J = -1/m*(np.sum(Y*np.log2(Yp)+(1-Y)*np.log2(1-Yp))) 

        self.J = J

        return J

    def compute_G(self,X):

        a = (2*math.pi)*(self.W[1]*X[:,1]) + (math.pi/3)*self.W[0]
        g0 = self.W[3] * np.cos(a) * (math.pi/3)
        g1 = self.W[3] * np.cos(a) * X[:,1] 
        g2 = X[:,2] #self.W[3] * np.cos(a) * X[:,2]
        g3 = np.sin(a)
        g4 = np.ones((len(X)))

        G = np.array([g0,g1,g2,g3,g4],dtype='float')
        return G

    def compute_gradient(self, params=None, L=1):

        m = len(self.X)
        Y = self.Y
        
        Yp,X,z = self.gen_hypothesis(self.X,params)
        
        if(self.use_regularization):
            W = self.W
            W[0,0] = 0
            G_reg = L/m * W

            if(self.type=='harmonic'):
                G = self.compute_G(X) @ (Yp-Y) + G_reg
            else:
                X = np.transpose(X)
                G = 1/m*(X @ (Yp-Y)) + G_reg
            
        else:
            if(self.type=='harmonic'):
                G = self.compute_G(X) @ (Yp-Y)
            else:
                X = np.transpose(X)
                G = 1/m*(X @ (Yp-Y))
        self.G = G
    
            
        return G.squeeze()

    def train(self,max_iters=100,L=1,verbose=False):

        options = {'maxiter' : '', 'disp':''}
        
        options['maxiter'] = max_iters
        options['disp'] = verbose

        P = optimizer.minimize(fun=self.compute_cost,x0=self.W,method='CG',jac=self.compute_gradient,args=(L),options=options)
        
        return P

    def predict(self, X,thresh=0.5,prob=False):
        
        Y_hat,X,z = self.gen_hypothesis(X,self.W)

        if(not prob):
            Y_hat = 1*(Y_hat > thresh)

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
                
                boundary_point = np.array([xx[j,i],yy[j,i]],dtype='float').reshape(1,2)
                Yp,X,z = self.gen_hypothesis(boundary_point,self.W)
                
                if(abs(z)<0.001):
                    boundary = np.append(boundary,np.array(boundary_point),axis=0)
        
        return boundary

    def show_decision_boundary(self,ax,cmap='Greys'):

        res = 0.02
        X = self.X
        x = np.arange(min(X[:,0]),max(X[:,0]),res)
        y = np.arange(min(X[:,1]),max(X[:,1]),res)
        xx, yy = np.meshgrid(x, y)
        
        X = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(X,prob=True)
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.2)
        ax.contour(xx, yy, Z, colors='k', linewidths=0.7)
        
        return 