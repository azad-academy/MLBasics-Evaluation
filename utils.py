'''
Azad-Academy
Author: J. Rafid Siddiqui
jrs@azaditech.com
https://www.azaditech.com

'''

import sys
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import math
from itertools import permutations
import matplotlib

import sklearn
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


from IPython.display import display


def plot_data(X,Y,canvas=None,xtitle=None,ytitle=None,colors=None,plt_title=None,color_map=plt.cm.RdBu):
            
    if(colors is None):
        colors = np.random.rand(max(Y)+1,3)    
        
    if(canvas is None):
        fig, ax = plt.subplots(figsize=(11,8))
    else:
        ax = canvas        
    
    if(plt_title is not None):
        ax.set_title(plt_title)  
    
    
    if(X.shape[1]>2):
        ax.scatter3D(X[:,0],X[:,1],X[:,2],color=np.array(colors)[Y],alpha=0.6)  #plotting the 3D points
        ax.grid(False)
    else:
        ax.scatter(X[:,0],X[:,1],color=np.array(colors)[Y],alpha=0.6,cmap=color_map)  #plotting the 2D points
            
    if(xtitle is not None):
        ax.set_xlabel(xtitle,fontweight='bold',fontsize=16)
    
    if(xtitle is not None):
        ax.set_ylabel(ytitle,fontweight='bold',fontsize=16)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

       
def sigmoid(z):
    return 1/(1 + np.exp(-z))
def sigmoid_grad(z):
    return sigmoid(z)*(1-sigmoid(z))