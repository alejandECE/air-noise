#  Created by Luis Alejandro (alejand@umich.edu)
import numpy as np
import matplotlib.pyplot as plt

def display_random_subset(X,y,count=9,order='C',mfcc=20):
    indexes = np.random.permutation(X.shape[0])
    if count > len(indexes):
        count = len(indexes)
    if count <= 0:
        return
    s = np.sqrt(count)
    m = int(np.floor(s))
    n = int(np.ceil(s))
    fig, axes = plt.subplots(n,m)
    fig.set_size_inches(4*n,4*m)
    fig.set_tight_layout(tight=0.1)
    for i,ax in enumerate(axes.ravel()):
        if i <= count:
            ax.pcolormesh(X[indexes[i],:].reshape((mfcc,-1),order=order), cmap='magma')
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(y[indexes[i]])
        else:
            fig.delaxes(ax)    
            
def display_class_subset(X,y,category,count=9,order='C',mfcc=20):
    indexes = (y == category).nonzero()
    indexes = np.random.permutation(indexes)[0]
    if count > len(indexes):
        count = len(indexes)
    if count <= 0:
        return
    s = np.sqrt(count)
    m = int(np.floor(s))
    n = int(np.ceil(s))
    fig, axes = plt.subplots(n,m)
    fig.set_size_inches(4*n,4*m)
    fig.set_tight_layout(tight=0.1)
    for i,ax in enumerate(axes.ravel()):
        if i <= count:
            ax.pcolormesh(X[indexes[i],:].reshape((mfcc,-1),order=order), cmap='magma')
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(category)
        else:
            fig.delaxes(ax)    