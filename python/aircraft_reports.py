#  Created by Luis Alejandro (alejand@umich.edu)
import numpy as np
import matplotlib.pyplot as plt

def display_random_subset(X,y,count=9,order='C',shape=(20,34),labels=[]):
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
            ax.pcolormesh(X[indexes[i],:].reshape((shape),order=order), cmap='magma')
            ax.set_xticks(())
            ax.set_yticks(())
            if len(labels) > 0:
                ax.set_title(labels[y[indexes[i]]])
        else:
            fig.delaxes(ax)    
            
def display_class_subset(X,y,category,count=9,order='C',shape=(20,34),labels=None):
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
            ax.matshow(X[indexes[i],:].reshape((shape),order=order), cmap='winter')
            ax.set_xticks(())
            ax.set_yticks(())
            if len(labels) > 0:
                ax.set_title(labels[category])
        else:
            fig.delaxes(ax)    