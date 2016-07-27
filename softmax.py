""" 
    Softmax
    Converting the  scores to probabilities 
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')

scores = np.array([3.0, 1.0, 0.2])
def softmax(x):
    res = np.zeros_like(x)
    
    if len(x.shape) == 2:
        res = np.exp(x)/np.exp(x).sum(axis=1, keepdims=True)
    else:
        res = np.exp(x)/np.exp(x).sum(axis=0, keepdims=True)
    return res

#print(softmax(scores))

#plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x/10, np.ones_like(x), 0.2*np.ones_like(x)])
#print(softmax(scores).T)
# scores = np.vstack([x*10, np.ones_like(x), 0.2*np.ones_like(x)]) 
plt.plot(x, softmax(scores).T, linewidth=1)
plt.savefig('softmax.png')