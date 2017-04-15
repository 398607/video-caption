import numpy as np
import theano
import theano.tensor as tensor

a = np.zeros((64, 1536))
b = np.zeros((1536, 100))
c1 = np.dot(a, b).sum(axis=1)
c2 = tensor.dot(a, b).sum(axis=1)
print c1.shape

