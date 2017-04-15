import numpy as np
import theano
import theano.tensor as tensor

ctx_mean = tensor.matrix('ctx_mean', dtype='float32')
attr = tensor.matrix('attr', dtype='float32')
W = tensor.matrix('w', dtype='float32')
c2 = ((tensor.dot(ctx_mean, W) - attr)**2).sum(axis=1)

f = theano.function([ctx_mean, attr, W], c2, allow_input_downcast=True)

ctx = np.asmatrix([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])
attr = np.asmatrix([[1, 0, 0], [0, 1, 1]])
W = np.asmatrix([[0, 1, 1, 1, 1, 0, 0, 1, 1, 1], [1, 1, 1, 0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 1, 0, 1]]).T

x = f(ctx, attr, W)
print x