import numpy as np

test = np.arange(25).reshape(5,5)

print(test)

def sqrnorm(v):
    return np.einsum('...i,...i', v, v)

print(sqrnorm(test))
