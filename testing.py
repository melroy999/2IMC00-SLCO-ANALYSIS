import numpy as np

A = 0.8 * np.random.rand(4, 4)
A[0, 0] = 1
A[1, 1] = 1
A[2, 2] = 1
A[3, 3] = 1
det = np.linalg.det(A)
print(det)
