import numpy as np
z = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
A = np.exp(z)/sum(np.exp(z))
print(A)