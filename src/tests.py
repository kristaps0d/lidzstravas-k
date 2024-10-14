import numpy as np
from src.definitions import *

# 10. Parbaudam:
print("[test]: M*N^t = 0\n", np.matmul(
    M,
    np.transpose(N)
), "\n")

print("[test]: M*J - I = 0\n", np.matmul(
    M,
    J
) - I, "\n")
