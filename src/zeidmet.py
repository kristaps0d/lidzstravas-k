import numpy as np

# https://github.com/kristaps0d/zidmet/blob/main/calc.py
# Simplified reimplementation
def Calculate(A, X, B, target_error:float=0.00001):
    U = np.triu(A, k=1)
    L = np.tril(A, k=0)

    while True:
        N_X = np.matmul(
            np.linalg.inv(L),
            (B - np.matmul(U, X))
        )

        MAE = np.max(np.abs(N_X - X))

        if MAE <= target_error:
            return N_X
        
        X = N_X