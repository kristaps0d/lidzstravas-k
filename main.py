import numpy as np
from src.definitions import *
import src.zeidmet as zeidmet

# 14-10-2024
# Atrast:
#   1. U mezglos
#   2. I zaros

G = np.matmul(
    M,
    np.matmul(
        np.linalg.inv(np.diag(R.reshape(-1))),
        np.transpose(M)
    )
)

U = np.matmul(
    np.linalg.inv(G),
    (I - np.matmul(
        M,
        np.matmul(
            np.linalg.inv(np.diag(R.reshape(-1))),
            E
        )
    ))
)

I_z = np.matmul(
    np.linalg.inv(np.diag(R.reshape(-1))),
    (E + np.matmul(
        np.transpose(M),
        U
    ))
)

R_k = np.matmul(
    N,
    np.matmul(
        np.diag(R.reshape(-1)),
        np.transpose(N)
    )
)

E_k = np.matmul(
    N,
    np.matmul(
        (E - np.diag(R.reshape(-1))),
        J
    )
)

I_k = zeidmet.Calculate(
    R_k, 
    np.random.rand(R_k.shape[1], E_k.shape[1]), 
    E_k
)

I_z_k = np.matmul(
    np.transpose(N),
    I_k
) + J