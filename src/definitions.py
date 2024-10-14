import numpy as np

# Varianta nr. 85
E1, E2, E3 = 5.39, 17.3, 7.06
R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12 = 81, 46.7, 51.9, 20.3, 58.1, 59.8, 65.7, 58.1, 51.9, 20.3, 58.1, 46.7
J1, J2 = 0.103, 4.11

#
M = np.array([
    [-1, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -1, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, -1, 0, 0, 0],
    [0, 0, 0, -1, -1, 0, -1, -1, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, -1]
])

N = np.array([
    [-1, 1, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, -1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -1, 0, -1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, -1, 0, 1, -1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1]
])

E = np.array([
    [0],
    [0],
    [E1],
    [0],
    [0],
    [E2],
    [0],
    [0],
    [0],
    [E3],
    [0],
    [0]
])

R = np.array([
    [R1],
    [R2],
    [R3],
    [R4],
    [R5],
    [R6],
    [R8],
    [R7],
    [R9],
    [R11],
    [R12],
    [R10]
])

I = np.array([
    [0],
    [J1],
    [0],
    [-J2],
    [-J1],
    [J2]
])

J = np.array([
    [0],
    [J1],
    [0],
    [0],
    [J1],
    [0],
    [0],
    [0],
    [J2],
    [0],
    [0],
    [0]
])