import numpy as np
import random

def grad_U(Ui, Yij, Vj, reg, eta):
    grad_u = reg * Ui - (Yij - Ui * Vj.T) * Vj
    return eta * grad_u

def grad_V(Vj, Yij, Ui, reg, eta):
    grad_v = reg * Vj - (Yij - Ui * Vj.T) * Ui
    return eta * grad_v

def get_err(U, V, Y, reg=0.0):
    error = 0
    for i in range(Y.shape[0]):
        Y_hat = U[Y[i][0]-1] * V[Y[i][1]-1].T
        error += (Y[i][2] - Y_hat[0][0])**2
    error = np.sqrt(1 / Y.shape[0] * error) + reg/2 * (np.linalg.norm(U, ord = "fro") + np.linalg.norm(V, ord = "fro"))
    return error.A1[0]

def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300):
    U = np.matrix(np.random.uniform(-0.5, 0.5, size = (M, K)))
    V = np.matrix(np.random.uniform(-0.5, 0.5, size = (N, K)))
    error = float("inf")
    for epoch in range(max_epochs):
        listY = list(range(Y.shape[0]))
        random.shuffle(listY)
        for i in listY:
            U[Y[i][0]-1] -= grad_U(U[Y[i][0]-1], Y[i][2], V[Y[i][1]-1], reg, eta)
            V[Y[i][1]-1] -= grad_V(V[Y[i][1]-1], Y[i][2], U[Y[i][0]-1], reg, eta)
        error_new = get_err(U, V, Y, reg)
        print("epoch:", epoch+1, ", error:", error_new)
        if epoch == 0: error_1 = error_new
        if epoch == 1: delta_0_1 = np.abs(error_1 - error_new)
        delta_i = error - error_new
        if epoch > 1 and delta_i / delta_0_1 < eps:
            break
        error = error_new
    return U, V, get_err(U, V, Y, reg=0)
