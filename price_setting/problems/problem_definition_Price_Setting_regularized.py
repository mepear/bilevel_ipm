import cvxpy as cp
import numpy as np

class BilevelProblem_regularized:
    def __init__(self, data):
        self.A = data['A']
        self.B = data['B']
        self.b = data['b']
        self.c_1 = data['c_1']
        self.c_2 = data['c_2']
        self.E_matrices_2 = data['E_matrices_2']
        self.g_scalars_2 = data['g_scalars_2']
        self.n = data['n']
        self.num_constraints_h1 = data['num_constraints_1']
        self.num_constraints_h2 = 4 * self.n
        self.kappa = 0.01

    def f(self, T, x, y):
        return -T.T @ x

    def g(self, T, x, y):
        return (self.c_1 + T).T @ x + self.c_2.T @ y + self.kappa * x.T @ x + self.kappa * y.T @ y

    def h_1(self, T, x, y, i):
        return self.b[i] - self.A[i] @ x - self.B[i] @ y

    def h_2(self, T, x, y, i):
        constraint_type = i // self.n
        index = i % self.n

        g = self.g_scalars_2[1]
    
        if constraint_type == 0:
            return -x[index]
        elif constraint_type == 1:
            return x[index] - g[index]
        elif constraint_type == 2:
            return -y[index]
        elif constraint_type == 3:
            return y[index] - g[index]

    def gradient_f_xy(self, T, x, y):
        grad_x = -T.copy()
        grad_y = -np.zeros_like(y)
        return grad_x, grad_y

    def gradient_f_T(self, T, x, y):
        grad_T = -x.copy()
        return grad_T

    def gradient_g_xy(self, T, x, y):
        grad_x = self.c_1 + T + 2 * self.kappa * x
        grad_y = self.c_2 + 2 * self.kappa * y
        return grad_x, grad_y

    def gradient_g_T(self, T, x, y):
        grad_T = x.copy()
        return grad_T

    def gradient_h_1_xy(self, T, x, y, i):
        grad_x = -self.A[i]
        grad_y = -self.B[i]
        return grad_x, grad_y

    def gradient_h_1_T(self, T, x, y, i):
        return np.zeros_like(T)

    def gradient_h_2_T(self, T, x, y, i):
        return np.zeros_like(T)

    def gradient_h_2_xy(self, T, x, y, i):
        constraint_type = i // self.n
        index = i % self.n
        grad_x = np.zeros_like(x)
        grad_y = np.zeros_like(y)
        if constraint_type == 0 or constraint_type == 1:
            grad_x[index] = self.E_matrices_2[constraint_type][index, index]
        elif constraint_type == 2 or constraint_type == 3:
            grad_y[index] = self.E_matrices_2[constraint_type][index, index]
        return grad_x, grad_y

    def hessian_f_T_xy(self, T, x, y):
        I = -np.eye(self.n)
        zeros = np.zeros((self.n, self.n))
        return np.hstack((I, zeros))

    def hessian_f_xy_xy(self, T, x, y):
        return np.zeros((2 * self.n, 2 * self.n))
    
    def hessian_g_T_xy(self, T, x, y):
        I = np.eye(self.n)
        zeros = np.zeros((self.n, self.n))
        return np.hstack((I, zeros))

    def hessian_g_xy_xy(self, T, x, y):
        H = np.zeros((2 * self.n, 2 * self.n))
        H[:self.n, :self.n] = 2 * self.kappa * np.eye(self.n)
        H[self.n:, self.n:] = 2 * self.kappa * np.eye(self.n)
        return H

    def hessian_h_1_T_xy(self, T, x, y, i):
        return np.zeros((self.n, 2*self.n))

    def hessian_h_1_xy_xy(self, T, x, y, i):
        return np.zeros((2 * self.n, 2 * self.n))

    def hessian_h_2_xy_xy(self, T, x, y, i):
        return np.zeros((2 * self.n, 2 * self.n))


