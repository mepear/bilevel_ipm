import numpy as np

class BilevelProblem:
    def __init__(self, data):
        self.A = data['A']
        self.b = data['b']
        self.C = data['C']
        self.d = data['d']
        self.D = data['D']
        self.E_matrices_1 = data['E_matrices_1']
        self.g_scalars_1 = data['g_scalars_1']
        self.E_matrices_2 = data['E_matrices_2']
        self.g_scalars_2 = data['g_scalars_2']
        self.n = data['n']
        self.num_constraints_h1 = data['num_constraints_1']
        self.num_constraints_h2 = 2 * self.n

    def f(self, x, y):
        z = np.concatenate([x, y])
        return 0.5 * z.T @ self.A @ z + self.b.T @ z

    def g(self, x, y):
        return 0.5 * y.T @ self.C @ y + self.d.T @ y + x.T @ self.D @ y

    def h_1(self, x, y, i):
        return x.T @ self.E_matrices_1[i] @ y - self.g_scalars_1[i]

    def h_2(self, x, y, i):
        constraint_type = i // self.n
        index = i % self.n
        return self.E_matrices_2[constraint_type][index, index] * y[index] - self.g_scalars_2[constraint_type][index]

    def gradient_f_x(self, x, y):
        z = np.concatenate([x, y])
        grad_f = self.A @ z + self.b
        return grad_f[:self.n]

    def gradient_f_y(self, x, y):
        z = np.concatenate([x, y])
        grad_f = self.A @ z + self.b
        return grad_f[self.n:]

    def gradient_g_x(self, x, y):
        return self.D @ y

    def gradient_g_y(self, x, y):
        return self.C @ y + self.d + self.D.T @ x

    def gradient_h_1_x(self, x, y, i):
        return self.E_matrices_1[i] @ y

    def gradient_h_1_y(self, x, y, i):
        return self.E_matrices_1[i].T @ x

    def gradient_h_2_x(self, x, y, i):
        return np.zeros_like(x)

    def gradient_h_2_y(self, x, y, i):
        constraint_type = i // self.n
        index = i % self.n
        grad = np.zeros(self.n)
        grad[index] = self.E_matrices_2[constraint_type][index, index]
        return grad

    def hessian_g_xy(self, x, y):
        return self.D

    def hessian_g_yy(self, x, y):
        return self.C
    
    def hessian_h_1_xy(self, x, y, i):
        return self.E_matrices_1[i]

    def hessian_h_1_yy(self, x, y, i):
        return np.zeros((self.n, self.n))

    def hessian_h_2_yy(self, x, y, i):
        return np.zeros((self.n, self.n))


