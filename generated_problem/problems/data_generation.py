import numpy as np

def generate_problem_data(n, seed=0):
    
    np.random.seed(seed)
    
    A = np.random.randn(2*n, 2*n)
    A = 0.5 * (A + A.T) + 2*n * np.eye(2*n)
    b = np.random.randn(2*n)
    
    C = np.random.randn(n, n)
    C = 0.5 * (C + C.T) + n * np.eye(n)
    d = np.random.randn(n)
    D = np.random.randn(n, n) * 10
    
    num_constraints_1 = 50
    E_matrices_1 = [np.random.randn(n, n) for _ in range(num_constraints_1)]
    g_scalars_1 = np.random.rand(num_constraints_1)
    
    E_matrices_2 = [np.eye(n), -np.eye(n)]
    g_scalars_2 = [100 * np.ones(n), 100 * np.ones(n)]
    
    return {
        'A': A,
        'b': b,
        'C': C,
        'd': d,
        'D': D,
        'E_matrices_1': E_matrices_1,
        'g_scalars_1': g_scalars_1,
        'E_matrices_2': E_matrices_2,
        'g_scalars_2': g_scalars_2,
        'n': n,
        'num_constraints_1': num_constraints_1
    }
