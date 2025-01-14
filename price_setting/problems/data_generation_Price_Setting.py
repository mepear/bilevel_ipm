import numpy as np

def generate_problem_data(n, seed=0):
    np.random.seed(seed)
    num_constraints_1 = 20
    
    A = np.abs(np.random.randn(num_constraints_1, n))
    B = np.abs(np.random.randn(num_constraints_1, n))
    for i in range(num_constraints_1):
        if np.all(A[i] <= 1):
            max_idx = np.argmax(A[i])
            A[i, max_idx] = A[i, max_idx] + (1 - A[i, max_idx]) + 0.1

    for i in range(num_constraints_1):
        if np.all(B[i] <= 1):
            max_idx = np.argmax(B[i])
            B[i, max_idx] = B[i, max_idx] + (1 - B[i, max_idx]) + 0.1

    B = B * 0.2

    b = np.abs(np.random.randn(num_constraints_1))
    
    c_1 = np.ones(60)
    c_2 = np.ones(60)
    
    E_matrices_2 = [-np.eye(n), np.eye(n), -np.eye(n), np.eye(n)]

    g_scalars_2 = 100 * [np.zeros(n), np.ones(n),  np.zeros(n), np.ones(n)]

    
    return {
        'A': A,
        'B': B,
        'b': b,
        'c_1': c_1,
        'c_2': c_2,
        'E_matrices_2': E_matrices_2,
        'g_scalars_2': g_scalars_2,
        'n': n,
        'num_constraints_1': num_constraints_1
    }
