import numpy as np
import cvxpy as cp
import time

class BSG:
    def __init__(self, problem, hparams):
        self.problem = problem
        BSG_params = hparams.get('BSG', {})

        self.alpha_x = BSG_params.get('alpha_x', 1e-4)
        self.alpha_y = BSG_params.get('alpha_y', 1e-4)
        self.alpha_z = BSG_params.get('alpha_z', 1e-4)
        self.epsilon_x = BSG_params.get('epsilon_x', 1e-4)
        self.epsilon_y = BSG_params.get('epsilon_y', 1e-4)
        self.epsilon_z = BSG_params.get('epsilon_z', 1e-4)
        self.max_iters_x = BSG_params.get('max_iters_x', 10000)
        self.max_iters_y = BSG_params.get('max_iters_y', 10000)
        self.max_iters_z = BSG_params.get('max_iters_z', 10000)
        
    def project_to_constraints(self, x, y_init):
        n = self.problem.n
        y = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(y - y_init))
        constraints = []

        for i in range(self.problem.num_constraints_h1):
            constraints.append(self.problem.h_1(x, y, i) <= 0)
        for i in range(self.problem.num_constraints_h2):
            constraints.append(self.problem.h_2(x, y, i) <= 0)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP)
        return y.value
    
    def constraint_vector(self, x, y):
        constraints = []
    
        for i in range(self.problem.num_constraints_h1):
            constraints.append(self.problem.h_1(x, y, i))
    
        for i in range(self.problem.num_constraints_h2):
            constraints.append(self.problem.h_2(x, y, i))
    
        constraint_vector = np.array(constraints)
        return constraint_vector

    def compute_inner_product_y(self, x, y, z_vector):
        num_constraints_h1 = self.problem.num_constraints_h1
        num_constraints_h2 = self.problem.num_constraints_h2
        n = self.problem.n
        total_constraints = num_constraints_h1 + num_constraints_h2
        
        H_y = np.zeros((total_constraints, n))
        
        for i in range(num_constraints_h1):
            H_y[i, :] = self.problem.gradient_h_1_y(x, y, i)
        
        for i in range(num_constraints_h2):
            H_y[num_constraints_h1 + i, :] = self.problem.gradient_h_2_y(x, y, i)
        
        z_grad_y = H_y.T @ z_vector
        
        return z_grad_y

    def hessian_inner_product_h_z_xy(self, x, y, z):
        hessian = np.zeros((self.problem.n, self.problem.n))

        for i in range(self.problem.num_constraints_h1):
            hessian_matrix = self.problem.hessian_h_1_xy(x, y, i)
            hessian += hessian_matrix * z[i]

        return hessian
    
    def hessian_inner_product_h_z_yy(self, x, y, z):
        hessian = np.zeros((self.problem.n, self.problem.n))

        for i in range(self.problem.num_constraints_h1):
            hessian += self.problem.hessian_h_1_yy(x, y, i) * z[i]

        return hessian

    def compute_z_hadamard_grad_x_h(self, x, y, z):
        num_constraints = self.problem.num_constraints_h1 + self.problem.num_constraints_h2
        H_x = np.zeros((num_constraints, self.problem.n))

        for i in range(self.problem.num_constraints_h1):
            H_x[i, :] = self.problem.gradient_h_1_x(x, y, i)

        for i in range(self.problem.num_constraints_h2):
            H_x[self.problem.num_constraints_h1 + i, :] = self.problem.gradient_h_2_x(x, y, i)

        H_x_T = H_x.T

        z = z.reshape(1, -1)
        hadamard_matrix = H_x_T * z

        return hadamard_matrix
    
    def compute_z_hadamard_grad_y_h(self, x, y, z):
        num_constraints = self.problem.num_constraints_h1 + self.problem.num_constraints_h2
        H_y = np.zeros((num_constraints, self.problem.n))

        for i in range(self.problem.num_constraints_h1):
            H_y[i, :] = self.problem.gradient_h_1_y(x, y, i)

        for i in range(self.problem.num_constraints_h2):
            H_y[self.problem.num_constraints_h1 + i, :] = self.problem.gradient_h_2_y(x, y, i)

        H_y_T = H_y.T

        z = z.reshape(1, -1)
        hadamard_matrix = H_y_T * z

        return hadamard_matrix
    
    def G_x(self, x_init, y_init, z_init):
        x = x_init.copy()
        y = y_init.copy()
        z = z_init.copy()
        hessian_inner_product_h_z_xy = self.hessian_inner_product_h_z_xy(x, y, z)
        L_xy = self.problem.hessian_g_xy(x, y) + hessian_inner_product_h_z_xy
        hadamard_matrix_z_x = self.compute_z_hadamard_grad_x_h(x, y, z)
        concatenated_matrix = np.hstack((L_xy, hadamard_matrix_z_x))

        return concatenated_matrix
    
    def G_v(self, x_init, y_init, z_init):
        x = x_init.copy()
        y = y_init.copy()
        z = z_init.copy()
        hessian_inner_product_h_z_yy = self.hessian_inner_product_h_z_yy(x, y, z)
        L_yy = self.problem.hessian_g_yy(x, y) + hessian_inner_product_h_z_yy
        hadamard_matrix_z_y = self.compute_z_hadamard_grad_y_h(x, y, z)
        constraint_vector = self.constraint_vector(x, y)
        constraint_matrix = np.diag(constraint_vector)

        num_constraints = self.problem.num_constraints_h1 + self.problem.num_constraints_h2
        H_y = np.zeros((num_constraints, self.problem.n))
        for i in range(self.problem.num_constraints_h1):
            H_y[i, :] = self.problem.gradient_h_1_y(x, y, i)
        for i in range(self.problem.num_constraints_h2):
            H_y[self.problem.num_constraints_h1 + i, :] = self.problem.gradient_h_2_y(x, y, i)
        
        combined_matrix = np.block([
            [L_yy, hadamard_matrix_z_y],
            [H_y, constraint_matrix]
        ])
        return combined_matrix

    def Lagrangian_l(self, x, y_init, z_init, start_time, max_elapsed_time):
        z = z_init.copy()
        y = y_init.copy()
        for outer_iter in range (self.max_iters_z):
            if outer_iter == 0:
                z_mid = z
            else:
                z_mid = z + (outer_iter - 1) / (outer_iter + 2) * (z - z_old)

            for inner_iter in range (self.max_iters_y):
                gradient_L_g_y = self.problem.gradient_g_y(x,y) + self.compute_inner_product_y(x, y, z_mid)
                if np.linalg.norm(gradient_L_g_y) <= self.epsilon_y:
                    # print(f"Inner loop for y converges when iter = {inner_iter}")
                    break
                # print(f"Inner iter for y={inner_iter}, Projected Gradient norm w.r.t y={np.linalg.norm(gradient_L_g_y)}")
                y = y - self.alpha_y * gradient_L_g_y
           
           
            z_new = z_mid + self.alpha_z * self.constraint_vector(x, y)
            z_new = np.maximum(z_new, 0)
            z_old = z
            
            
            if (np.linalg.norm(z_new - z_old) / self.alpha_z) <= self.epsilon_z:
                # print(f"Outer loop for z converges when iter = {outer_iter}")
                break
            # print(f"Outer iter for z={outer_iter}, Projected Gradient norm w.r.t z={(np.linalg.norm(z_new - z_old) / self.alpha_z)}")
            z = z_new
            elapsed_time = time.time() - start_time
            if elapsed_time > max_elapsed_time:
                print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting loop.")
                break
        return y, z
    
    def check_multiplier(self, x, y, z):
        for i in range(self.problem.num_constraints_h1):
            hi_val = self.problem.h_1(x, y, i)
            if abs(hi_val) <= 1e-10:
                print(f"h1 {i}-th constraint is active, multiplier: {z[i]}")
            
        for i in range(self.problem.num_constraints_h2):
            hi_val = self.problem.h_2(x, y, i)
            if abs(hi_val) <= 1e-10:
                print(f"h2 {i}-th constraint is active, multiplier: {z[i + self.problem.num_constraints_h2]}")

    def bsg(self, x_init, y_init, z_init, max_elapsed_time):
        x = x_init.copy()
        y = y_init.copy()
        z = z_init.copy()
        history = []
        start_time = time.time()
        L = np.hstack((np.eye(self.problem.n), np.zeros((self.problem.n, self.problem.num_constraints_h1 + self.problem.num_constraints_h2)))).T

        for iter in range (self.max_iters_x):
            [y, z] = self.Lagrangian_l(x, y, z, start_time, max_elapsed_time)

            f_x = self.problem.gradient_f_x(x, y)
            G_x = self.G_x(x, y, z)
            G_v = self.G_v(x, y, z)
            # print(f"Condition number of G_v: {np.linalg.cond(G_v)}")
            # self.check_multiplier(x, y, z)
            G_v_inv = np.linalg.inv(G_v)
            f_y = self.problem.gradient_f_y(x, y)

            Grad = -(f_x - G_x @ G_v_inv @ L @ f_y)

            if np.linalg.norm(Grad) < self.epsilon_x:
                print("Main loop converged at iteration", iter)
                break
            elapsed_time = time.time() - start_time 
            if elapsed_time > max_elapsed_time:
                print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting loop.")
                break
            x_new = x - (self.alpha_x / np.sqrt(iter + 1)) * Grad
            x= x_new
 
            f_value = self.problem.f(x, y)
            history.append({
                'iteration': iter,
                'x': x.copy(),
                'y': y.copy(),
                'f_value': f_value,
                'grad_norm': np.linalg.norm(Grad),
                'time': elapsed_time
            })
            print(f"f(x, y) = {f_value}, grad_norm of hyperfunction= {np.linalg.norm(Grad)}")
            
        return x, y, history
