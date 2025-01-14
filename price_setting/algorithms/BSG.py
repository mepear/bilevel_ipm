import numpy as np
import cvxpy as cp
import time

class BSG:
    def __init__(self, problem, hparams):
        self.problem = problem
        BSG_params = hparams.get('BSG', {})

        self.alpha_T = BSG_params.get('alpha_T', 1e-4)
        self.alpha_xy = BSG_params.get('alpha_xy', 1e-4)
        self.alpha_z = BSG_params.get('alpha_z', 1e-4)
        self.epsilon_T = BSG_params.get('epsilon_T', 1e-4)
        self.epsilon_xy = BSG_params.get('epsilon_xy', 1e-4)
        self.epsilon_z = BSG_params.get('epsilon_z', 1e-4)
        self.max_iters_T = BSG_params.get('max_iters_T', 10000)
        self.max_iters_xy = BSG_params.get('max_iters_xy', 10000)
        self.max_iters_z = BSG_params.get('max_iters_z', 10000)
        
    def project_to_constraints(self, T, x_init, y_init):
        n = self.problem.n
        x_var = cp.Variable(n)
        y_var = cp.Variable(n)

        objective = cp.Minimize(cp.sum_squares(x_var - x_init) + cp.sum_squares(y_var - y_init))

        constraints = []
        M = self.M

        for i in range(self.problem.num_constraints_h1):
            constraints.append(self.problem.h_1(T, x_var, y_var, i) <= -M)
        for i in range(self.problem.num_constraints_h2):
            constraints.append(self.problem.h_2(T, x_var, y_var, i) <= -M)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP)
        if prob.status not in ["optimal", "optimal_inaccurate"] or x_var.value is None or y_var.value is None:
            # print(f"Projection failed with status: {prob.status}")
            return x_init, y_init
        return x_var.value, y_var.value
    
    def constraint_vector(self, T, x, y):
        constraints = []
    
        for i in range(self.problem.num_constraints_h1):
            constraints.append(self.problem.h_1(T, x, y, i))
    
        for i in range(self.problem.num_constraints_h2):
            constraints.append(self.problem.h_2(T, x, y, i))
    
        constraint_vector = np.array(constraints)
        return constraint_vector

    def compute_inner_product_xy(self, T, x, y, z_vector):
        num_constraints_h1 = self.problem.num_constraints_h1
        num_constraints_h2 = self.problem.num_constraints_h2
        n = self.problem.n
        total_constraints = num_constraints_h1 + num_constraints_h2
        H_xy = np.zeros((total_constraints, 2 * n))
        
        for i in range(num_constraints_h1):
            (grad_x, grad_y) = self.problem.gradient_h_1_xy(T, x, y, i)
            grad_xy = np.concatenate([grad_x, grad_y])
            H_xy[i, :] = grad_xy
        
        for i in range(num_constraints_h2):
            (grad_x, grad_y) = self.problem.gradient_h_2_xy(T, x, y, i)
            grad_xy = np.concatenate([grad_x, grad_y])
            H_xy[num_constraints_h1 + i, :] = grad_xy
        
        z_grad_xy = H_xy.T @ z_vector
        
        return z_grad_xy

    def hessian_inner_product_h_z_T_xy(self, T, x, y, z):
        hessian = np.zeros((self.problem.n, 2 * self.problem.n))

        for i in range(self.problem.num_constraints_h1):
            hessian_matrix = self.problem.hessian_h_1_T_xy(T, x, y, i)
            hessian += hessian_matrix * z[i]

        return hessian
    
    def hessian_inner_product_h_z_xy_xy(self, T, x, y, z):
        hessian = np.zeros((2 * self.problem.n, 2 * self.problem.n))

        for i in range(self.problem.num_constraints_h1):
            hessian += self.problem.hessian_h_1_xy_xy(T, x, y, i) * z[i]

        return hessian

    def compute_z_hadamard_grad_T_h(self, T, x, y, z):
        num_constraints = self.problem.num_constraints_h1 + self.problem.num_constraints_h2
        H_T = np.zeros((num_constraints, self.problem.n))

        for i in range(self.problem.num_constraints_h1):
            H_T[i, :] = self.problem.gradient_h_1_T(T, x, y, i)

        for i in range(self.problem.num_constraints_h2):
            H_T[self.problem.num_constraints_h1 + i, :] = self.problem.gradient_h_2_T(T, x, y, i)

        H_T_T = H_T.T

        z = z.reshape(1, -1)
        hadamard_matrix = H_T_T * z

        return hadamard_matrix
    
    def compute_z_hadamard_grad_xy_h(self, T, x, y, z):
        num_constraints = self.problem.num_constraints_h1 + self.problem.num_constraints_h2
        H_xy = np.zeros((num_constraints, 2 * self.problem.n))

        for i in range(self.problem.num_constraints_h1):
            grad_x, grad_y = self.problem.gradient_h_1_xy(T, x, y, i)
            H_xy[i, :] = np.concatenate([grad_x, grad_y])

        for i in range(self.problem.num_constraints_h2):
            grad_x, grad_y = self.problem.gradient_h_2_xy(T, x, y, i)
            H_xy[self.problem.num_constraints_h1 + i, :] = np.concatenate([grad_x, grad_y])

        H_xy_T = H_xy.T

        z = z.reshape(1, -1)
        hadamard_matrix = H_xy_T * z

        return hadamard_matrix
    
    def G_T(self, T_init, x_init, y_init, z_init):
        T = T_init.copy()
        x = x_init.copy()
        y = y_init.copy()
        z = z_init.copy()
        hessian_inner_product_h_z_T_xy = self.hessian_inner_product_h_z_T_xy(T, x, y, z)
        L_T_xy = self.problem.hessian_g_T_xy(T, x, y) + hessian_inner_product_h_z_T_xy
        hadamard_matrix_z_T = self.compute_z_hadamard_grad_T_h(T, x, y, z)
        concatenated_matrix = np.hstack((L_T_xy, hadamard_matrix_z_T))

        return concatenated_matrix
    
    def G_v(self, T_init, x_init, y_init, z_init):
        T = T_init.copy()
        x = x_init.copy()
        y = y_init.copy()
        z = z_init.copy()
        hessian_inner_product_h_z_xy_xy = self.hessian_inner_product_h_z_xy_xy(T, x, y, z)
        L_xy_xy = self.problem.hessian_g_xy_xy(T, x, y) + hessian_inner_product_h_z_xy_xy
        hadamard_matrix_z_xy = self.compute_z_hadamard_grad_xy_h(T, x, y, z)
        constraint_vector = self.constraint_vector(T, x, y)
        constraint_matrix = np.diag(constraint_vector)

        num_constraints = self.problem.num_constraints_h1 + self.problem.num_constraints_h2
        H_xy = np.zeros((num_constraints, 2 * self.problem.n))
        for i in range(self.problem.num_constraints_h1):
            grad_x, grad_y = self.problem.gradient_h_1_xy(T, x, y, i)
            H_xy[i, :] = np.concatenate([grad_x, grad_y])
        for i in range(self.problem.num_constraints_h2):
            grad_x, grad_y = self.problem.gradient_h_2_xy(T, x, y, i)
            H_xy[self.problem.num_constraints_h1 + i, :] = np.concatenate([grad_x, grad_y])
        
        combined_matrix = np.block([
            [L_xy_xy, hadamard_matrix_z_xy],
            [H_xy, constraint_matrix]
        ])
        return combined_matrix

    def Lagrangian_l(self, T, x_init, y_init, z_init, start_time, max_elapsed_time):
        z = z_init.copy()
        x = x_init.copy()
        y = y_init.copy()
        for outer_iter in range (self.max_iters_z):
            for inner_iter in range (self.max_iters_xy):
                (grad_x, grad_y) = self.problem.gradient_g_xy(T, x, y)
                grad_xy = np.concatenate([grad_x, grad_y])
                gradient_L_g_xy = grad_xy + self.compute_inner_product_xy(T, x, y, z)
                if np.linalg.norm(gradient_L_g_xy) <= self.epsilon_xy:
                    # print(f"Inner loop for xy converges when iter = {inner_iter}")
                    break
                # print(f"Inner iter for xy={inner_iter}, Projected Gradient norm w.r.t xy={np.linalg.norm(gradient_L_g_xy)}")
                x, y = x - self.alpha_xy * gradient_L_g_xy[:len(x)], y - self.alpha_xy * gradient_L_g_xy[len(x):]
           
            stepsize = self.alpha_z #/ np.sqrt(outer_iter + 1)
            z_new = z + stepsize * self.constraint_vector(T, x, y)
            z_new = np.maximum(z_new, 0)
            
            
            if (np.linalg.norm(z_new - z) / stepsize) <= self.epsilon_z:
                # print(f"Outer loop for z converges when iter = {outer_iter}")
                break
            # print(f"Outer iter for z={outer_iter}, Function value ={self.problem.g(T, x, y)+ z.T @ self.constraint_vector(T, x, y)}, Projected Gradient norm w.r.t z={(np.linalg.norm(z_new - z) / stepsize)}")
            z = z_new
            elapsed_time = time.time() - start_time
            if elapsed_time > max_elapsed_time:
                # print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting loop.")
                break
        return x, y, z
    
    def bsg(self, T_init, x_init, y_init, z_init, max_elapsed_time, step_size_type="const"):
        start_time = time.time()
        T = T_init.copy()
        x = x_init.copy()
        y = y_init.copy()
        z = z_init.copy()
        x_temp, y_temp, z_temp = self.Lagrangian_l(T, x, y, z, start_time, max_elapsed_time)
        if x_temp is None and y_temp is None and z_temp is None:
            print("Time limit exceeded in Lagrangian_l. Exiting bsg.")
        else:
            x, y, z = x_temp, y_temp, z_temp
        f_T = self.problem.gradient_f_T(T, x, y)
        G_T = self.G_T(T, x, y, z)
        G_v = self.G_v(T, x, y, z)
        G_v_inv = np.linalg.inv(G_v)
        f_x, f_y = self.problem.gradient_f_xy(T, x, y)
        f_xy = np.concatenate([f_x, f_y])
        L = np.hstack((np.eye(2 * self.problem.n), np.zeros((2 * self.problem.n, self.problem.num_constraints_h1 + self.problem.num_constraints_h2)))).T
        Grad = f_T - G_T @ G_v_inv @ L @ f_xy
        history = []

        for iter in range (self.max_iters_T):
            # print(f"Main iteration {iter + 1}")
            f_value = self.problem.f(T, x, y)

            elapsed_time = time.time() - start_time
            history.append({
                'iteration': iter,
                'T': T.copy(),
                'x': x.copy(),
                'y': y.copy(),
                'f_value': f_value,
                'grad_norm': np.linalg.norm(Grad),
                'time': elapsed_time
            })
            # print(f"f(T, x, y) = {f_value}, grad_norm of hyperfunction= {np.linalg.norm(Grad)}")
            
            if np.linalg.norm(Grad) < self.epsilon_T:
                # print("Main loop converged at iteration", iter)
                break
            
            elapsed_time = time.time() - start_time 
            if elapsed_time > max_elapsed_time:
                # print(f"Time limit exceeded: {elapsed_time:.2f} seconds. Exiting loop.")
                break

            if step_size_type == "const":
                T_new = T - self.alpha_T * Grad
            elif step_size_type == "diminish":
                T_new = T - self.alpha_T / np.sqrt(iter + 1) * Grad
            else:
                raise ValueError("step_size_type can only be 'const' or 'diminish'")
            
            x_temp, y_temp, z_temp = self.Lagrangian_l(T, x, y, z, start_time, max_elapsed_time)
            if x_temp is None and y_temp is None and z_temp is None:
                print("Time limit exceeded in Lagrangian_l. Exiting bsg.")
            else:
                x, y, z = x_temp, y_temp, z_temp
            
            T = T_new
            f_T = self.problem.gradient_f_T(T, x, y)
            G_T = self.G_T(T, x, y, z)
            G_v = self.G_v(T, x, y, z)
            G_v_inv = np.linalg.inv(G_v)
            
            
            f_x, f_y = self.problem.gradient_f_xy(T, x, y)
            f_xy = np.concatenate([f_x, f_y])

            Grad = f_T - G_T @ G_v_inv @ L @ f_xy
            
            
        return T, x, y, history
